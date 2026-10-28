# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Exercise the CLI and actual generated Bash scripts without Docker or a GPU."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ryzers.entrypoint import main
from ryzers.runner import DockerRunner

ROOT = Path(__file__).resolve().parents[1]


def documented_commands():
    """Read executable examples, including continued lines and inline commands."""
    for path in sorted(p for p in ROOT.rglob('*') if p.suffix in ('.md', '.sh')):
        if '.git' in path.parts:
            continue
        text = path.read_text(encoding='utf-8').replace('\\\n', '')
        fenced = False
        for number, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith('```'):
                fenced = not fenced
                continue
            if path.suffix == '.sh':
                candidates = [line.lstrip().lstrip('#').strip()]
            else:
                candidates = [line.strip()] if fenced else re.findall(r'`([^`\n]+)`', line)
            for candidate in candidates:
                if re.match(r'^(?:\w+=\S+\s+)*ryzers (?:run|build)(?:\s|$)', candidate):
                    yield str(path.relative_to(ROOT)), number, candidate


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ryzers-runner-test-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        previous = Path.cwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, previous)
        bindir = self.directory / 'bin'
        bindir.mkdir()
        docker = bindir / 'docker'
        docker.write_text(
            '#!' + sys.executable + '\n'
            'import json,os,sys\n'
            'from pathlib import Path\n'
            'Path(os.environ["CAPTURE"]).write_text(json.dumps({"argv":sys.argv[1:],'
            '"forwarded":os.environ.get("AUDIT_FORWARD")}))\n'
            'sys.exit(int(os.environ.get("DOCKER_STATUS", "0")))\n'
        )
        docker.chmod(0o755)
        xhost = bindir / 'xhost'
        xhost.write_text('#!/bin/sh\nexit 0\n')
        xhost.chmod(0o755)
        self.capture = self.directory / 'capture.json'
        self.env = patch.dict(os.environ, {
            'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
            'CAPTURE': str(self.capture), 'AUDIT_FORWARD': 'test value with spaces',
            'DOCKER_STATUS': '0',
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        # A wildcard must not accidentally expand against the host directory.
        (self.directory / 'host-only.glb').touch()

    def script(self, legacy=False, image='audit-image', filename=None):
        runner = DockerRunner(image, script_name=filename)
        runner.build_runscript('--env AUDIT_FORWARD --volume "$PWD/host dir:/data"')
        path = Path(runner.script_name)
        if legacy:
            path.write_text(path.read_text().replace(' "$@"\n', ' $1\n'))
        return runner, path.read_bytes()

    def received(self, image='audit-image'):
        capture = json.loads(self.capture.read_text())
        argv = capture['argv']
        self.assertEqual(argv[:6], [
            'run', '--env', 'AUDIT_FORWARD', '--volume',
            str(self.directory / 'host dir') + ':/data', image,
        ])
        self.assertEqual(capture['forwarded'], 'test value with spaces')
        command = argv[6:]
        if len(command) >= 3 and command[0] in ('python', 'python3') and command[1] == '-c':
            compile(command[2], '<Python command override>', 'exec')
        return command

    def cli(self, args):
        with patch.object(sys, 'argv', ['ryzers', *args]), patch.object(
                DockerRunner, 'get_last_container_name', return_value='audit-image'):
            main()

    def test_argument_matrix_on_new_and_existing_scripts(self):
        cases = [
            ('', []), ('   ', []), ('bash', ['bash']),
            ('python script.py --mask-index 14 3', ['python', 'script.py', '--mask-index', '14', '3']),
            ("python -c 'import torch; print(1 + 2)'", ['python', '-c', 'import torch; print(1 + 2)']),
            ("python -c 'import torch; print([(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])'", ['python', '-c', 'import torch; print([(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])']),
            ('python -c "print(\'quoted value\')"', ['python', '-c', "print('quoted value')"]),
            ("tool 'mesh exports/模型.glb' ''", ['tool', 'mesh exports/模型.glb', '']),
            (r'tool escaped\ space C:\\data\\mesh.glb', ['tool', 'escaped space', r'C:\data\mesh.glb']),
            ("tool '*.glb' '$HOME' '$(touch unexpected)' '`touch unexpected`' ';' '|' '>' '&'",
             ['tool', '*.glb', '$HOME', '$(touch unexpected)', '`touch unexpected`', ';', '|', '>', '&']),
            ('tool *.glb $HOME', ['tool', '*.glb', '$HOME']),
            ("bash -c 'printf \"%s\\n\" \"hello world\" | cat'", ['bash', '-c', 'printf "%s\\n" "hello world" | cat']),
            ("tool 'first\nsecond'", ['tool', 'first\nsecond']),
            ('tool --name=value -x -- -literal', ['tool', '--name=value', '-x', '--', '-literal']),
        ]
        for legacy in (False, True):
            for command, expected in cases:
                with self.subTest(legacy=legacy, command=command):
                    runner, original = self.script(legacy, filename='run script.sh')
                    runner.docker_cmdstr = command
                    runner()
                    self.assertEqual(self.received(), expected)
                    self.assertEqual(Path(runner.script_name).read_bytes(), original)
                    self.assertFalse((self.directory / 'unexpected').exists())

    def test_all_documented_commands_through_cli(self):
        examples = list(documented_commands())
        self.assertGreater(len(examples), 80, 'Documentation inventory unexpectedly empty')
        for source, number, command in examples:
            words = shlex.split(command.replace('<imagename>', 'named-image'), comments=True)
            assignments = {}
            while words and words[0] != 'ryzers':
                key, value = words.pop(0).split('=', 1)
                assignments[key] = value
            self.assertEqual(words.pop(0), 'ryzers')
            args = words
            if args[0] == 'build':
                with self.subTest(source=source, line=number, command=command):
                    with patch('ryzers.entrypoint.build') as build, patch.dict(os.environ, assignments):
                        self.cli(args)
                    build.assert_called_once()
                continue
            image = args[args.index('--name') + 1] if '--name' in args else 'audit-image'
            for legacy in (False, True):
                with self.subTest(source=source, line=number, legacy=legacy, command=command):
                    runner, original = self.script(legacy, image=image)
                    with patch.dict(os.environ, assignments):
                        self.cli(args)
                    override = args[1:]
                    if '--name' in override:
                        index = override.index('--name')
                        del override[index:index + 2]
                    expected = shlex.split(override[0]) if override else []
                    self.assertEqual(self.received(image), expected)
                    self.assertEqual(Path(runner.script_name).read_bytes(), original)

    def test_unbalanced_quotes_do_not_launch_docker(self):
        for legacy in (False, True):
            for command in ["python -c 'unfinished", 'tool "unfinished', 'tool trailing\\']:
                with self.subTest(legacy=legacy, command=command):
                    self.capture.unlink(missing_ok=True)
                    runner, _ = self.script(legacy)
                    runner.docker_cmdstr = command
                    with self.assertRaises(ValueError):
                        runner()
                    self.assertFalse(self.capture.exists())

    def test_direct_generated_script_takes_argument_vector(self):
        runner, _ = self.script()
        args = ['python', '-c', 'print("hello world")', '', '*.glb']
        subprocess.run(['bash', runner.script_name, *args], check=True)
        self.assertEqual(self.received(), args)

    def test_docker_failure_is_not_reported_as_success(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), patch.dict(os.environ, {'DOCKER_STATUS': '29'}):
                runner, _ = self.script(legacy)
                with self.assertRaises(subprocess.CalledProcessError) as failure:
                    runner()
                self.assertEqual(failure.exception.returncode, 29)
                with self.assertRaises(subprocess.CalledProcessError):
                    self.cli(['run'])

    def test_forwarded_programs_execute_with_the_intended_arguments(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                runner, _ = self.script(legacy)
                runner.docker_cmdstr = "python3 -c 'import sys; print(sys.argv[1])' 'mesh exports/模型.glb'"
                runner()
                output = subprocess.check_output(self.received(), text=True)
                self.assertEqual(output, 'mesh exports/模型.glb\n')
                runner.docker_cmdstr = "bash -c 'printf \"%s\" \"$AUDIT_FORWARD\" | cat'"
                runner()
                output = subprocess.check_output(self.received(), text=True)
                self.assertEqual(output, 'test value with spaces')

    def test_missing_script_does_not_launch_docker(self):
        with self.assertRaises(FileNotFoundError):
            DockerRunner('missing')()
        self.assertFalse(self.capture.exists())

    def test_custom_script_keeps_its_file_context(self):
        path = self.directory / 'custom.sh'
        path.write_text('#!/bin/bash\ncd "$(dirname "${BASH_SOURCE[0]}")"\n'
                        'docker run --env AUDIT_FORWARD --volume "$PWD/host dir:/data" audit-image "$@"\n')
        runner = DockerRunner('audit-image', "tool 'some path'", str(path))
        runner()
        self.assertEqual(self.received(), ['tool', 'some path'])


if __name__ == '__main__':
    unittest.main()
