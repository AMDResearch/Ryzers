import os
import sys

sys.path.insert(
    0, os.environ.get("RHO_SUPPORT_ROOT", "/ryzers/notebooks/scripts")
)
from rho_multitask_demo import evaluate_cli

raise SystemExit(evaluate_cli())
