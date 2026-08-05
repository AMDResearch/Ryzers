# LeRobot

Contains everything you need to build & run a ROCm-enabled LeRobot container.

## Build & Run the Docker Container

Running the image verifies LeRobot 0.6 and performs a GPU forward/backward pass.

```bash
ryzers build lerobot
ryzers run
```

## Training and Controlling Robot Arms

For this example we use the LeRobot [SO-101](https://huggingface.co/docs/lerobot/en/so101) leader and follower arms, however you can easily swap them with a different robot arm type in the following scripts.

### 1. Reference & Config
- **Guide:** Hugging Face “Imitation Learning on Real-World Robots”  
  <https://huggingface.co/docs/lerobot/en/il_robots>  
- **`config.yaml`:**  
  - Pay attention to the TODO items - add your own `HF_TOKEN` from Hugging Face, and map your robot and video devices accordingly. Step 2. makes this simpler and more reproducible, but is optional.

**Important:** you will likely need read/write permissions enabled for the serial devices before you start the docker.

```bash
sudo chmod 666 /dev/ttyACM*
```

Once you've updated your config make sure to rebuild the lerobot docker.

```bash
ryzers build lerobot
```

After initial setup, steps 3-5 should be run inside an interactive shell of the docker container:
```
ryzers run bash
```

---

### 2. USB device mapping (optional)

Your serial and video devices may change indexes in `/dev` between sessions or when you re-plug them. To save the hassle of trying to figure out the device index every time we can map them to consistent named pointers by their serial IDs.

#### USB-serial mapping

1. **Record serial IDs**
   ```bash
   ls -l /dev/serial/by-id/
   ```
2. **Create or edit** `99-usb-serial.rules` with your favorite editor:
   ```
   sudo vim /etc/udev/rules.d/99-usb-serial.rules
   ```
   
   Add the following:
   ```ini
   SUBSYSTEM=="tty", ATTRS{serial}=="<leader-serial>",   SYMLINK+="ttyACM_leader"
   SUBSYSTEM=="tty", ATTRS{serial}=="<follower-serial>", SYMLINK+="ttyACM_follower"
   ```
   Replace `<leader-serial>` and `<follower-serial>` with the values from step 1.
3. **Reload rules & trigger udev**
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

#### USB-video mapping

1. **List webcam details**
   ```bash
   for dev in /dev/video*; do
       echo "=== $dev ==="
       udevadm info --query=all --name=$dev | grep -E "ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL|DEVPATH"
   done
   ```
2. **Create or edit** `99-usb-video.rules` with your favorite editor.
   ```
   sudo vim /etc/udev/rules.d/99-usb-video.rules
   ```
   You can use `ID_SERIAL_SHORT` from step 1. as the serial number for each device. Give the symlink any name that's meaningful to you.
   ```ini
   KERNEL=="video[0-9]*", SUBSYSTEM=="video4linux", ATTRS{serial}=="<cam1-serial-short>", SYMLINK+="webcam_top"
   KERNEL=="video[0-9]*", SUBSYSTEM=="video4linux", ATTRS{serial}=="<cam2-serial-short>", SYMLINK+="webcam_front"
   ```
3. **Install & update devices**
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

Your USB serial ports and cameras should mount exactly as specified. E.g. the robot and teleop ports will be available as `/dev/ttyACM_leader` and `/dev/ttyACM_follower`.

---

### 3. Collecting dataset

In order to train the policy we will need data for our specific embodiment. We use an SO-101 setup with two C270 USB webcams, however you can use more or less cameras.

#### Teleoperation (optional)

Before starting any data collection tasks you can make sure your setup works by running `lerobot-teleoperate`. We will re-use a lot of these parameters in `lerobot-record`.

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_follower \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_leader \
    --teleop.id=my_awesome_leader_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/webcam_top, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: /dev/webcam_front, width: 640, height: 480, fps: 30}}" \
    --display_data=true
```

#### Record a dataset

We record 30 episodes of manually placing a green cube into a mug. Make sure to set `robot.cameras` with the resolution and index according to your setup. Adjust dataset parameters like number of episodes or durations as needed for your task.

Set `dataset.push_to_hub=true` if you want to upload the dataset to Hugging Face.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_follower \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_leader \
    --teleop.id=my_awesome_leader_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/webcam_top, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: /dev/webcam_front, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=${HF_USER}/cube_test_dataset \
    --dataset.num_episodes=30 \
    --dataset.single_task="place green cube in mug" \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=5 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --play_sounds=false
```

You can later visualize individual episodes in your collected dataset using `lerobot-dataset-viz`. You can do this with local cached dataset - no need to upload anything online.
```bash
lerobot-dataset-viz \
    --repo-id=${HF_USER}/cube_test_dataset \
    --episode-index=0
```

<img src="images/lerobot_record_dataset_so101.gif">

### 4. Train a policy

Using the collected dataset you can use it to train a policy like [ACT](https://github.com/tonyzhaozh/act) or [pi0](https://www.physicalintelligence.company/blog/pi0). Depending on your dataset size you should be able to train a small policy like ACT within a couple hours on the Strix Halo iGPU. Adjust training parameters as required for your policy and dataset.

```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/cube_test_dataset \
    --policy.type=act \
    --output_dir=/ryzers/mounted/outputs/train/place_cube_act \
    --job_name=place_cube \
    --policy.device=cuda \
    --policy.repo_id=${HF_USER}/place_cube_act \
    --steps=20000 \
    --save_freq=2000
```

### 5. Run inference

Use `lerobot-rollout` to deploy the trained policy:

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=/ryzers/mounted/outputs/train/place_cube_act/checkpoints/last/pretrained_model/ \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_follower \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/webcam_top, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: /dev/webcam_front, width: 640, height: 480, fps: 30}}" \
    --task="place green cube in mug" \
    --duration=60
```

Observe your arm doing its tasks autonomously!

<img src="images/evaluating_act_policy_so101.gif">

Now you are well equipped to run the LeRobot stack on your Strix Halo machine. Try tackling different tasks, collect more data or explore other policies - have fun!

## Troubleshooting

### Arms out of sync

If there's a big difference between movements of the leader and follower you can re-run calibration:
```bash
lerobot-calibrate  --teleop.type=so101_leader     --teleop.port=/dev/ttyACM_leader     --teleop.id=my_awesome_leader_arm
lerobot-calibrate  --robot.type=so101_follower    --robot.port=/dev/ttyACM_follower    --robot.id=my_awesome_follower_arm
```

### Timeouts

If you run into motor bus timeout issues, you may need to increase the number of communication retries, here's a oneliner to make that change from an interactive session:
```bash
find . -type f -name "*.py" -exec sed -i.bak 's/num_retry: int = 0/num_retry: int = 10/g' {} +
```

### Camera issues

If the 2nd camera doesn't connect and you see something like:
```bash
RuntimeError: OpenCVCamera(/dev/webcam_top) read failed (status=False).
```

It might be a USB controller bandwidth limitation - try connecting cameras to different usb controllers or reduce resolution/fps.
