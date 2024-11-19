## LeRobot Docker Setup

Contains everything you need to build & run a ROCm-enabled LeRobot container.

### 1. Reference & Config
- **Guide:** Hugging Face “Getting Started with Real-World Robots”  
  <https://huggingface.co/docs/lerobot/getting_started_real_world_robot>  
- **`config.yaml`:**  
  - Replace the `TODO` items with the correct setup. This includes setting up Hugging Face for storing your dataset, the correct video configs, model_paths, etc.

---

### 2. USB-Serial Mapping 

1. **Record serial IDs**
   ```bash
   ls -l /dev/serial/by-id/
   ```
2. **Create or edit** `99-usb-serial.rules`:
   ```ini
   SUBSYSTEM=="tty", ATTRS{idVendor}=="2f5d", ATTRS{idProduct}=="2202", ATTRS{serial}=="<leader-serial>",   SYMLINK+="ttyACM_leader"
   SUBSYSTEM=="tty", ATTRS{idVendor}=="2f5d", ATTRS{idProduct}=="2202", ATTRS{serial}=="<follower-serial>", SYMLINK+="ttyACM_follower"
   ```
   Replace `<leader-serial>` and `<follower-serial>` with the values from step 1.  
3. **Install & update devices**
   ```bash
   sudo cp 99-usb-serial.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

Your cameras, model files, and USB-serial ports will now mount exactly as specified.

---

### 3. Build & Run

To verify the lerobot installation simply run the built ryzer -- this will run one of the lerobot examples as a test.
```bash
ryzers build lerobot
ryzers run
```

To run an interactive session start with bash:
```bash
ryzers run bash
```

---

### 4. Collecting Data & Running Inference

For this example we use the [koch v1.1](https://github.com/jess-moss/koch-v1-1) leader and follower arms, however you can easily swap them with different ones by changing the `robot.type` and `teleop.type` parameters.

#### Record a dataset

Make sure to set `robot.cameras` with the resolution and index according to your setup.

```bash

python -m lerobot.record \
    --robot.type=koch_follower \
    --robot.port=/dev/ttyACM_kochfollower \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: ${VIDEO_PATH_1}, width: 640, height: 480, fps: 15}, side: {type: opencv, index_or_path: ${VIDEO_PATH_2}, width: 640, height: 480, fps: 15}}" \
    --teleop.type=koch_leader \
    --teleop.port=/dev/ttyACM_kochleader \
    --teleop.id=my_awesome_leader_arm \
    --display_data=false \
    --dataset.repo_id=${HF_USER}/name_of_dataset \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick up the green block and place it in the mug" \
    --dataset.reset_time_s=5 \
    --dataset.episode_time_s=10 \
    --dataset.fps=15 \
    --play_sounds false
```

#### Train a policy

Using the collected dataset you can use it to train a policy like [ACT](https://github.com/tonyzhaozh/act) or [pi0](https://www.physicalintelligence.company/blog/pi0). If you are using an iGPU it may take a long time to train, in which case you may want to do it on a bigger GPU. Once trained you can easily use this model with an iGPU for inference. Set the policy you want to train by setting `policy.type`.

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/name_of_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_test \
  --job_name=act_test \
  --policy.device=cuda \
  --wandb.enable=false
```

#### Run inference

Make sure your model is mounted via `config.yaml`, then run the evaluation script with the `policy.path` parameter set. For evaluation `dataset.repo_id` name must start with **eval**.
```bash
python -m lerobot.record \
    --robot.type=koch_follower \
    --robot.port=/dev/ttyACM_kochfollower \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: ${VIDEO_PATH_1}, width: 640, height: 480, fps: 15}, side: {type: opencv, index_or_path: ${VIDEO_PATH_2}, width: 640, height: 480, fps: 15}}" \
    --teleop.type=koch_leader \
    --teleop.port=/dev/ttyACM_kochleader \
    --teleop.id=my_awesome_leader_arm \
    --dataset.repo_id=${HF_USER}/eval_name_of_dataset \
    --dataset.single_task="${DEPLOYMENT_TASK}" \
    --policy.path=${MODEL_CKPT_PATH} \
    --play_sounds=false
```


## Troubleshooting

If you run into motor bus timeout issues, you may need to increase the number of communication retries, here's a oneliner to make that change from an interactive session:
```bash
sed -i 's/num_retry: int = 0/num_retry: int = 5/g' lerobot/lerobot/common/motors/motors_bus.py
```

If you are using dynamixel motors as found on the koch v1.1 arms, you may need to apply drive mode so the shoulder does not move in reverse:
```bash
sed -i 's/apply_drive_mode = False/apply_drive_mode = True/g' lerobot/lerobot/common/motors/dynamixel/dynamixel.py
```
