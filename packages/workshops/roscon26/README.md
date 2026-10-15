<img width="322" height="344" alt="image" src="https://us1.discourse-cdn.com/flex022/uploads/ros/original/3X/d/0/d09219dc514d1cee96dc88eb9fcb955d5d9faf4c.png" />


These are the materials presented at ROSCon 2026 in Toronto.  

The hands-on materials are in the folders for the 4 sections - Fine-tuning, Reinforcement Learning, Simulators, and Local Inference (Code-as-Policy.

| Section | Package | Contents |
|---------|---------|----------|
| Fine-tuning | _not yet ported_ | LoRA fine-tune MolmoAct2 on LIBERO, then drive the policy in simulation |
| Reinforcement Learning | [`roscon26-rllearning`](roscon26-rllearning) | Roll out three Brax PPO `PandaPickCube` checkpoints and watch the policy improve |
| Simulators | _not yet ported_ | Genesis / Unitree G1 humanoid locomotion, flat ground to stairs |
| Local Inference (Code-as-Policy) | _not yet ported_ | Lemonade, RAI robot agents, CaP-X, and harness optimisation |

Each section is a standalone ryzer. Build one with, for example:

```bash
ryzers build jupyterlab roscon26-rllearning
```

And the presentation itself is [here](https://github.com/AMDResearch/Ryzers/releases/download/v0.2/ROSCON_2026_FINAL.pdf). 
