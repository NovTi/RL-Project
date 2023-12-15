# This is source code for the final project of course Reinforcement Learning



## Training Instruction
We train all the algorithms on NTY Greene HPC.


```
sbatch scripts/ppo_train.sh ppo_lr05_ent15_cp2 0
```
For example, if you want to train the PPO algorithm in NYU HPC with the above command. `ppo_lr05_ent15_cp2` means the learning rate $5 \cdot 1e^{-5}$, entropy coefficient  $15 \cdot 0.001$, clip range $2\cdot 0.1$. The last parameter `0` means seed is 0.
