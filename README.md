# nanoGRPO

Full, self-contained RLHF playground built on top of nanoGPT.

## Overview

This repository provides a minimal implementation of Reinforcement Learning from Human Feedback (RLHF) using two approaches:
1. Gradient-Reweighted Preference Optimization (GRPO)
2. Proximal Policy Optimization (PPO)

The entire implementation is pure PyTorch with no dependency on external RL libraries.

## Directory Layout

```
nanogrpo/
├── README.md
├── grpo.py               # GRPO loss (token-level advantage)              
├── ppo.py                # PPO-clip loss + value head                    
├── reward_model.py       # Tiny transformer Reward Model                 
├── train_grpo.py         # GRPO fine-tune driver                         
├── train_ppo.py          # PPO fine-tune driver
├── model.py              # Minimal GPT model implementation
├── rlhf_demo.py          # Self-contained single-file demo  
├── run_example.py        # Full example with dummy data                      
├── utils/
│   ├── __init__.py
│   ├── metrics.py        # CSV logger                                    
│   └── plot_metrics.py   # Rewards / entropy / advantage curves          
├── config/
│   ├── grpo_shakespeare.json
│   └── ppo_shakespeare.json
└── requirements.txt
``` 