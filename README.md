# SnakeAI
A reinforcement-learning Snake agent that uses a simple DQN-style neural network (PyTorch) to learn and play a pygame-based Snake game, logging training progress and showing real-time improvement.

## Features
- Pygame Snake environment with rewards (+10 eat, -10 death, idle cap)
- 11-feature state
- Epsilon-greedy exploration with replay memory (100k) and batch training
- Two-layer MLP with Adam + MSE
- Model checkpoint saving (`model.pth`)
- Logged training progress

## Running It
1. Clone this repository
``` Bash
git clone https://github.com/ChimkinSoup/SnakeAI.git
```
2. Install dependencies
``` Bash
pip install -r requirements.txt
```
3. Train the agaent
``` Bash
python src/agent.py
```
- The Snake game window will appear after a few seconds
- Training metrics are written to `training_plot.png` in `src/` after each game
