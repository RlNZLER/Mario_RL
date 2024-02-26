# Advanced Machine Learning
## Task 2
You are required to use Machine Learning to tackle the problem of “Game Learning”. Your goal in this task is to train Deep Reinforcement Learning (DRL) agents that receive image inputs from a game simulator, and that output game actions to play the game autonomously.
The following simulator, but only version SuperMarioBros2-v1, will be used to play the game:

https://github.com/Kautenja/gym-super-mario-bros

You are required to use your knowledge acquired in the module regarding DRL agents, and knowledge acquired from additional recommended readings. This will be useful to investigate the performance of those agents and to compare and criticise them so you can recommend your best agent. You are expected to evaluate your agents using metrics such as Avg. Reward, Avg. Game Score, Avg. Steps Per Episode, and Training and Test Times – and any others that you consider relevant. You are expected to train at least three different agents, which can differ in their state representation (CNN, Transformer, CNN-Transformer) and/or different learning algorithms or training methodologies. Once you have decided on the agents that you want to report, you should train them with multiple seeds and average their results—to reduce the potential noise (due to randomness) in the performance of your models. If you report learning curves, they should be based on those average results instead of using a single seed (run). You are expected to justify your choices in terms of architectures, hyperparameters and algorithms. 

In this assignment, you are free to train any DRL agent, in any programming language, to preprocess the data, and to implement your solutions whenever possible. While you are free to use libraries such as PFRL, StableBaselines or Pearl (among others), you should mention your resources used, acknowledge them appropriately, and compare between agents in your report.

Please read the Criterion Reference Grid for details on how your work will be graded.

Steps:
1. python3 -m venv env
2. source env/bin/activate
3. sudo apt-get update
4. sudo apt-get install python3
5. pip install --upgrade pip
6. pip install nes-py
7. pip install gym-super-mario-bros==7.3.0
8. pip install jinja2 pyyaml typeguard
8. pip install setuptools==65.5.0 "wheel<0.40.0"
9. pip install gym==0.21.0
10. pip install stable-baselines3[extra]==1.6.0
11. pip freeze > requirements.txt
