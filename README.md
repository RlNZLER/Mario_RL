# Advanced Machine Learning
## Task 2
You are required to use Machine Learning to tackle the problem of “Game Learning”. Your goal in this task is to train Deep Reinforcement Learning (DRL) agents that receive image inputs from a game simulator, and that output game actions to play the game autonomously.
The following simulator, but only version SuperMarioBros2-v1, will be used to play the game:

[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)

[Double Deep Q-Networks](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

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
12. pip install -r requirements.txt

Here's a suggested approach to tackle this assignment:

1. **Choose DRL Algorithms**: Start by selecting at least three different DRL algorithms or methodologies for training your agents. These could include popular algorithms such as Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and Deep Deterministic Policy Gradients (DDPG), among others. Each algorithm may have different strengths and weaknesses, so it's beneficial to explore a variety of approaches.

2. **Design State Representations**: Experiment with different state representations for your agents. You can try using Convolutional Neural Networks (CNNs) to process image inputs directly, or you can explore other techniques such as Transformers for sequence-to-sequence learning. You may also consider combining CNNs with Transformers for more sophisticated representations.

3. **Experiment with Hyperparameters**: Tune the hyperparameters of your models, including learning rates, discount factors, exploration rates (for epsilon-greedy exploration), network architectures, and any algorithm-specific parameters. Hyperparameter tuning plays a crucial role in the performance of your agents, so it's important to experiment with different settings.

4. **Train Multiple Agents with Multiple Seeds**: Train each of your selected agents with multiple seeds to account for randomness in the training process. This helps reduce the variance in performance and provides more reliable results. Average the performance metrics (such as average reward, game score, steps per episode, etc.) across different seeds to get a more robust evaluation of each agent.

5. **Evaluate and Compare Agents**: Evaluate the performance of each agent using various metrics, including average reward, game score, steps per episode, training and test times, and any other relevant metrics you consider. Compare the performance of different agents and justify your choices in terms of architectures, hyperparameters, and algorithms.

6. **Document and Report Findings**: Document your experiments, including details of the architectures, hyperparameters, training methodologies, and results obtained by each agent. Provide visualizations of learning curves and performance metrics to support your findings. Discuss the strengths and weaknesses of each agent and make recommendations for the best-performing agent based on your evaluation.

7. **Acknowledge Resources Used**: Make sure to acknowledge any libraries, frameworks, or resources you used in your experiments, such as PFRL, StableBaselines, or other DRL libraries. Mention any additional readings or resources that influenced your approach.

By following these steps and conducting thorough experiments, you'll be able to fulfill the requirements of your assignment and gain valuable insights into training DRL agents to play the Super Mario Bros game autonomously. Good luck with your assignment! If you have any further questions or need assistance along the way, feel free to ask.
