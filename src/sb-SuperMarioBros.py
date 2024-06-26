#####################################################
# sb-SuperMarioBros.py
#
# This program trains deep reinforcement learning agents to solve the decision-making
# problem of the the so-called SuperMarioBros. See details in the following link: 
# https://www.gymlibrary.dev/environments/box2d/lunar_lander/
#
# Although the dependencies have been installed in the PCs of the labs. You may 
# should be aware the following in case of installation in your own PCs:
# pip install tensorflow[and-cuda]==2.15.1

# pip install nes-py
# pip install gym-super-mario-bros==7.3.0
# pip install setuptools==65.5.0 "wheel<0.40.0"
# pip install gym==0.21.0
# pip install gymnasium
# pip install stable-baselines3[extra]==1.8.0
# pip install sb3_contrib==1.8.0

# Version: 1.0 -- extended version from sb-LunarLander.py
# Date: 10 March 2024
# Contact: hcuayahuitl@lincoln.ac.uk
#####################################################

import os
import csv
import sys
import gym
import time 
import pickle
import random
import datetime
import subprocess
import webbrowser
from typing import Callable
from stable_baselines3 import DQN,A2C,PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3.common import atari_wrappers


if len(sys.argv) not in [3, 4]:
    print("USAGE: sb-SuperMarioBros2-v1.py (train|test) (DQN|A2C|PPO) [seed_number]")
    exit(0)
    
    
# Path to the TensorBoard log directory
tensorboard_log_dir = "tensorboard_log"
    
# Start TensorBoard as a background process
tensorboard_process = subprocess.Popen(
    ['tensorboard', '--logdir', tensorboard_log_dir],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# Optionally, open a web browser automatically pointing to TensorBoard
webbrowser.open("http://localhost:6006/", new=2)

print("TensorBoard is running. Visit http://localhost:6006/ in your web browser.")

# It's generally a good idea to give TensorBoard a few seconds to start up before the training starts
time.sleep(5)

def log_training_results(algorithm, seed, learning_rate, gamma, num_training_steps, mean_reward, std_reward, avg_game_score, training_time, testing_time):
    # log training results to CSV
    log_filename = "training_log.csv"
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write results to CSV
    with open(log_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Check if the file is empty
        file.seek(0, 2)
        if file.tell() == 0:
            # If empty, write header
            writer.writerow(["Timestamp", "Algorithm", "Seed", "Learning Rate","Gamma", "No. of training steps", "Mean Reward", "Std Reward", "Avg Game Score", "Training Time", "Testing Time"])
        # Write data to CSV
        writer.writerow([timestamp, algorithm, seed, learning_rate, gamma, num_training_steps, mean_reward, std_reward, avg_game_score, training_time, testing_time])


environmentID = "SuperMarioBros2-v1"
trainMode = True if sys.argv[1] == 'train' else False
learningAlg = sys.argv[2] 
# seed = random.randint(0,1000) if trainMode else int(sys.argv[3])
seed = int(sys.argv[3])
if not os.path.exists("policy"):
    os.makedirs("policy")
policyFileName = "policy/"+learningAlg+"-"+environmentID+"-seed"+str(seed)+".policy.pkl"
num_training_steps = 100_000
num_test_episodes = 20
policy_rendering = True

# create the learning environment 
def make_env(gym_id, seed):
    env = gym_super_mario_bros.make(gym_id)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = atari_wrappers.MaxAndSkipEnv(env, 4)
    env = atari_wrappers.NoopResetEnv(env, noop_max=30)
    env = atari_wrappers.ClipRewardEnv(env)
    env.seed(seed)	
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

environment = make_env(environmentID, seed)

# create the agent's model using one of the selected algorithms
# note: exploration_fraction=0.9 means that it will explore 90% of the training steps
if learningAlg == "DQN":
    learning_rate=0.00274725624042349
    gamma=0.904697625355928
    model = DQN("CnnPolicy", environment, seed=seed, learning_rate=learning_rate, gamma=gamma, buffer_size=10000, batch_size=46, exploration_fraction=0.4071996216115, exploration_final_eps=0.0522320894370654, verbose=1, tensorboard_log="tensorboard_log/DQN_tensorboard_log")
elif learningAlg == "A2C":
    learning_rate=0.00248345618648342
    gamma=0.855424134003397
    model = A2C("CnnPolicy", environment, seed=seed, learning_rate=learning_rate, gamma=gamma, vf_coef=0.486741019618398, ent_coef=0.0247134804959192, verbose=1, tensorboard_log="tensorboard_log/A2C_tensorboard_log")
elif learningAlg == "PPO":
    learning_rate=0.0000270874382318458
    gamma=0.930044843085905
    model = PPO("CnnPolicy", environment, seed=seed, learning_rate=learning_rate, gamma=gamma, ent_coef=0.0673625262803434, verbose=1, tensorboard_log="tensorboard_log/PPO_tensorboard_log")
else:
    print("UNKNOWN learningAlg="+str(learningAlg))
    exit(0)

# train the agent or load a pre-trained one
if trainMode:
    start_time = time.time()  # Start measuring training time
    model.learn(total_timesteps=num_training_steps, progress_bar=True)
    
    end_time = time.time() # Stop measuring training time
    training_time = end_time - start_time  # Calculate training time
    
    print("Saving policy "+str(policyFileName))
    pickle.dump(model.policy, open(policyFileName, 'wb'))
else:
    print("Loading policy...")
    with open(policyFileName, "rb") as f:
        policy = pickle.load(f)
    model.policy = policy

print("Evaluating policy...")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
print("EVALUATION: mean_reward=%s std_reward=%s" % (mean_reward, std_reward))

# visualise the agent's learnt behaviour.
steps_per_episode = 0
reward_per_episode = 0
total_cummulative_reward = 0
episode = 1
start_time = time.time()  # Start measuring testing time.
env = model.get_env()
obs = env.reset()
while True and policy_rendering:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    steps_per_episode += 1
    reward_per_episode += reward
    total_game_score = info[0]['score']
    if any(done):
        print("episode=%s, steps_per_episode=%s, reward_per_episode=%s, total_game_score=%s" % (episode, steps_per_episode, reward_per_episode, total_game_score))
        total_cummulative_reward += reward_per_episode
        steps_per_episode = 0
        reward_per_episode = 0
        episode += 1
        obs = env.reset()
    env.render("human")
    if episode >= num_test_episodes:
        avg_cummulative_reward = total_cummulative_reward/num_test_episodes
        avg_game_score = total_game_score/num_test_episodes
        print("total_cummulative_reward=%s avg_cummulative_reward=%s avg_game_score=%s" % \
              (total_cummulative_reward, avg_cummulative_reward, avg_game_score))
        policy_rendering = False
        break
env.close()
end_time = time.time() # Stop measuring testing time.
testing_time = end_time - start_time  # Calculate training time.

if trainMode:
    log_training_results(learningAlg, seed, learning_rate, gamma, num_training_steps, mean_reward, std_reward, avg_game_score, training_time, testing_time)
