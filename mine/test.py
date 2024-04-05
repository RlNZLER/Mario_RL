import warnings
import csv
import sys
import gym
import time
import pickle
import random
import datetime
import optuna
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3.common import atari_wrappers

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global settings
ALGORITHM = "A2C"  # Choose from "DQN", "A2C", "PPO"
N_TRIALS = 10
N_TIMESTEPS = int(2e4)
N_EVAL_EPISODES = 5
ENV_ID = "SuperMarioBros2-v1"
EVAL_FREQ = 4000  # Frequency of evaluations within each trial

class OptunaCallback(BaseCallback):
    def __init__(self, trial, eval_env, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, verbose=1):
        super(OptunaCallback, self).__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
            self.trial.report(mean_reward, self.n_calls // self.eval_freq)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
            if self.trial.should_prune():
                return False  # Stop the trial by returning False
        return True

def log_training_results(algorithm, hyperparameters, mean_reward, std_reward, training_time):
    log_filename = "hyperparameter_log.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        file.seek(0, 2)
        if file.tell() == 0:
            writer.writerow(["Timestamp", "Algorithm", "Hyperparameters", "Mean Reward", "Std Reward", "Training Time"])
        writer.writerow([timestamp, algorithm, str(hyperparameters), mean_reward, std_reward, training_time])

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

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.9999)
    env = make_env(ENV_ID, random.randint(0, 1000))
    eval_env = make_env(ENV_ID, random.randint(1001, 2000))  # Separate eval environment

    if ALGORITHM == "DQN":
        exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1)
        BUFFER_SIZE = 10000
        model = DQN("MlpPolicy", env, verbose=0, gamma=gamma, learning_rate=learning_rate,
                    exploration_final_eps=exploration_final_eps, buffer_size=BUFFER_SIZE)
    elif ALGORITHM == "A2C":
        n_steps = trial.suggest_int('n_steps', 5, 256)
        model = A2C("MlpPolicy", env, verbose=0, gamma=gamma, n_steps=n_steps, learning_rate=learning_rate)
    elif ALGORITHM == "PPO":
        n_steps = trial.suggest_int('n_steps', 64, 2048, log=True)
        ent_coef = trial.suggest_float('ent_coef', 0.00000001, 0.1, log=True)
        batch_size = trial.suggest_int('batch_size', 8, 256)
        model = PPO("MlpPolicy", env, verbose=0, gamma=gamma, learning_rate=learning_rate,
                    n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef)

    callback = OptunaCallback(trial, eval_env)
    start_time = time.time()
    model.learn(total_timesteps=N_TIMESTEPS, callback=callback)
    training_time = time.time() - start_time

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES)
    hyperparameters = {
        'learning_rate': learning_rate,
        'gamma': gamma,
        'exploration_final_eps': exploration_final_eps if ALGORITHM == "DQN" else None,
        'n_steps': n_steps if ALGORITHM in ["A2C", "PPO"] else None,
        'ent_coef': ent_coef if ALGORITHM == "PPO" else None,
    }
    log_training_results(ALGORITHM, hyperparameters, mean_reward, std_reward, training_time)

    env.close()
    eval_env.close()

    return mean_reward

if __name__ == "__main__":
    start_time = time.time()
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    elapsed_time = time.time() - start_time

    print('\nElapsed time for {} trials: {:.2f} seconds'.format(N_TRIALS, elapsed_time))
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
