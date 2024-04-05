#####################################################
# Optimized sb-SuperMarioBros.py with Global Algorithm Selector
#####################################################

import warnings
import csv
import sys
import gym
import time
import pickle
import random
import datetime
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv, ClipRewardEnv

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global settings
ALGORITHM = "DQN"  # Choose from "DQN", "A2C", "PPO"
N_TRIALS = 2
N_TIMESTEPS = int(2e4)
N_EVAL_EPISODES = 5
ENV_ID = "SuperMarioBros2-v1"
    
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
    env = MaxAndSkipEnv(env, 4)
    env = NoopResetEnv(env, noop_max=30)
    env = ClipRewardEnv(env)
    env.seed(seed)	
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.9999)
    env = make_env(ENV_ID, random.randint(0, 1000))
    
    if ALGORITHM == "DQN":
        exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1)
        BUFFER_SIZE = 10000
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            gamma=gamma,
            learning_rate=learning_rate,
            exploration_final_eps=exploration_final_eps,
            buffer_size=BUFFER_SIZE
        )
    elif ALGORITHM == "A2C":
        n_steps = trial.suggest_int('n_steps', 5, 256)
        model = A2C(
            "MlpPolicy", 
            env, 
            verbose=0, 
            gamma=gamma, 
            n_steps=n_steps, 
            learning_rate=learning_rate
        )
    elif ALGORITHM == "PPO":
        n_steps = trial.suggest_int('n_steps', 64, 2048, log=True)
        ent_coef = trial.suggest_float('ent_coef', 0.00000001, 0.1, log=True)  # Adjusted for consistency
        batch_size = trial.suggest_int('batch_size', 8, 256) 
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            gamma=gamma,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,  # Use if optimizing
            ent_coef=ent_coef
        )
    else:
        raise ValueError("Unsupported algorithm specified.")

    start_time = time.time()
    model.learn(total_timesteps=N_TIMESTEPS)
    training_time = time.time() - start_time
    
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=N_EVAL_EPISODES)
    
    hyperparameters = {
        'learning_rate': learning_rate,
        'gamma': gamma,
        'exploration_final_eps': exploration_final_eps if ALGORITHM == "DQN" else None,
        'n_steps': n_steps if ALGORITHM in ["A2C", "PPO"] else None,
        'ent_coef': ent_coef if ALGORITHM == "PPO" else None,
    }
    log_training_results(ALGORITHM, hyperparameters, mean_reward, std_reward, training_time)
    
    env.close()
    
    return mean_reward

if __name__ == "__main__":
    start_time = time.time()
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    elapsed_time = time.time() - start_time
    
    print(f'\nElapsed time for {N_TRIALS} trial: {elapsed_time:.2f} seconds')
    print(f'Number of finished trials: {len(study.trials)}')
    print(f'Best trial: {study.best_trial.params}')
    
    # Plot the optimization history
    optimization_history = plot_optimization_history(study)
    optimization_history.show()

    # Plot the importance of hyperparameters
    param_importances = plot_param_importances(study)
    param_importances.show()

    # Plot individual hyperparameters or combinations and their effects on the objective
    slice_plot = plot_slice(study)
    slice_plot.show()