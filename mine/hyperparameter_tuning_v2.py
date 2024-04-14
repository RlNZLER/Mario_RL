################################################################
# Hyperparameter tuning for sb-SuperMarioBros.py version 2
################################################################

import warnings
import csv
import sys
import gym
import time
import pickle
import random
import datetime
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    NoopResetEnv,
    ClipRewardEnv,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global settings
ALGORITHM = "PPO"  # Choose from "DQN", "A2C", "PPO"
N_TRIALS = 50
N_TIMESTEPS = int(2e4)
N_EVAL_EPISODES = 5
ENV_ID = "SuperMarioBros2-v1"


def log_training_results(
    algorithm, hyperparameters, mean_reward, std_reward, training_time
):
    log_filename = "hyperparameter_log.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        file.seek(0, 2)
        if file.tell() == 0:
            writer.writerow(
                [
                    "Timestamp",
                    "Algorithm",
                    "Hyperparameters",
                    "Mean Reward",
                    "Std Reward",
                    "Training Time",
                ]
            )
        writer.writerow(
            [
                timestamp,
                algorithm,
                str(hyperparameters),
                mean_reward,
                std_reward,
                training_time,
            ]
        )


def calculate_batch_size(n_steps, min_batch_size=8, max_batch_size=256):
    total_steps = n_steps
    # Start with the max batch size and decrease until finding a divisor or reaching the min batch size
    for potential_batch_size in range(max_batch_size, min_batch_size - 1, -1):
        if total_steps % potential_batch_size == 0:
            return potential_batch_size
    # If no divisor is found within bounds, return the minimum batch size
    return min_batch_size


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
    # Common hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.9999)
    env = make_env(ENV_ID, random.randint(0, 1000))

    if ALGORITHM == "DQN":
        buffer_size = trial.suggest_int("buffer_size", 10000, 100000)
        batch_size = trial.suggest_int("batch_size", 32, 256)
        train_freq = trial.suggest_int("train_freq", 1, 100)
        target_update_interval = trial.suggest_int("target_update_interval", 100, 5000)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)

        model = DQN(
            "CnnPolicy", 
            env,
            verbose=0,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=1000,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=train_freq,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            tensorboard_log=None
        )
    elif ALGORITHM == "A2C":
        n_steps = trial.suggest_int("n_steps", 5, 256)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
        ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1)

        model = A2C(
            "CnnPolicy", 
            env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            tensorboard_log=None
        )
    elif ALGORITHM == "PPO":
        n_steps = trial.suggest_int("n_steps", 64, 2048, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
        n_epochs = trial.suggest_int("n_epochs", 1, 10)
        batch_size = calculate_batch_size(n_steps)

        model = PPO(
            "CnnPolicy",
            env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            n_epochs=n_epochs,
            batch_size=batch_size,
            tensorboard_log=None
        )
    else:
        raise ValueError("Unsupported algorithm specified.")

    start_time = time.time()
    model.learn(total_timesteps=N_TIMESTEPS)
    training_time = time.time() - start_time

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=N_EVAL_EPISODES)

    hyperparameters = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        # Specific to DQN
        "exploration_final_eps": exploration_final_eps if ALGORITHM == "DQN" else None,
        # Specific to A2C and PPO
        "n_steps": n_steps if ALGORITHM in ["A2C", "PPO"] else None,
        # Specific to PPO
        "ent_coef": ent_coef if ALGORITHM == "PPO" else None,
    }
    log_training_results(ALGORITHM, hyperparameters, mean_reward, std_reward, training_time)

    env.close()

    return mean_reward



if __name__ == "__main__":
    start_time = time.time()
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    elapsed_time = time.time() - start_time

    print(f"\nElapsed time for {N_TRIALS} trial: {elapsed_time:.2f} seconds")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.params}")

    # Plot the optimization history
    optimization_history = plot_optimization_history(study)
    optimization_history.show()

    # Plot the importance of hyperparameters
    param_importances = plot_param_importances(study)
    param_importances.show()

    # Plot individual hyperparameters or combinations and their effects on the objective
    slice_plot = plot_slice(study)
    slice_plot.show()
