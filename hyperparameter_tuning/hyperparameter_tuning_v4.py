################################################################
# Hyperparameter tuning for sb-SuperMarioBros.py version 4
################################################################

import warnings
import csv
import time
import torch
import optuna
import random
import atexit
import datetime
import subprocess
import webbrowser
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym_super_mario_bros
from stable_baselines3.common.monitor import Monitor
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    NoopResetEnv,
    ClipRewardEnv,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global settings
N_TRIALS = 2
N_TIMESTEPS = int(2e4)
N_EVAL_EPISODES = 5
ENV_ID = "SuperMarioBros2-v1"
ALGORITHMS = ["DQN", "PPO", "A2C"]  # Algorithms to test
TENSORBOARD_BASE_DIR = "./tensorboard_hyperparameter_logs/"  # Base directory for TensorBoard logs


def cleanup_process():
    tensorboard_process.kill()
    
def log_training_results(algorithm, seed, hyperparameters, mean_reward, std_reward, training_time):
    log_filename = "tensorboard_hyperparameter_logs/hyperparameter_log_v4.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        file.seek(0, 2)
        if file.tell() == 0:
            writer.writerow([
                "Timestamp",
                "Algorithm",
                "Seed",
                "Hyperparameters",
                "Mean Reward",
                "Std Reward",
                "Training Time",
            ])
        writer.writerow([
            timestamp,
            algorithm,
            seed,
            str(hyperparameters),
            mean_reward,
            std_reward,
            training_time,
        ])

def save_optimization_history(study, filename="./plots/optimization_history.png"):
    fig = plot_optimization_history(study)
    fig.write_image(filename)
    fig.show()
    del fig
    
def save_param_importances(study, filename="./plots/param_importances.png"):
    fig = plot_param_importances(study)
    fig.write_image(filename)
    fig.show()
    del fig
    
def save_slice_plot(study, filename="./plots/slice_plot.png"):
    fig = plot_slice(study)
    fig.write_image(filename)
    fig.show()
    del fig
    
def calculate_batch_size(n_steps, min_batch_size=8, max_batch_size=256):
    total_steps = n_steps
    # Start with the max batch size and decrease until finding a divisor or reaching the min batch size
    for potential_batch_size in range(max_batch_size, min_batch_size - 1, -1):
        if total_steps % potential_batch_size == 0:
            return potential_batch_size
    # If no divisor is found within bounds, return the minimum batch size
    return min_batch_size

def make_env(gym_id, seed, log_dir="./env_logs"):
    env = gym_super_mario_bros.make(gym_id)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = MaxAndSkipEnv(env, 4)
    env = NoopResetEnv(env, noop_max=30)
    env = ClipRewardEnv(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Wrap the environment with Monitor
    env = Monitor(env, filename=log_dir, allow_early_resets=True)
    
    return env


def objective(trial, algorithm):
    # TensorBoard directory for this trial
    tensorboard_log = f"{TENSORBOARD_BASE_DIR}/{algorithm}/trial_{trial.number}"
    
    seed = random.randint(0, 1000)

    # Specify hyperparameters based on the algorithm
    hyperparameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "gamma": trial.suggest_float("gamma", 0.8, 0.9999),
        "buffer_size": None,
        "batch_size": None,
        "train_freq": None,
        "target_update_interval": None,
        "exploration_fraction": None,
        "exploration_final_eps": None,
        "n_steps": None,
        "vf_coef": None,
        "ent_coef": None,
        "n_epochs": None
    }

    # Initialize environment
    env = make_env(ENV_ID, seed)

    # Create and return model based on the algorithm
    if algorithm == "DQN":
        hyperparameters.update({
            "buffer_size": trial.suggest_int("buffer_size", 5000, 10000),
            "batch_size": trial.suggest_int("batch_size", 32, 256),
            "train_freq": trial.suggest_int("train_freq", 1, 100),
            "target_update_interval": trial.suggest_int("target_update_interval", 100, 5000),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
            "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1)
        })

        model = DQN(
            "CnnPolicy", 
            env,
            verbose=0,
            learning_rate=hyperparameters["learning_rate"],
            buffer_size=hyperparameters["buffer_size"],
            learning_starts=1000,
            batch_size=hyperparameters["batch_size"],
            gamma=hyperparameters["gamma"],
            train_freq=hyperparameters["train_freq"],
            target_update_interval=hyperparameters["target_update_interval"],
            exploration_fraction=hyperparameters["exploration_fraction"],
            exploration_final_eps=hyperparameters["exploration_final_eps"],
            tensorboard_log=tensorboard_log
        )
    elif algorithm == "A2C":
        hyperparameters.update({
            "n_steps": trial.suggest_int("n_steps", 5, 256),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "ent_coef": trial.suggest_float("ent_coef", 0.0001, 0.1)
        })

        model = A2C(
            "CnnPolicy", 
            env,
            verbose=0,
            learning_rate=hyperparameters["learning_rate"],
            n_steps=hyperparameters["n_steps"],
            gamma=hyperparameters["gamma"],
            vf_coef=hyperparameters["vf_coef"],
            ent_coef=hyperparameters["ent_coef"],
            tensorboard_log=tensorboard_log
        )
    elif algorithm == "PPO":
        hyperparameters.update({
            "n_steps": trial.suggest_int("n_steps", 64, 2048),
            "ent_coef": trial.suggest_float("ent_coef", 0.0001, 0.1),
            "n_epochs": trial.suggest_int("n_epochs", 1, 10),
        })
        hyperparameters["batch_size"] = calculate_batch_size(hyperparameters["n_steps"])

        model = PPO(
            "CnnPolicy",
            env,
            verbose=0,
            learning_rate=hyperparameters["learning_rate"],
            n_steps=hyperparameters["n_steps"],
            gamma=hyperparameters["gamma"],
            ent_coef=hyperparameters["ent_coef"],
            n_epochs=hyperparameters["n_epochs"],
            batch_size=hyperparameters["batch_size"],
            tensorboard_log=tensorboard_log
        )
    else:
        raise ValueError("Unsupported algorithm specified.")

    # Learn and evaluate policy
    try:
        start_time = time.time()
        model.learn(total_timesteps=N_TIMESTEPS)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES, return_episode_rewards=False)
        training_time = time.time() - start_time
    finally:
        # Attempt to clear the replay buffer
        if hasattr(model, 'replay_buffer'):
            del model.replay_buffer  # Ensure buffer is deleted
        del model  # Explicitly delete the model
        env.close()  # Close the environment
        torch.cuda.empty_cache()  # Clear GPU cache after freeing the model and buffer

    # Log the training results including all hyperparameters
    log_training_results(algorithm, seed, hyperparameters, mean_reward, std_reward, training_time)

    return mean_reward

if __name__ == "__main__":
    # Start TensorBoard as a background process
    tensorboard_process = subprocess.Popen(
        ['tensorboard', '--logdir', TENSORBOARD_BASE_DIR],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Register cleanup function to kill TensorBoard when the script exits
    atexit.register(cleanup_process)

    # Open a web browser automatically pointing to TensorBoard
    webbrowser.open("http://localhost:6006/", new=2)

    print("TensorBoard is running. Visit http://localhost:6006/ in your web browser.")
    time.sleep(5)
    
    # Define the SQLite URL to save the study
    sqlite_url = "sqlite:///hyperparameter_study.db"

    
    overall_best_reward = -float('inf')
    best_algorithm = None

    for algorithm in ALGORITHMS:
        # Create or load a study
        study = optuna.create_study(study_name="Hyperparameter_Optimization_SB3",
                                    direction="maximize",
                                    storage=sqlite_url,
                                    load_if_exists=True)
        study.optimize(lambda trial: objective(trial, algorithm), n_trials=N_TRIALS, show_progress_bar=True)

        # Output results
        print(f"Results for {algorithm}:")
        print(f"Best trial until now: {study.best_trial.value}")
        print(f"Best hyperparameters: {study.best_trial.params}")

        # Update the best algorithm based on the reward
        if study.best_trial.value > overall_best_reward:
            overall_best_reward = study.best_trial.value
            best_algorithm = algorithm

        # Show and Save the visualizations
        save_optimization_history(study, f"./plots/{algorithm}_optimization_history.png")
        save_param_importances(study, f"./plots/{algorithm}_param_importances.png")
        save_slice_plot(study, f"./plots/{algorithm}_slice_plot.png")

    print(f"The best performing algorithm is {best_algorithm} with a reward of {overall_best_reward}")
