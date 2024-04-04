import sys
import gym
import pickle
import random
import tkinter as tk
from tkinter import ttk
from typing import Callable
from stable_baselines3 import DQN,A2C,PPO
from stable_baselines3.common.evaluation import evaluate_policy

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from stable_baselines3.common import atari_wrappers


class LunarLanderGUI:
    def __init__(self, master):
        self.master = master
        master.title("Lunar Lander")

        self.mode_label = ttk.Label(master, text="Select Mode:")
        self.mode_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')

        self.mode_var = tk.StringVar()
        self.train_radio = ttk.Radiobutton(master, text="Train", variable=self.mode_var, value="Train")
        self.train_radio.grid(row=0, column=1, padx=10, pady=5)
        self.test_radio = ttk.Radiobutton(master, text="Test", variable=self.mode_var, value="Test")
        self.test_radio.grid(row=0, column=2, padx=10, pady=5)

        self.algorithm_label = ttk.Label(master, text="Select Algorithm:")
        self.algorithm_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')

        self.algorithm_var = tk.StringVar()
        self.dqn_radio = ttk.Radiobutton(master, text="DQN", variable=self.algorithm_var, value="DQN")
        self.dqn_radio.grid(row=1, column=1, padx=10, pady=5)
        self.a2c_radio = ttk.Radiobutton(master, text="A2C", variable=self.algorithm_var, value="A2C")
        self.a2c_radio.grid(row=1, column=2, padx=10, pady=5)
        self.ppo_radio = ttk.Radiobutton(master, text="PPO", variable=self.algorithm_var, value="PPO")
        self.ppo_radio.grid(row=1, column=3, padx=10, pady=5)

        self.submit_button = ttk.Button(master, text="Submit", command=self.submit)
        self.submit_button.grid(row=2, columnspan=4, padx=10, pady=10)

    def submit(self):
        mode = self.mode_var.get().lower()
        algorithm = self.algorithm_var.get().upper()
        seed = 123  # Set a default seed here or ask user for seed input
        self.master.destroy()
        if mode == "train":
            sys.argv = [sys.argv[0], "train", algorithm]
        elif mode == "test":
            sys.argv = [sys.argv[0], "test", algorithm, str(seed)]  # Change str(seed) to the seed input if asked
        else:
            print("Invalid mode selected.")
            return
        exec(open("mine/sb-SuperMarioBros.py").read())

def main():
    root = tk.Tk()
    app = LunarLanderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()