from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Step 1: Create the environment
env = gym_super_mario_bros.make('SuperMarioBros2-v1')

# Step 2: Constrain the action space
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Step 3: Optionally, set a seed for reproducibility
env.seed(123)

# Optionally, render the initial state
env.render()

# Now you can start interacting with the environment
