import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros2-v1')
obs = env.reset()

print(obs.shape)

done = False

while not done:
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
    
env.close()