from src.envs import LinearEnv
import numpy as np

env = LinearEnv.LinearEnv()

print(env.step(np.array([4])))
