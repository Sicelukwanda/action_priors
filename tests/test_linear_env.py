import unittest
from src.envs import LinearEnv
import numpy as np



class EnvTestCases(unittest.TestCase):
    def test_init_state_not_random(self):
        env = LinearEnv.LinearEnv(randomize=False)

        self.assertEqual(env.s,np.array([1]))


if __name__ == '__main__':
    unittest.main()
