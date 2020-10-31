import unittest
import numpy as np
from src.agents.AgentMLPTF import AgentMLPTF
import tensorflow as tf




# agent.visualise_model()
# print(agent.summary())

class MyTestCase(unittest.TestCase):
    def test_single_input(self):
        # reset seeds
        np.random.seed(1)
        tf.random.set_seed(1)
        # instantiate agent
        agent = AgentMLPTF()
        action = agent(np.array([2]))["Action"]
        self.assertEqual(action, np.array([0.57267284],dtype=np.float32))

    def test_batch_input(self):
        # reset seeds
        np.random.seed(1)
        tf.random.set_seed(1)
        # instantiate agent
        agent = AgentMLPTF()
        action = agent(np.array([[2]]))["Action"]
        self.assertEqual(action, np.array([0.57267284],dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
