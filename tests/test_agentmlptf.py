import unittest
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)

# instantiate agent
agent = AgentMLPTF()


agent.visualise_model()
# print(agent.summary())

print("Agent:\n", agent(np.array([2])))

class MyTestCase(unittest.TestCase):
    def test_single_input(self):
        self.assertEqual(True, False)

    def test_batch_input(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
