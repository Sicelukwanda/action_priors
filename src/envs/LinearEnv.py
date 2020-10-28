import numpy as np
import matplotlib.pyplot as plt


class LinearEnv:
    """
    A univariate environment on a horizontal line that has regions of highly negative reward.
    """

    def __init__(self, s_goal=6, randomize=False, s_max=10, visualize=False):
        """
        s_goal - goal state
        randomize - bool flag for whether state should be fixed or randomized at each reset() call
        s_max - hard boundary to the right of the environment
        """

        # initialise env
        self.random_s0 = randomize
        self.visualize = visualize
        self.goal_tol = 0.5
        self.max_steps = 20

        self.s_goal = np.expand_dims(s_goal, axis=0)
        self.s_max = np.expand_dims(s_max, axis=0)
        self.s = self.reset()

    def reset(self):
        self.max_steps = 20

        if self.random_s0:
            self.s = np.random.beta(a=0.5, b=0.5, size=(1,)) * self.s_max
            return self.s
        else:
            self.s = np.expand_dims(1.0, axis=0)
            return self.s

    def step(self, a):

        assert np.ndim(a) == 1, "incorrect action dim"
        # clip the action
        a = np.clip(a, -0.5, 0.5)

        done = False
        # self.s += np.array(a)
        self.add = np.add(self.s, a, out=self.s, casting="unsafe")

        # enforce boundaries
        if self.s < 0:
            self.s = np.array([0.0])
        elif self.s > self.s_max:
            self.s = self.s_max

        # check if goal reached or max steps reached
        if np.abs(self.s - self.s_goal) < self.goal_tol or self.max_steps == 0:
            done = True
            r = 0.0
        else:
            # compute reward
            r = self.reward(self.s)
            self.max_steps -= 1

        if self.visualize:
            self.render()

        return self.s, r, done, {}

    def reward(self, s):
        # insert code for checking if the agent is
        # within circle boundaries (highly penalising states)
        return - np.abs(s - self.s_goal)

    def render(self):
        plt.scatter(self.s, 0)
        plt.scatter(self.s_goal, 0)
