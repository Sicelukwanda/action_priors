import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

        # rendering
        self.ax = None

    def reset(self):
        self.steps = 0

        if self.random_s0:
            self.s = np.random.choice(np.array([0.5, 1, self.s_max-0.5, self.s_max-1])) + np.random.normal(loc=0, scale=0.2, size=(1,))
            return self.s
        else:
            self.s = np.expand_dims(1.0, axis=0)
            return self.s

    def step(self, a):

        assert np.ndim(a) == 1, "incorrect action dim " + str(np.ndim(a))
        # clip the action
        a = np.clip(a, -1, 1)

        r = 0.0
        wall_penalty = -100

        done = False

        self.s = np.array(self.s) + np.array(a)

        # enforce boundaries
        if self.s < 0:
            self.s = np.array([0.0])
            r = wall_penalty
        elif self.s > self.s_max:
            self.s = self.s_max
            r = wall_penalty

        # check if goal reached or max steps reached
        if np.abs(self.s - self.s_goal) < self.goal_tol or self.steps == self.max_steps - 1:
            done = True
            r = 0.0
        else:
            # compute reward
            r += self.reward(self.s)
            self.steps += 1

        if self.visualize:
            self.render(a)

        return self.s, r, done, {}

    def reward(self, s):
        # insert code for checking if the agent is
        # within circle boundaries (highly penalising states)
        return -(s - self.s_goal)**2


    def render(self, a, azimuth=60, elevation=20):

        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')

        self.ax.scatter(self.s, self.steps*5, a, c='r', marker='o')
        self.ax.azim = azimuth
        self.ax.elev = elevation

        self.ax.set_xlabel('state')
        self.ax.set_ylabel('timestep')
        self.ax.set_zlabel('action')
        # draw starting and goal lines

