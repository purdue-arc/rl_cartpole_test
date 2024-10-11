import numpy as np

import gymnasium as gym
from gymnasium import spaces

class TouchBallEnv(gym.Env):

    def __init__(self, field_width=304.8, field_height=426.72):
        self.field_width = field_width
        self.field_height = field_height

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=[0, 0], high=[self.field_width, self.field_height], dtype=float),
                "ball": spaces.Box(low=[0, 0], high=[self.field_width, self.field_height], dtype=float),
            }
        )

        self._agent_location = np.array([-1, -1], dtype=int)
        self._ball_location = np.array([-1, -1], dtype=int)

        self.action_space = spaces.Dict(
            {
                "throttle": spaces.Box(low=-1, high=1, dtype=float),
                "steer": spaces.Box(low=-1, high=1, dtype=float),
            }
        )