import numpy as np
from enum import Enum

import gymnasium as gym
import math
from gymnasium import spaces

class Actions(Enum):
    TURN_R = 0
    TURN_L = 1
    FORWARD = 2

class TouchBallNoPhysicsEnv(gym.Env):

    def __init__(self, field_width=304.8, field_height=426.72):
        self.field_width = field_width
        self.field_height = field_height
        self.turn_speed = 5
        self.move_speed = 5
        self.close_enough_radius = 3

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agentPos": spaces.Box(low=[0, 0], high=[self.field_width, self.field_height], dtype=float),
                "agentAngle" : spaces.Box(low=[0], high=[360], dtype=float),
                "ballPos": spaces.Box(low=[0, 0], high=[self.field_width, self.field_height], dtype=float),
            }
        )

        self._agent_location = np.array([-1, -1], dtype=float)
        self._agent_angle = np.array([0], dtype=float)
        self._ball_location = np.array([-1, -1], dtype=float)

        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        return {"agentPos": self._agent_location, "agentAngle": self._agent_angle, "ballPos": self._ball_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._ball_location, ord=1
            )
        }
        
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([self.np_random.uniform(0, self.field_width)], [self.np_random.uniform(0, self.field_height)])

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._ball_location = np.array([self.np_random.uniform(0, self.field_width)], [self.np_random.uniform(0, self.field_height)])

        # Random agent orientation
        self._agent_angle = np.array([self.np_random.uniform(0, 360)])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        # Perform action
        if (action == Actions.TURN_L):
            self._agent_angle -= self.turn_speed            

        elif (action == Actions.TURN_R):
            self._agent_angle += self.turn_speed

        elif (action == Actions.FORWARD):
            delta_x = self.move_speed * math.cos(math.radians(self._agent_angle))
            delta_y = self.move_speed * math.sin(math.radians(self._agent_angle))
            self._agent_location += np.array([delta_x, delta_y])
            self._agent_location[0] = np.clip(self._agent_location[0], 0, self.field_width) 
            self._agent_location[1] = np.clip(self._agent_location[1], 0, self.field_height)


        # Check if Terminated
        dist = np.linalg.norm(self._agent_location - self._ball_location)
        terminated = dist <= self.close_enough_radius

        # Rewards

        reward = 0
        if (terminated):
            reward += 1000


        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info

