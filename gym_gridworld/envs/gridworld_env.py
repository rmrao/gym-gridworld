import gym
import sys
import os
import time
import copy
import math
import pickle as pkl

from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt

from .GridWorld import GridMap
from .constants import *

class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_envs = 0 
    def __init__(self):
        self.actions = [0, 1, 2, 3, 4]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {0: [0,0], 1:[-1, 0], 2:[1,0], 3:[0,-1], 4:[0,1]}
        for act in self.action_pos_dict:
            self.action_pos_dict[act] = np.array(self.action_pos_dict[act])
        ''' set observation space '''
        self.obs_shape = [24, 24, 3]  # observation space shape
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float64)
        
        ''' set other parameters '''
        self.restart_once_done = False  # restart or not once done
        self._verbose = False # to show the environment or not 
        self._max_episode_steps = 2000

        self._cost_to_move = -1
        self._goal_reward = 2

        ''' initialize system state ''' 
        self._sample_grid_map()
        self.reset()

        self._seed = GridworldEnv.num_envs
        GridworldEnv.num_envs += 1
        self.this_fig_num = GridworldEnv.num_envs
        if self._verbose == True:
            self.fig = plt.figure(self.this_fig_num)
            plt.show(block=False)
            plt.axis('off')
            self._render()

    @property
    def timelimit(self):
        return self._max_episode_steps

    @timelimit.setter
    def timelimit(self, value):
        self._max_episode_steps = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def _sample_grid_map(self, n_rooms=None):
        if n_rooms is None:
            # n_rooms = np.random.randint(1, 4)
            n_rooms = 3
        elif np.isscalar(n_rooms):
            n_rooms = np.random.randint(1, n_rooms)
        elif len(n_rooms == 2):
            n_rooms = np.random.randint(n_rooms[0], n_rooms[1])
        else:
            raise ValueError("n_rooms must be scalar or have length two.")

        self.gridmap = GridMap(n_rooms, max_size=self.obs_shape[:2], add_doors=True)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.gridmap = pkl.load(f)
        mapshape = self.gridmap.map.shape
        self.obs_shape = list(mapshape) + [3]
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float64)
        
        self.reset()
        return self.gridmap.map

    def save(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.gridmap, f)
    
    def step(self, action):
        ''' return next observation, reward, finished, success '''
        action = int(action)
        self._num_steps += 1
        info = {}
        info['success'] = False
        nxt_agent_state = self.agent_state + self.action_pos_dict[action]
        timed_out = self._num_steps >= self._max_episode_steps

        reward = self._cost_to_move
        done = False
        new_state = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]

        if new_state in [SPACE]: # moving to empty space or door
            self.agent_state = nxt_agent_state
            info['success'] = True
        elif new_state in [WALL, DOOR]:
            info['success'] = False
        elif new_state == SWITCH:
            self._press_switch()
            self.agent_state = nxt_agent_state
            info['success'] = True
        elif new_state == TARGET:
            self.agent_state = nxt_agent_state
            info['success'] = True
            reward += self._goal_reward
            done = True

        self.observation = self._gridmap_to_observation(self.current_grid_map)
        self._render()
        return (self.observation, reward, done or timed_out, info)
    
    def _press_switch(self):
        self._switch_pressed = True
        self.current_grid_map[self.current_grid_map == DOOR] = SPACE 

    @property
    def _show_target(self):
        # y, x = self.agent_state
        # return self.gridmap._goal_room == self.gridmap.find_agent_room(y, x)
        return True

    def reset(self):
        # self.start_grid_map = self._sample_grid_map()
        self._num_steps = 0
        self._switch_pressed = False
        self.start_grid_map = self.gridmap.reset()
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.grid_map_shape = self.start_grid_map.shape

        ''' agent state: start, target, current state '''
        self.agent_state = copy.deepcopy(self.gridmap.agent_start)
        # must come after call to self.agent_state
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self._render()
        return self.observation

    def sample_new(self):
        self._sample_grid_map()
        self.reset()

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.zeros(list(grid_map.shape) + [3], dtype=np.uint8)
        for color in COLORS:
            mask = grid_map == color
            observation[mask, :] = COLORS[color]
        if not self._show_target:
            observation[grid_map == TARGET] = SPACE
        observation[self.agent_state[0], self.agent_state[1], :] = COLORS[AGENT]
        return observation
  
    def _render(self, mode='human', close=False):
        if self._verbose == False:
            return
        img = self.observation
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.00001)
        return 

class GridworldEnvV2(GridworldEnv):

    def __init__(self):
        super().__init__()
        self._block_move_reward = -self._cost_to_move
        self._reward_type = 'standard'
        self._goal_reward = 1


    def _sample_grid_map(self, n_rooms=None):
        if n_rooms is None:
            # n_rooms = np.random.randint(1, 4)
            n_rooms = 1
        elif np.isscalar(n_rooms):
            n_rooms = np.random.randint(1, n_rooms)
        elif len(n_rooms == 2):
            n_rooms = np.random.randint(n_rooms[0], n_rooms[1])
        else:
            raise ValueError("n_rooms must be scalar or have length two.")
        
        self.gridmap = GridMap(n_rooms, max_size=self.obs_shape[:2], add_block=True)

    def step(self, action):
        ''' return next observation, reward, finished, success '''
        action = int(action)
        self._num_steps += 1
        info = {}
        info['success'] = False
        nxt_agent_state = self.agent_state + self.action_pos_dict[action]
        nxt_block_state = self.block_state
        if action == 0 and np.sum(np.abs(self.agent_state - self.block_state)) == 1:
            nxt_agent_state, nxt_block_state = nxt_block_state, nxt_agent_state
        timed_out = self._num_steps >= self._max_episode_steps

        reward = self._cost_to_move
        done = False
        move_block = False
        new_state = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]
        if np.all(nxt_agent_state == self.block_state):
            nxt_block_state = self.block_state + self.action_pos_dict[action] if action != 0 else self.agent_state
            new_state = self.current_grid_map[nxt_block_state[0], nxt_block_state[1]]
            move_block = True
            if self._reward_type == 'intrinsic':
                reward += 0.5*self._block_move_reward
                self._block_move_reward /= 1.1
            if self._reward_type == 'curriculum1':
                reward += 1
        
        if self._reward_type in ['shaped', 'curriculum2']:
            sq_dist = np.sum(np.square(self.gridmap._goal_state - self.block_state))
            initial_dist = np.sqrt(sq_dist)

        if new_state in [SPACE]: # moving to empty space or door
            self.block_state = nxt_block_state
            self.agent_state = nxt_agent_state
            info['success'] = True
        elif new_state in [WALL, DOOR]:
            info['success'] = False
        elif new_state == SWITCH:
            self._press_switch()
            self.block_state = nxt_block_state
            self.agent_state = nxt_agent_state
            info['success'] = True
        elif new_state == TARGET:
            self.block_state = nxt_block_state
            self.agent_state = nxt_agent_state
            info['success'] = True
            if move_block:
                reward += self._goal_reward
                done = True

        if self._reward_type == 'shaped':
            sq_dist = np.sum(np.square(self.gridmap._goal_state - self.block_state))
            dist = np.sqrt(sq_dist)
            reward += initial_dist - dist
        if self._reward_type == 'curriculum2':
            sq_dist = np.sum(np.square(self.gridmap._goal_state - self.block_state))
            dist = np.sqrt(sq_dist)
            if dist < initial_dist:
                reward += 1

        self.observation = self._gridmap_to_observation(self.current_grid_map)
        self._render()
        return (self.observation, reward, done or timed_out, info)

    def reset(self):
        # self.start_grid_map = self._sample_grid_map()
        self._num_steps = 0
        self._block_move_reward = -self._cost_to_move
        self._switch_pressed = False
        self.start_grid_map = self.gridmap.reset()
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.grid_map_shape = self.start_grid_map.shape

        ''' agent state: start, target, current state '''
        self.agent_state = np.array(self.gridmap.agent_start)
        self.block_state = np.array(self.gridmap.block_start)
        # must come after call to self.agent_state
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self._render()
        return self.observation

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        observation = super()._gridmap_to_observation(grid_map, obs_shape)
        observation[self.block_state[0], self.block_state[1], :] = COLORS[BLOCK]
        return observation

    @property
    def reward_type(self):
        return self._reward_type

    @reward_type.setter
    def reward_type(self, val):
        if val not in ['standard', 'shaped', 'intrinsic', 'curriculum1', 'curriculum2']:
            raise ValueError("Unknown reward type")
        self._reward_type = val

