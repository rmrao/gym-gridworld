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
    num_env = 0 
    def __init__(self):
        self.actions = [0, 1, 2, 3, 4]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {0: [0,0], 1:[-1, 0], 2:[1,0], 3:[0,-1], 4:[0,1]}
 
        ''' set observation space '''
        self.obs_shape = [24, 24, 3]  # observation space shape
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape)
        
        ''' set other parameters '''
        self.restart_once_done = False  # restart or not once done
        self._verbose = False # to show the environment or not 
        self._max_episode_steps = 2000

        ''' initialize system state ''' 
        self._sample_grid_map()
        self._reset()


        GridworldEnv.num_env += 1
        self.this_fig_num = GridworldEnv.num_env 
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

        self.gridmap = GridMap(n_rooms, max_size=self.obs_shape[:2])

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.gridmap = pkl.load(f)
        mapshape = self.gridmap.map.shape
        self.obs_shape = list(mapshape) + [3]
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape)
        
        self.reset()
        return self.gridmap.map

    def save(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.gridmap, f)
    
    def _sample_maze(self, n_row, n_col, epsilon=0.2):
        maze = np.ones((n_row, n_col), dtype=np.int32)
        visited = np.zeros(maze.shape, dtype=np.bool)
        # maze[:,(0, -1)] = 1
        # maze[(0, -1), :] = 1
        # visited[:,(0, -1)] = True
        # visited[(0, -1), :] = True
        def get_neighbors(loc):
            neighbors = np.tile(loc, (4, 1)) + np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
            neighbors = neighbors[np.all(neighbors >= 0, 1) & (neighbors[:,0] < n_row) & (neighbors[:,1] < n_col)]
            return neighbors
        
        start = np.array([np.random.randint(n_row), np.random.randint(n_col)])
        locs = [start]
        while locs:
            loc = locs.pop(np.random.randint(len(locs)))
            neighbors = get_neighbors(loc)
            if neighbors.shape[0] < 4:
                visited[loc[0], loc[1]] = True
            elif np.count_nonzero(visited[neighbors[:,0], neighbors[:,1]]) < 2 or np.random.random() < epsilon:
                maze[loc[0], loc[1]] = 0
                visited[loc[0], loc[1]] = True
                for neighbor in neighbors:
                    if not visited[neighbor[0], neighbor[1]]:
                        locs.append(neighbor)

        while True:
            goal = np.array([np.random.randint(n_row), np.random.randint(n_col)])
            if maze[goal[0], goal[1]] == 0:
                maze[goal[0], goal[1]] = 3
                break
        while True:
            agent = np.array([np.random.randint(n_row), np.random.randint(n_col)])
            if maze[agent[0], agent[1]] == 0:
                maze[agent[0], agent[1]] = 4
                break

        return maze

    def _step(self, action):
        ''' return next observation, reward, finished, success '''
        action = int(action)
        self._num_steps += 1
        info = {}
        info['success'] = False
        nxt_agent_state = self.agent_state + self.action_pos_dict[action]
        timed_out = self._num_steps >= self._max_episode_steps

        reward = 0
        done = False
        new_state = self.current_grid_map[nxt_agent_state]

        if new_color in [SPACE]: # moving to empty space or door
            self.agent_state = nxt_agent_state
            info['success'] = True
        elif new_color in [WALL, DOOR]:
            info['success'] = False
        elif new_color == SWITCH:
            self._press_switch()
            self.agent_state = nxt_agent_state
            info['success'] = True
        elif new_color == TARGET:
            self.agent_state = nxt_agent_state
            info['success'] = True
            reward = 1
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

    def _reset(self):
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
        self._reset()

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.zeros(list(grid_map.shape) + [3], dtype=np.uint8)
        for color in COLORS:
            mask = grid_map == color
            observation[mask, :] = COLORS[color]
        if not self._show_target:
            observation[grid_map == TARGET] = SPACE
        observation[self.agent_state, :] = COLORS[AGENT]

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
