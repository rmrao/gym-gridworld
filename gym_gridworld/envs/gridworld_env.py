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
    
        ''' initialize system state ''' 
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, 'plan1.txt')        
        # self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map
        self.start_grid_map = self._sample_grid_map()
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        
        self.grid_map_shape = self.start_grid_map.shape

        ''' agent state: start, target, current state '''
        self.agent_start_state, _ = self._get_agent_start_target_state(
                                    self.start_grid_map)
        _, self.agent_target_state = self._get_agent_start_target_state(
                                    self.start_grid_map)
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.observation = self._gridmap_to_observation(self.start_grid_map)


        self._max_episode_steps = 2000
        self._num_steps = 0

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

        return self.gridmap.map

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
        nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                            self.agent_state[1] + self.action_pos_dict[action][1])
        timed_out = self._num_steps >= self._max_episode_steps
        if action == 0: # stay in place
            info['success'] = True
            doreward = np.all(nxt_agent_state == self.agent_target_state)
            return (self.observation, int(doreward), doreward or timed_out, info) 
        if nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]:
            info['success'] = False
            return (self.observation, 0, timed_out, info)
        if nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1]:
            info['success'] = False
            return (self.observation, 0, timed_out, info)
        # successful behavior
        new_color = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]
        if new_color in [SPACE, DOOR]: # moving to empty space or door
            self.current_grid_map[self.agent_state] = self.start_grid_map[self.agent_state]
            if self.current_grid_map[self.agent_state] == AGENT:
                self.current_grid_map[self.agent_state] = SPACE # agent start state will be 4 in start_grid_map
            self.current_grid_map[nxt_agent_state] = AGENT
            self.agent_state = copy.deepcopy(nxt_agent_state)
        elif new_color == WALL: # wall
            info['success'] = False
            return (self.observation, 0, timed_out, info)
        elif new_color == TARGET: # target
            self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = new_color+4
            self.agent_state = copy.deepcopy(nxt_agent_state)
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        self._render()
        if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1]:
            target_observation = copy.deepcopy(self.observation)
            if self.restart_once_done:
                self.observation = self._reset()
                info['success'] = True
                return (self.observation, 1, True, info)
            else:
                info['success'] = True
                return (target_observation, 1, True, info)
        else:
            info['success'] = True
            return (self.observation, 0, timed_out, info)

    @property
    def _show_target(self):
        y, x = self.agent_state
        return self.gridmap._goal_room == self.gridmap.find_agent_room(y, x)

    def _reset(self):
        # self.start_grid_map = self._sample_grid_map()
        self._num_steps = 0
        self.start_grid_map = self.gridmap.reset()
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.grid_map_shape = self.start_grid_map.shape

        ''' agent state: start, target, current state '''
        self.agent_start_state, _ = self._get_agent_start_target_state(
                                    self.start_grid_map)
        _, self.agent_target_state = self._get_agent_start_target_state(
                                    self.start_grid_map)
        self.agent_state = copy.deepcopy(self.agent_start_state)

        # must come after call to self.agent_state
        self.observation = self._gridmap_to_observation(self.start_grid_map)

        self._render()
        return self.observation

    def sample_new(self):
        self._num_steps = 0
        self.start_grid_map = self._sample_grid_map()
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        
        self.grid_map_shape = self.start_grid_map.shape

        ''' agent state: start, target, current state '''
        self.agent_start_state, _ = self._get_agent_start_target_state(
                                    self.start_grid_map)
        _, self.agent_target_state = self._get_agent_start_target_state(
                                    self.start_grid_map)
        self.agent_state = copy.deepcopy(self.agent_start_state)

        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self._render()
        return self.observation

    def _read_grid_map(self, grid_map_path):
        grid_map = open(grid_map_path, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array)
        return grid_map_array

    def _get_agent_start_target_state(self, start_grid_map):
        start_y, start_x = np.where(start_grid_map == AGENT)
        target_y, target_x = np.where(start_grid_map == TARGET)
        if len(start_y) != 1:
            raise ValueError("Start state not specified correctly")
        if len(target_y) != 1:
            raise ValueError("Target state not specified correctly")
        start_state = (start_y[0], start_x[0])
        target_state = (target_y[0], target_x[0])

        # for i in range(start_grid_map.shape[0]):
        #     for j in range(start_grid_map.shape[1]):
        #         this_value = start_grid_map[i,j]
        #         if this_value == 4:
        #             start_state = [i,j]
        #         if this_value == 3:
        #             target_state = [i,j]
        return start_state, target_state

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.zeros(list(grid_map.shape) + [3], dtype=np.uint8)
        for color in COLORS:
            mask = grid_map == color
            observation[mask, :] = COLORS[color]
        if not self._show_target:
            observation[grid_map == TARGET] = SPACE

        # Reshape the image:
        # im = Image.fromarray(observation)
        # observation = np.array(im.resize(obs_shape[:2]))
        # gs0 = math.ceil(observation.shape[0] / grid_map.shape[0])
        # gs1 = math.ceil(observation.shape[1] / grid_map.shape[1])
        # for i in range(grid_map.shape[0]):
        #     for j in range(grid_map.shape[1]):
        #         observation[i * gs0:(i+1)*gs0, j*gs1:(j+1)*gs1] = COLORS[grid_map[i,j]]
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
 
    def change_start_state(self, sp):
        ''' change agent start state '''
        ''' Input: sp: new start state '''
        if self.agent_start_state[0] == sp[0] and self.agent_start_state[1] == sp[1]:
            _ = self._reset()
            return True
        elif self.start_grid_map[sp[0], sp[1]] != 0:
            return False
        else:
            s_pos = copy.deepcopy(self.agent_start_state)
            self.start_grid_map[s_pos[0], s_pos[1]] = 0
            self.start_grid_map[sp[0], sp[1]] = 4
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_start_state = [sp[0], sp[1]]
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self._reset()
            self._render()
        return True
        
    
    def change_target_state(self, tg):
        if self.agent_target_state[0] == tg[0] and self.agent_target_state[1] == tg[1]:
            _ = self._reset()
            return True
        elif self.start_grid_map[tg[0], tg[1]] != 0:
            return False
        else:
            t_pos = copy.deepcopy(self.agent_target_state)
            self.start_grid_map[t_pos[0], t_pos[1]] = 0
            self.start_grid_map[tg[0], tg[1]] = 3
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_target_state = [tg[0], tg[1]]
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self._reset()
            self._render()
        return True
    
    def get_agent_state(self):
        ''' get current agent state '''
        return self.agent_state

    def get_start_state(self):
        ''' get current start state '''
        return self.agent_start_state

    def get_target_state(self):
        ''' get current target state '''
        return self.agent_target_state

    def _jump_to_state(self, to_state):
        ''' move agent to another state '''
        info = {}
        info['success'] = True
        if self.current_grid_map[to_state[0], to_state[1]] == 0:
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 4:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.agent_state = [to_state[0], to_state[1]]
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self._render()
                return (self.observation, 0, False, info)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 6:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 2
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.agent_state = [to_state[0], to_state[1]]
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self._render()
                return (self.observation, 0, False, info)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 7:  
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 3
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.agent_state = [to_state[0], to_state[1]]
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self._render()
                return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 4:
            return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 1:
            info['success'] = False
            return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 3:
            self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_grid_map[to_state[0], to_state[1]] = 7
            self.agent_state = [to_state[0], to_state[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self._render()
            if self.restart_once_done:
                self.observation = self._reset()
                return (self.observation, 1, True, info)
            return (self.observation, 1, True, info)
        else:
            info['success'] = False
            return (self.observation, 0, False, info)

    def _close_env(self):
        plt.close(1)
        return
    
    def jump_to_state(self, to_state):
        a, b, c, d = self._jump_to_state(to_state)
        return (a, b, c, d) 

