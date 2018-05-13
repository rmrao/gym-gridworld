import numpy as np

NULL = -1
SPACE = 0
WALL = 1
DOOR = 2
TARGET = 3
AGENT = 4
SWITCH = 5
BLOCK = 6

COLORS = {NULL   : np.array([128, 128, 128]), 
          SPACE  : np.array([0, 0, 0]), 
          WALL   : np.array([128, 128, 128]), 
          DOOR   : np.array([0, 50, 50]), 
          TARGET : np.array([0, 255, 0]), 
          AGENT  : np.array([255, 0, 0]),
          SWITCH : np.array([200, 200, 0]),
          BLOCK  : np.array([255, 140, 0])}
