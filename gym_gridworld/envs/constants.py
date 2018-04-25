import numpy as np

NULL = -1
SPACE = 0
WALL = 1
DOOR = 2
TARGET = 3
AGENT = 4

COLORS = {NULL   : np.array([128, 128, 128]), 
          SPACE  : np.array([1, 1, 1]), 
          WALL   : np.array([128, 128, 128]), 
          DOOR   : np.array([0, 50, 50]), 
          TARGET : np.array([0, 255, 255]), 
          AGENT  : np.array([255, 0, 0])}