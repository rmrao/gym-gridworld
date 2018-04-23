import numpy as np
import matplotlib.pyplot as plt

class Room(object):
    # Coordinates are inclusive on top left, exclusive on bottom right
    def __init__(self, global_y, global_x, ysize, xsize):
        self.global_x = global_x
        self.global_y = global_y
        self.xsize = xsize
        self.ysize = ysize

        self._map = np.zeros((ysize, xsize), dtype=np.int32)
        self._map[(0,-1), :] = 1
        self._map[:, (0,-1)] = 1
        self._neighbors = [None]*4
    
    # TODO: There's something weird going on here...
    def clashes(self, other):
        return not (self.global_x > other.global_x + other.xsize or other.global_x > self.global_x + self.xsize) \
                or not (self.global_y > other.global_y + other.ysize or other.global_y > self.global_y + self.ysize)

    # TODO: Looks like it expects rooms to be offset by one. An off by one error somewhere
    def is_neighbor(self, other):
        if (self.global_x == other.global_x + other.xsize - 1 or other.global_x == self.global_x + self.xsize - 1):
            # Might be x-neighbor, need to check if ys overlap
            return self.global_y < other.global_y + other.ysize and self.global_y + self.ysize > other.global_y
        elif (self.global_y == other.global_y + other.ysize - 1 or other.global_y == self.global_y + self.ysize - 1):
            # Might be y-neighbor, need to check if xs overlap
            return self.global_x < other.global_x + other.xsize and self.global_x + self.xsize > other.global_x

        return False

    # def get_neighbors(self, list_of_rooms):
    #     for room in list_of_rooms:
    #         if self.is_neighbor(room):
    #             self._neighbors.append(room)
    #             room._neighbors.append(self)

    def create_door(self, other):
        if (self.global_x == other.global_x + other.xsize - 1):
            overlap = [max(self.global_y, other.global_y) + 1, min(self.global_y + self.ysize, other.global_y + other.ysize) - 1]
            door_loc = np.random.randint(overlap[0], overlap[1])
            self._map[door_loc - self.global_y, 0] = 2
            other._map[door_loc - other.global_y, -1] = 2
        elif (other.global_x == self.global_x + self.xsize - 1):
            overlap = [max(self.global_y, other.global_y) + 1, min(self.global_y + self.ysize, other.global_y + other.ysize) - 1]
            door_loc = np.random.randint(overlap[0], overlap[1])
            self._map[door_loc - self.global_y, -1] = 2
            other._map[door_loc - other.global_y, 0] = 2
        elif (self.global_y == other.global_y + other.ysize - 1):
            overlap = [max(self.global_x, other.global_x) + 1, min(self.global_x + self.xsize, other.global_x + other.xsize) - 1]
            door_loc = np.random.randint(overlap[0], overlap[1])
            self._map[0, door_loc - self.global_x] = 2
            other._map[-1, door_loc - other.global_x] = 2
        elif (other.global_y == self.global_y + self.ysize - 1):
            overlap = [max(self.global_x, other.global_x) + 1, min(self.global_x + self.xsize, other.global_x + other.xsize) - 1]
            door_loc = np.random.randint(overlap[0], overlap[1])
            self._map[-1, door_loc - self.global_x] = 2
            other._map[0, door_loc - other.global_x] = 2
        else:
            raise ValueError("Input room is not a neighbor.")


    def render(self):
        plt.imshow(1 - self._map, cmap='gray')
        plt.show(block=False)

    @property
    def position(self):
        return (slice(self.global_y, self.global_y + self.ysize), slice(self.global_x, self.global_x + self.xsize))

    @property
    def interior(self):
        return (slice(self.global_y + 1, self.global_y + self.ysize - 1), slice(self.global_x + 1, self.global_x + self.xsize - 1))

    @property
    def doors(self):
        doorsy, doorsx = np.where(self._map == 2)
        return (doorsy + self.global_y, doorsx + self.global_x)
    
    @property
    def map(self):
        return self._map

class GridMap(object):

    def __init__(self, n_rooms=5, max_size=(128, 128), gridmap=None):
        if gridmap is not None:
            self._map = gridmap
            self.max_size = gridmap.shape
            self.reset()
            return

        self._map = None
        self.max_size = max_size

        self._rooms = []
        for _ in range(100): # try a bunch to create the correct number of rooms
            self.create_room()
            if len(self._rooms) == n_rooms:
                break

        # get doors
        for room in self._rooms:
            self._map[room.doors] = 2

        new_map = np.zeros(self.max_size, dtype=np.int32) - 1
        new_map[:self._map.shape[0], :self._map.shape[1]] = self._map
        self._map = new_map

        self._add_agent_and_target()

    @classmethod
    def fromfile(cls, filename):
        gridmap = np.load(filename)
        return cls(gridmap=gridmap)

    def render(self):
        plt.imshow(self._map)
        plt.show(block=False)

    def _request_location(self, room):
        if self._map is None:
            self._map = np.zeros(room.map.shape, dtype=np.int32) - 1
            return True

        if (room.global_x < 0 or room.global_y < 0):
            xshift = max(0, -room.global_x)
            yshift = max(0, -room.global_y)
    
            curr_shape = self._map.shape
            new_shape = (curr_shape[0] + yshift, curr_shape[1] + xshift)
            if new_shape[0] > self.max_size[0] or new_shape[1] > self.max_size[1]:
                return "Room too big!"

            room.global_x += xshift
            room.global_y += yshift
            for curr_room in self._rooms:
                curr_room.global_x += xshift
                curr_room.global_y += yshift

            new_map = np.zeros(new_shape, dtype=np.int32) - 1
            new_map[yshift:, xshift:] = self._map
            self._map = new_map

        xmax = room.global_x + room.xsize
        ymax = room.global_y + room.ysize

        if (xmax > self._map.shape[1] or ymax > self._map.shape[0]):
            curr_shape = self._map.shape
            new_shape = (max(curr_shape[0], ymax), max(curr_shape[1], xmax))
            if new_shape[0] > self.max_size[0] or new_shape[1] > self.max_size[1]:
                return "Room too big!"
            new_map = np.zeros(new_shape, dtype=np.int32) - 1
            new_map[:self._map.shape[0], :self._map.shape[1]] = self._map
            self._map = new_map

        if np.any(self._map[room.interior] != -1):
            return False

        return True

    def create_room(self):
        ysize, xsize = np.random.randint(7, 15, 2)
        if not self._rooms:
            room = Room(0, 0, ysize, xsize)
            if self._request_location(room) is True:
                self._map[room.position] = room.map
                self._rooms.append(room)
            else:
                raise RuntimeError("Shouldn't ever get here!")
        else:
            while True:
                neighbor_room = np.random.choice(self._rooms)
                if all(neighbor_room._neighbors): # check to make sure room doesn't have four neighbors already
                    continue
                possible_locs = [i for i, r in enumerate(neighbor_room._neighbors) if r is None]
                loc = np.random.choice(possible_locs)
                if loc == 0:
                    xloc = neighbor_room.global_x - xsize + 1
                    yloc = neighbor_room.global_y # TODO: randomize this somewhat
                elif loc == 1:
                    yloc = neighbor_room.global_y - ysize + 1
                    xloc = neighbor_room.global_x # TODO: randomize this somewhat
                elif loc == 2:
                    xloc = neighbor_room.global_x + neighbor_room.xsize - 1
                    yloc = neighbor_room.global_y # TODO: randomize this somewhat
                elif loc == 3:
                    yloc = neighbor_room.global_y + neighbor_room.ysize - 1
                    xloc = neighbor_room.global_x # TODO: randomize this somewhat
                else:
                    raise RuntimeError("Something wrong with possible_locs array.")

                room = Room(yloc, xloc, ysize, xsize)
                res = self._request_location(room)
                if res is True:
                    self._map[room.position] = room.map
                    self._rooms.append(room)
                    neighbor_room._neighbors[loc] = room
                    room._neighbors[(loc + 2) % 4] = neighbor_room
                    room.create_door(neighbor_room)
                    break
                elif res == "Room too big!":
                    return False
        return True

    def _add_agent_and_target(self):
        y, x = np.where(self._map == 0)
        loc = np.random.randint(len(y))
        y = y[loc]
        x = x[loc]
        self._map[y, x] = 4

        y, x = np.where(self._map == 0)
        loc = np.random.randint(len(y))
        y = y[loc]
        x = x[loc]
        self._map[y, x] = 3

    def _remove_agent_and_target(self):
        self._map[self._map == 4] = 0
        self._map[self._map == 3] = 0

    def reset(self):
        self._remove_agent_and_target()
        self._add_agent_and_target()
        return self.map

    @property
    def map(self):
        return self._map

    # def create_room(self, neighboring_room=None):
    #     xstart, xend, ystart, yend = (None,)*4
    #     if neighboring_room is None:
    #         unoccupied = np.where(self._map == -1)
    #         location = np.random.randint(unoccupied[0].shape[0])
    #         ystart = unoccupied[0][location]
    #         xstart = unoccupied[1][location]

    #         ymax = np.where(self._map[ystart:, xstart] != -1)[0]
    #         if len(ymax) == 0:
    #             ymax = self._size[0]
    #         else:
    #             ymax = ystart + ymax[0]

    #         yend = np.random.randint(ystart + 1, ymax + 1)

    #         xmax = np.where(self._map[ystart:yend, xstart:] != -1)[1]
    #         if len(xmax) == 0:
    #             xmax = self._size[1]
    #         else:
    #             xmax = xstart + np.min(xmax)

    #         xend = np.random.randint(xstart + 1, xmax + 1)
    #     else: # TODO: room above/below won't quite work... need to do it based on other rooms too
    #         room_left = neighboring_room.global_x
    #         room_right = self._size[1] - (neighboring_room.global_x + neighboring_room.xsize)
    #         room_above = neighboring_room.global_y
    #         room_below = self._size[0] - (neighboring_room.global_y + neighboring_room.ysize)

    #         probs = np.array([room_left, room_right, room_above, room_below])
    #         probs = probs / np.sum(probs)
    #         relative_loc = np.random.choice(4, p=probs)
    #         if relative_loc in [0, 1]:
    #             if relative_loc == 0:
    #                 xstart = np.random.randint(room_left)
    #                 xend = neighboring_room.global_x + 1
    #             else:
    #                 xstart = neighboring_room.global_x + neighboring_room.xsize - 1
    #                 xend = np.random.randint(xstart, xstart + room_right)

    #             occupied_space = np.where(self._map[:,xstart:xend] != -1)[0]
    #             print(occupied_space)
    #             if len(occupied_space) == 0:
    #                 ymin = 0
    #                 ymax = self._size[0]
    #             else:
    #                 ymin = occupied_space.min()
    #                 ymax = occupied_space.max()
    #             ystart = np.random.randint(ymin, ymax)
    #             yend = np.random.randint(ystart + 1, ymax + 1)
    #         else:
    #             if relative_loc == 3:
    #                 ystart = np.random.randint(room_above)
    #                 yend = neighboring_room.global_y + 1
    #             else:
    #                 ystart = neighboring_room.global_y + neighboring_room.ysize - 1
    #                 yend = np.random.randint(ystart, ystart + room_below)

    #             occupied_space = np.where(self._map[ystart:yend,:] != -1)[0]
    #             print(occupied_space)
    #             if len(occupied_space) == 0:
    #                 xmin = 0
    #                 xmax = self._size[1]
    #             else:
    #                 xmin = occupied_space.min()
    #                 xmax = occupied_space.max()
    #             xstart = np.random.randint(xmin, xmax)
    #             xend = np.random.randint(xstart + 1, xmax + 1)
    #     print(ystart, xstart, yend, xend)
    #     room = Room(ystart, xstart, yend - ystart, xend - xstart)
    #     self._map[room.position] = room.map
    #     self._rooms.append(room)
    #     # Check for neighbors

# TODO: Creating doors - do an initial min spanning tree, followed by a random opening of doors based on some parameter
