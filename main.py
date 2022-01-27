import tkinter as tk

import cv2
import numpy as np
import time
from PIL import Image, ImageTk

N_ENTRIES = 100
HEIGHT = 620
WIDTH = 600


def for_all_objects(func):
    """Decorator Applying a function to all objects, other arguments are passed directly"""

    def wrapper(caller, objectIDs, *args, **kwargs):
        for objID in objectIDs:
            func(caller, objID,  *args, **kwargs)

    return wrapper

def for_all_positional(func):
    """Decorator Applying a function to all objects, keyword arguments are passed directly,
    positional arguments are reduced for each object as well"""

    def wrapper(caller, objectIDs, *args, **kwargs):
        if len(args) == 0:
            return for_all_objects(func)
        elif len(args) == 1:
            for objID, parameter in zip(objectIDs, *args):
                func(caller, objID, parameter, **kwargs)
        else:
            for objID, *parameter in zip(objectIDs, *args):
                func(caller, objID, *parameter, **kwargs)

    return wrapper

def for_all(func):
    """Decorator Applying a function to all objects, keyword arguments and
    positional arguments are reduced for each object as well"""
    
    def wrapper(caller, objectIDs, *args, **kwargs):
        keywords = kwargs.keys()
        amount_keywords = len(keywords)
        if amount_keywords == 0:
            return for_all_positional(func)

        else:
            for objID, parameter in zip(objectIDs, *args, **kwargs):
                func(caller, objID, *parameter[:-amount_keywords], **dict(zip(keywords, parameter[:amount_keywords])))

    return wrapper


class GameLoop(tk.Canvas):
    def __init__(self, fps=30):
        super().__init__(width=WIDTH, height=HEIGHT, background="black", highlightthickness=0)

        # game parameter
        self.fps = fps

        # debug variables
        self.counter = 0
        self.times = []

        # game variables
        self.positions = None
        self.position_IDs = None
        self.speeds = None
        self.object_IDs = None
        self.colors = None
        self.diameter = None
        self.masses = None

        # initialize simulation
        self.initialize_positions()
        self.initialize_speeds()
        self.inititalize_objects()

        # populate GUI
        self.pack()

        self.sort_and_sweep(self.positions, self.diameter, self.position_IDs)
        # start Gameloop
        self.gameloop()

    def sort_and_sweep(self, positions, diameter, position_IDs):
        """Do a sort and sweep to find colliding objects. Sorting is applied to the position_IDs, positions is not changed!"""

        # get axis with most variance
        var_x = np.var(positions[:,0])
        var_y = np.var(positions[:,1])
        selected_axis = int(var_y > var_x)


        # get start of object on axis and end of object on axis
        start_positions = positions[:, selected_axis]
        end_positions = (positions[:, selected_axis] + diameter)

        # combine to one array, with start and end points, alternatingly combined,
        # use order of position_ids to create in order to use previous sorting
        all_positions = np.empty((positions.shape[0] * 2,), dtype=float)
        all_positions[::2] = np.take(start_positions, position_IDs[:,selected_axis])
        all_positions[1::2] = np.take(end_positions, position_IDs[:,selected_axis])

        # sort
        sort_indices = np.argsort(all_positions)
        positions_sorted = np.take(all_positions, sort_indices)

        # go over sorting result, even numbers are start positions, odd end positions
        active_intervals = []
        possible_collision = set()
        even_counter = 0
        for ind in sort_indices:
            # even number = start position
            if ind % 2 == 0:
                active_intervals.append(ind >> 1) # bitshift right to divide by two
                position_IDs[even_counter, selected_axis] = ind >> 1
                even_counter += 1
            # odd number = end position
            else:

                # only current starting interval in list
                if len(active_intervals) == 1:
                    active_intervals = []

                # other elements in list, therefore collision possible
                else:
                    possible_collision = possible_collision.union(active_intervals)
                    active_intervals.remove((ind-1) >> 1)

        # go over other axis to find definitive collisions
        other_axis = int(not selected_axis)
        possible_collision = list(possible_collision)
        print(possible_collision)
        start_positions = np.take(positions[:, other_axis], possible_collision)
        end_positions = np.take(positions[:, other_axis], possible_collision) + np.take(diameter, possible_collision)

        print(len(possible_collision))
        print(start_positions.shape)
        print(start_positions)
        print(f"x: {var_x}, y: {var_y}, sel: {selected_axis}")
        #print(sort_indices)
        #print(possible_collision)
        #print(all_positions[positions_sorted])

    def hit(self, speed1 : np.ndarray, speed2,  mass1, mass2):
        """Calculate Kinematics for an elastic impact"""
        tmp_speed1 = speed1.copy()
        speed1 = mass1 * speed1 + mass2 * (2*speed2 - speed1)
        speed2 = mass2 * speed2 + mass1 * (2*tmp_speed1 - speed2)

    def initialize_speeds(self):
        """Fills the self.speeds with the starting speeds"""
        self.speeds = np.random.random_sample((N_ENTRIES,2)) * 4

    def initialize_positions(self):
        """Fills the self.positions with the starting positions"""
        self.positions = np.random.random_sample((N_ENTRIES,2)) * np.array([620, 600])
        self.position_IDs = np.vstack((np.arange(0, N_ENTRIES),np.arange(0,N_ENTRIES))).transpose()

    def inititalize_objects(self):
        """Create the initial objects at their start positions,
        give them colors (create self.colors) and diagmeter (self.diameter) and fills objectIDs"""

        self.colors = np.random.randint(0, 16777215, (self.positions.shape[0],), np.uint32)

        self.diameter = np.random.randint(1, 10, (self.positions.shape[0],), np.uint8)

        self.masses = self.diameter.copy() * 10

        self.object_IDs = np.empty((self.positions.shape[0],), dtype=np.uint16)

        for i in range(self.positions.shape[0]):
            self.object_IDs[i] = self.create_oval(self.positions[i, 0], self.positions[i, 1],
                                                  self.positions[i, 0] + self.diameter[i], self.positions[i, 1] + self.diameter[i],
                                                  tags=f"o{i}", fill=f"#{self.colors[i]:06x}")



    @for_all_positional
    def move(self, objID, position, speed, diameter):
        if position[0] >= 600 - diameter or position[0] < 0:
            speed[0] = -speed[0]
        if position[1] >= 620 - diameter or position[1] < 0:
            speed[1] = -speed[1]

        position += speed

        self.coords(objID, int(position[0]), int(position[1]), int(position[0]) + diameter, int(position[1]) + diameter)


    def gameloop(self):
        """Run the game, main loop"""

        # debugging game speed
        self.times.append(time.time_ns())

        # move positions
        self.move(self.object_IDs, self.positions, self.speeds, self.diameter)


        if self.counter < 1000:
            self.counter += 1

            self.after(int(1000 / self.fps), self.gameloop)


        else:
            print(self.times)
            dif = np.diff(np.array(self.times)) / 1000000000
            print(f"{np.mean(dif)} +- {np.std(dif)}")

            self.delete(tk.ALL)

class GameLoop2(tk.Label):
    def __init__(self, fps=30):
        
        self.frame = tk.Frame()
        super().__init__(self.frame)
        self.fps = fps
        self.counter = 0
        self.times = []
        scaler = 400
        self.random_positions = np.random.randint(0, 600, (N_ENTRIES, 2), dtype=np.uint16)
        #print(self.random_positions)
        self.colors = np.random.randint(0, 255, (self.random_positions.shape[0], 3), np.uint8)
        # method 1, create objects
        #self.object_IDs = np.empty((self.random_positions.shape[0],), dtype=int)
        #print(self.random_positions[0])
        #for i in range(self.random_positions.shape[0]):
        #    self.object_IDs[i] = self.create_oval(self.random_positions[i,0] , self.random_positions[i,1], self.random_positions[i,0] + 1, self.random_positions[i,1] + 1, tag=f"obj{i}", fill="#"+("%06x"%np.random.randint(0,16777215)))

        # method 2, create image
        self.pack()
        self.frame.pack()

        self.image_np = np.ones((620, 600, 3), dtype=np.uint8)

        for i, (xx,yy) in enumerate(self.random_positions):
            cc = (int(self.colors[i,0]), int(self.colors[i,1]), int(self.colors[i,2]))
            cv2.circle(self.image_np, (xx,yy), radius=5, color=cc, thickness=-1)
        #print(image[self.random_positions[:,0], self.random_positions[:,1],:].shape)
        #xx,yy = np.mgrid[:620, :600]

        #circles = (xx - self.random_positions[:, 0]) ** 2 + (yy - self.random_positions[:, 1]) ** 2
        #print(circles.shape)
        #print(circles)

        #for i in range(-5,5,1):
        #    for j in range(0, 5 - abs(i), 1):
        #        if j == 0:
        #            image[self.random_positions[:, 0] + i, self.random_positions[:, 0], :] = self.colors#

        #        image[self.random_positions[:, 0] + i, self.random_positions[:, 0] + j, :] = self.colors
        #        image[self.random_positions[:, 0] + i, self.random_positions[:, 0] - j, :] = self.colors

        # use masking
        #print((self.random_positions[:,0]))
        #r1 = np.arange(image.shape[0])
        #r2 = np.arange(image.shape[1])
        #mask1 = (self.random_positions[:,0][:, None] <= r1) & ((self.random_positions[:,0][:, None] + 10) >= r1)
        #mask2 = (self.random_positions[:,1][:, None] <= r2) & ((self.random_positions[:,1][:, None] + 10) >= r2)

        #print(self.random_positions[:,0][:, None])
        #print(r)
        #print(self.random_positions[:,0][:, None] <= r)
        #print((self.random_positions[:,0][:, None] <= r).shape)
        #print(self.random_positions[:, 0][:, None]  + 20 >= r)
        #print((self.random_positions[:,0][:, None] <= r) & (self.random_positions[:, 0][:, None]  + 20 >= r))
        #print(((self.random_positions[:,0][:, None] <= r) & (self.random_positions[:, 0][:, None]  + 20 >= r)).shape)
        #print(mask2.shape)
        #print(len(mask1))
        #print(image.shape)
        #print(image[:len(mask1)][mask1,:].shape)
        #print(image[:len(mask2)][mask2,:].shape)
        #for i in range(self.random_positions.shape[0]):
        #    image[int(self.random_positions[i,0]) : int(self.random_positions[i,0]) +1 ,int(self.random_positions[i,1]) : int(self.random_positions[i,1]) +1,:] = self.colors[i,:]




        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.image_np))
        self.config(image=self.img)
        self.image = self.img


        self.gameloop()

    def gameloop(self):
        self.times.append(time.time_ns())
        for i in range(self.random_positions.shape[0]):
            position = self.random_positions[i,:]
            self.image_np[position[0]-5:position[0]+6, position[1]-5:position[1]+6] = 0

            position += np.ones((2,), dtype=np.uint16)
            if position[0] >= 600:
                position[0] = 0
            if position[1] >= 620:
                position[1] = 0

            cc = (int(self.colors[i, 0]), int(self.colors[i, 1]), int(self.colors[i, 2]))
            cv2.circle(self.image_np, (position[0], position[1]), radius=5, color=cc, thickness=-1)

        # method 2
        #image = np.ones((620, 600, 3), dtype=np.uint8)
        #for i in range(self.random_positions.shape[0]):
        #    image[int(self.random_positions[i,0]) : int(self.random_positions[i,0]) +2 ,int(self.random_positions[i,1]) : int(self.random_positions[i,1]) +2,:] = self.colors[i]
            #self.image[int(self.random_positions[i, 0]) +1: int(self.random_positions[i, 0]) + 3,int(self.random_positions[i, 1]) + 1: int(self.random_positions[i, 1]) + 2, :]

        #image = np.ones((620, 600, 3), dtype=np.uint8)
        #for i, (xx,yy) in enumerate(self.random_positions):
        #    cc = (int(self.colors[i,0]), int(self.colors[i,1]), int(self.colors[i,2]))
        #    cv2.circle(image, (xx,yy), radius=5, color=cc, thickness=-1)
        # print(image[self.random_positions[:,0], self.random_positions[:,1],:].shape)
        #image[self.random_positions[:, 1], self.random_positions[:, 0], :] = self.colors

        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.image_np))
        self.config(image=self.img)
        self.image = self.img
        #   self.coords(self.object_IDs[i], position[0], position[1], position[0] + 1, position[1] + 1)
        if self.counter < 10:
            self.counter += 1

            self.after(1000, self.gameloop)


        else:
            dif = np.diff(np.array(self.times)) / 1000000000
            print(f"{np.mean(dif)} +- {np.std(dif)}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    top = tk.Tk()

    game = GameLoop()

    top.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
