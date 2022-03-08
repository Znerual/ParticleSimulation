import tkinter as tk
import typing

import numpy as np
import time

from math import sqrt
from numpy.linalg import norm
from tkinter import ttk

from ParticleSimulation.decorators import for_all_positional


class GameLoop(tk.Canvas):
    def __init__(self, root, HEIGHT, WIDTH, N_ENTRIES, fps=30):
        super().__init__(width=WIDTH, height=HEIGHT, background="black", highlightthickness=0)

        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.N_ENTRIES = N_ENTRIES

        self.root = root

        # game parameter
        self.fps = fps

        # debug variables
        self.counter = 0
        self.times = []

        # game variables
        self.positions: np.ndarray[typing.Tuple[int, int], np.dtype[np.float64]] = np.empty((N_ENTRIES, 2),
                                                                                            dtype=np.float64)
        self.position_IDs: np.ndarray[typing.Tuple[int], np.dtype[np.uint16]] = np.empty((N_ENTRIES,), dtype=np.uint16)
        self.lines: np.ndarray[typing.Tuple[int], np.dtype[np.uint16]] = np.empty((N_ENTRIES,), dtype=np.uint16)
        self.speeds: np.ndarray[typing.Tuple[int, int], np.dtype[np.float64]] = np.empty((N_ENTRIES, 2),
                                                                                         dtype=np.float64)
        self.object_IDs: np.ndarray[typing.Tuple[int], np.dtype[np.uint16]] = np.empty((N_ENTRIES,), dtype=np.uint16)
        self.colors: np.ndarray[typing.Tuple[int], np.dtype[np.uint32]] = np.empty((N_ENTRIES,), dtype=np.uint32)
        self.diameters: np.ndarray[typing.Tuple[int], np.dtype[np.int32]] = np.empty((N_ENTRIES,), dtype=np.int32)
        self.masses: np.ndarray[typing.Tuple[int], np.dtype[np.int32]] = np.empty((N_ENTRIES,), dtype=np.int32)
        self.table = ttk.Treeview(self.root)


        self.times = [time.time()]
        # initialize simulation
        self.initialize_positions()
        self.initialize_speeds()
        self.inititalize_objects()
        self.fix_init_positions()
        self.initialize_lines()
        self.create_position_speed_table()
        # populate GUI
        self.pack()

        # start Gameloop
        self.gameloop()

    def create_position_speed_table(self):
        """Create a table showing positions"""

        self.table.pack(side="right")

        self.table['columns'] = ("Position X", "Position Y", "Speed X", "Speed Y")
        self.table.column("#0", width=0, stretch=tk.NO)
        self.table.column("Position X", anchor=tk.CENTER, width=80)
        self.table.column("Position Y", anchor=tk.CENTER, width=80)
        self.table.column("Speed X", anchor=tk.CENTER, width=80)
        self.table.column("Speed Y", anchor=tk.CENTER, width=80)

        self.table.heading("#0", text="", anchor=tk.CENTER)
        self.table.heading("Position X", text="Position X", anchor=tk.CENTER)
        self.table.heading("Position Y", text="Position Y", anchor=tk.CENTER)
        self.table.heading("Speed X", text="Speed X", anchor=tk.CENTER)
        self.table.heading("Speed Y", text="Speed Y", anchor=tk.CENTER)

        for obj in range(self.object_IDs.shape[0]):
            self.table.insert(parent='', index='end', iid=obj, text='',
                              values=(round(self.positions[obj, 0], 1), round(self.positions[obj, 1], 1),
                                      round(self.speeds[obj, 0], 1), round(self.speeds[obj, 0], 1)))

    def update_table(self):
        for obj in range(self.object_IDs.shape[0]):
            self.table.item(obj, values=(
                round(self.positions[obj, 0], 1), round(self.positions[obj, 1], 1), round(self.speeds[obj, 0], 1),
                round(self.speeds[obj, 0], 1)))

    def check_static_collision(self, objID1, objID2):
        """Check if the two circles collide"""
        center1 = self.positions[objID1] + np.array([0.5, 0.5]) * self.diameters[objID1]
        center2 = self.positions[objID2] + np.array([0.5, 0.5]) * self.diameters[objID2]

        collide = np.sum((center1 - center2) ** 2) <= ((self.diameters[objID1] + self.diameters[objID2]) * 0.5) ** 2

        return collide

    def check_dynamic_collition(self, objID1, objID2):
        """Check if a collision would happen in between two frames, Real Time Rendering 4th edition"""

        r1 = self.diameters[objID1]
        r2 = self.diameters[objID2]
        v1 = self.speeds[objID1]
        v2 = self.speeds[objID2]
        center1 = self.positions[objID1] + np.array([0.5, 0.5]) * r1
        center2 = self.positions[objID2] + np.array([0.5, 0.5]) * r2

        c = np.dot((v2 - v1), (center1 - center2) / norm((center1 - center2), 2))

        return norm((center1 - center2), 2) - c - (r1 + r2) * 0.5 < 0

    def brute_force(self, objIDs):
        collisions = []
        for i in range(len(objIDs)):
            for j in range(i + 1, len(objIDs)):
                if self.check_dynamic_collition(i, j):  # and not col1 in moved_already and not col2 in moved_already:
                    collisions.append([i, j])


        return collisions

    def sort_and_sweep(self, positions, diameter, position_IDs):
        """Do a sort and sweep to find colliding objects. Sorting is applied to the position_IDs, positions is not changed!"""

        # get axis with most variance
        var_x = np.var(positions[:, 0])
        var_y = np.var(positions[:, 1])
        selected_axis = int(var_y > var_x)

        # get start of object on axis and end of object on axis, add the current speed to account for dynamic collision matching
        start_positions = positions[:, selected_axis] + np.minimum(self.speeds[:, selected_axis], 0)
        end_positions = (positions[:, selected_axis] + diameter) + np.maximum(self.speeds[:, selected_axis], 0)

        # print(self.speeds[:, selected_axis])
        # print(list(zip(start_positions, end_positions, diameter, self.speeds[:, selected_axis])))
        # combine to one array, with start and end points, alternatingly combined,
        # use order of position_ids to create in order to use previous sorting
        all_positions = np.empty((positions.shape[0] * 2,), dtype=np.float64)
        all_positions[::2] = np.take(start_positions, position_IDs[:, selected_axis])
        all_positions[1::2] = np.take(end_positions, position_IDs[:, selected_axis])

        # sort
        sort_indices = np.argsort(all_positions)

        # go over sorting result, even numbers are start positions, odd end positions
        # operation_stack = []
        active_intervals = []
        possible_collisions = []
        even_counter = 0
        for ind in sort_indices:
            # even number = start position
            if ind % 2 == 0:
                active_intervals.append(ind >> 1)  # bitshift right to divide by two
                position_IDs[even_counter, selected_axis] = ind >> 1
                even_counter += 1

            # odd number = end position
            else:

                # only current starting interval in list
                if len(active_intervals) == 1 and active_intervals[0] == (ind - 1) >> 1:
                    active_intervals = []

                # other elements in list, therefore collision possible
                else:
                    try:
                        possible_collisions.append(active_intervals.copy())
                        active_intervals.remove((ind - 1) >> 1)
                    except ValueError as exc:
                        print(self.positions[(ind - 1) >> 1])
                        print(self.diameters[(ind - 1) >> 1])
                        raise ValueError(exc)

        collisions = []
        moved_already = []
        # go over all possible collisions
        for possible_collision in possible_collisions:

            # only two possible for this pair
            if len(possible_collision) == 2:
                col1, col2 = possible_collision
                if self.check_dynamic_collition(col1, col2) and col1 not in moved_already and col2 not in moved_already:
                    collisions.append([col1, col2])

                    moved_already.extend([col1, col2])

            # small number of elements
            else:
                for i in range(len(possible_collision)):
                    col1 = possible_collision[i]
                    for j in range(i + 1, len(possible_collision)):
                        col2 = possible_collision[j]
                        if self.check_dynamic_collition(col1,
                                                        col2) and col1 not in moved_already and col2 not in moved_already:
                            collisions.append([col1, col2])

                            moved_already.extend([col1, col2])

        return collisions

    def initialize_lines(self):
        """Draw lines in movement direction"""
        self.lines = np.empty((self.object_IDs.shape[0],), dtype=int)
        for i in range(self.object_IDs.shape[0]):
            self.lines[i] = self.create_line(self.positions[i][0], self.positions[i][1],
                                             self.positions[i][0] + self.speeds[i][0] * 100,
                                             self.positions[i][1] + self.speeds[i][1] * 100, fill="white")

    def hit(self, objectID1, objectID2):
        """Calculate Kinematics for an elastic impact"""

        def calc_vec():
            norm_vec = (self.positions[objectID2] + np.array(
                [self.diameters[objectID2] * 0.5, self.diameters[objectID2] * 0.5]) - self.positions[
                            objectID1] - np.array(
                [self.diameters[objectID1] * 0.5, self.diameters[objectID1] * 0.5])).copy()
            norm_vec = norm_vec / sqrt(norm_vec[0] ** 2 + norm_vec[1] ** 2)
            tang_vec = np.array([norm_vec[1], -norm_vec[0]], dtype=np.float64)

            v1_norm = np.dot(self.speeds[objectID1], norm_vec)
            v2_norm = np.dot(self.speeds[objectID2], norm_vec)
            v1_tang = np.dot(self.speeds[objectID1], tang_vec)
            v2_tang = np.dot(self.speeds[objectID2], tang_vec)

            self.speeds[objectID1] = tang_vec * v1_tang + ((v1_norm * (
                    self.masses[objectID1] - self.masses[objectID2]) + 2 * self.masses[objectID2] * v2_norm) / (
                    self.masses[objectID1] + self.masses[objectID2])) * norm_vec

            self.speeds[objectID2] = tang_vec * v2_tang + ((v2_norm * (
                    self.masses[objectID2] - self.masses[objectID1]) + 2 * self.masses[objectID1] * v1_norm) / (
                    self.masses[objectID1] + self.masses[objectID2])) * norm_vec

        calc_vec()
        print(f"hit {objectID1}, {objectID2}")

    def fix_init_positions(self):
        """Moved the starting objects to not sit on top of each other"""
        collisions = self.sort_and_sweep(self.positions, self.diameters, self.position_IDs)
        for col in collisions:
            col1, col2 = col

            # create unit dif vector
            dif_vec = self.positions[col2] + np.array([self.diameters[col2] * 0.5, self.diameters[col2] * 0.5]) - \
                self.positions[col1] - np.array([self.diameters[col1] * 0.5, self.diameters[col1] * 0.5])
            dif_length = norm(dif_vec, 2)
            dif_vec /= dif_length

            # move distance
            move_distance = (self.diameters[col1] + self.diameters[col2]) * 0.25

            self.positions[col1] -= move_distance * dif_vec
            self.positions[col2] += move_distance * dif_vec

    def initialize_speeds(self):
        """Fills the self.speeds with the starting speeds"""
        self.speeds = np.random.random_sample((self.N_ENTRIES, 2)) * 0.5

    def initialize_positions(self):
        """Fills the self.positions with the starting positions"""
        self.positions = np.random.random_sample((self.N_ENTRIES, 2)) * np.array([self.WIDTH, self.HEIGHT])
        self.diameters = np.random.randint(5, 25, (self.positions.shape[0],), np.int32)

        for pos, dia in zip(self.positions, self.diameters):
            if pos[0] - dia < 0:
                pos[0] += dia
            elif pos[0] + dia > self.WIDTH:
                pos[0] -= dia

            if pos[1] - dia < 0:
                pos[1] += dia
            elif pos[1] + dia > self.HEIGHT:
                pos[1] -= dia

        self.position_IDs = np.vstack((np.arange(0, self.N_ENTRIES), np.arange(0, self.N_ENTRIES))).transpose()

    def inititalize_objects(self):
        """Create the initial objects at their start positions,
        give them colors (create self.colors) and diagmeter (self.diameter) and fills objectIDs"""

        self.colors = np.random.randint(0, 16777215, (self.positions.shape[0],), np.uint32)

        self.masses = self.diameters.copy() * 10

        self.object_IDs = np.empty((self.positions.shape[0],), dtype=np.uint16)

        for i in range(self.positions.shape[0]):
            self.object_IDs[i] = self.create_oval(self.positions[i, 0], self.positions[i, 1],
                                                  self.positions[i, 0] + self.diameters[i],
                                                  self.positions[i, 1] + self.diameters[i],
                                                  tags=f"o{i}", fill=f"#{self.colors[i]:06x}")

    @for_all_positional
    def move(self, objID, position, speed, diameter, lines):
        if position[0] >= self.WIDTH - diameter or position[0] < 0:
            speed[0] = -speed[0]
        if position[1] >= self.HEIGHT - diameter or position[1] < 0:
            speed[1] = -speed[1]

        position += speed

        self.coords(objID, int(position[0]), int(position[1]), int(position[0]) + diameter, int(position[1]) + diameter)
        self.coords(lines,
                    int(position[0] + diameter * 0.5), int(position[1] + diameter * 0.5),
                    int((position[0] + diameter * 0.5) + 30 * speed[0]),
                    int((position[1] + diameter * 0.5) + 30 * speed[1]))
        self.update_table()

    def gameloop(self):
        """Run the game, main loop"""

        # debugging game speed

        self.times.append(time.time() )
        # move positions
        collisions = self.sort_and_sweep(self.positions, self.diameters, self.position_IDs)
        for col in collisions:
            self.hit(*col)
        self.move(self.object_IDs, self.positions, self.speeds, self.diameters, self.lines)

        if self.counter < 1000000:
            self.counter += 1
            delay = (self.times[-1] - self.times[-2]) / 1000
            wait = max(0, 1000 / self.fps - delay)
            self.after(int(wait), self.gameloop)

        else:
            print(self.times)
            dif = np.diff(np.array(self.times)) / 1000000
            print(f"{np.mean(dif)} +- {np.std(dif)}")

            self.delete(tk.ALL)
