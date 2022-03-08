import tkinter as tk
import numpy as np
import time
import cv2

from decorators import *
from main import HEIGHT, WIDTH, N_ENTRIES

class GameLoop2(tk.Label):
    def __init__(self, fps=30):

        self.frame = tk.Frame()
        super().__init__(self.frame)
        self.fps = fps
        self.counter = 0
        self.times = []
        scaler = 400
        self.random_positions = np.random.randint(0, min(HEIGHT, WIDTH), (N_ENTRIES, 2), dtype=np.uint16)
        self.colors = np.random.randint(0, 255, (self.random_positions.shape[0], 3), np.uint8)
        # method 1, create objects
        # self.object_IDs = np.empty((self.random_positions.shape[0],), dtype=int)
        # print(self.random_positions[0])
        # for i in range(self.random_positions.shape[0]):
        #    self.object_IDs[i] = self.create_oval(self.random_positions[i,0] , self.random_positions[i,1], self.random_positions[i,0] + 1, self.random_positions[i,1] + 1, tag=f"obj{i}", fill="#"+("%06x"%np.random.randint(0,16777215)))

        # method 2, create image
        self.pack()
        self.frame.pack()

        self.image_np = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8)

        for i, (xx, yy) in enumerate(self.random_positions):
            cc = (int(self.colors[i, 0]), int(self.colors[i, 1]), int(self.colors[i, 2]))
            cv2.circle(self.image_np, (xx, yy), radius=5, color=cc, thickness=-1)




        # use masking
        # print((self.random_positions[:,0]))
        # r1 = np.arange(image.shape[0])
        # r2 = np.arange(image.shape[1])
        # mask1 = (self.random_positions[:,0][:, None] <= r1) & ((self.random_positions[:,0][:, None] + 10) >= r1)
        # mask2 = (self.random_positions[:,1][:, None] <= r2) & ((self.random_positions[:,1][:, None] + 10) >= r2)


        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.image_np))
        self.config(image=self.img)
        self.image = self.img

        self.gameloop()

    def gameloop(self):
        self.times.append(time.time_ns())
        for i in range(self.random_positions.shape[0]):
            position = self.random_positions[i, :]
            self.image_np[position[0] - 5:position[0] + 6, position[1] - 5:position[1] + 6] = 0

            position += np.ones((2,), dtype=np.uint16)
            if position[0] >= 600:
                position[0] = 0
            if position[1] >= 620:
                position[1] = 0

            cc = (int(self.colors[i, 0]), int(self.colors[i, 1]), int(self.colors[i, 2]))
            cv2.circle(self.image_np, (position[0], position[1]), radius=5, color=cc, thickness=-1)

        # method 2
        # image = np.ones((620, 600, 3), dtype=np.uint8)
        # for i in range(self.random_positions.shape[0]):
        #    image[int(self.random_positions[i,0]) : int(self.random_positions[i,0]) +2 ,int(self.random_positions[i,1]) : int(self.random_positions[i,1]) +2,:] = self.colors[i]
        # self.image[int(self.random_positions[i, 0]) +1: int(self.random_positions[i, 0]) + 3,int(self.random_positions[i, 1]) + 1: int(self.random_positions[i, 1]) + 2, :]


        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.image_np))
        self.config(image=self.img)
        self.image = self.img
        if self.counter < 10:
            self.counter += 1

            self.after(1000, self.gameloop)


        else:
            dif = np.diff(np.array(self.times)) / 1000000000
            print(f"{np.mean(dif)} +- {np.std(dif)}")
