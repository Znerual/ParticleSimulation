import tkinter as tk
import platform

from gameloop1 import GameLoop

N_ENTRIES = 5
HEIGHT = 256
WIDTH = 256
TABLE_WIDTH = 256

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    major, minor, patchlevel = platform.python_version_tuple()
    if int(major) != 3:
        raise Exception("Require python 3.XX")


    top = tk.Tk()

    game = GameLoop(top, HEIGHT, WIDTH, N_ENTRIES)

    top.mainloop()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
