import atexit
from threading import Thread

import cv2
import numpy as np

from packages.vision.vision import Vision

# True to continue code runnning
g_running = True

_vision = Vision()

cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
cv2.namedWindow('normal', cv2.WINDOW_NORMAL)

def normalize_depth_array(depth_array):
    normalized_depth = cv2.normalize(depth_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_depth = np.uint8(normalized_depth)
    return normalized_depth

# Run at start of program
def start():
    pass

# Run every loop
def update():
    threads:list[Thread] = []
    threads.append(Thread(target=_vision.update))
    
    threads[-1].start()

    for thread in threads:
        thread.join()
        threads.remove(thread)
    
    cv2.imshow('depth', normalize_depth_array(_vision.SAFE_DEPTH))
    cv2.resizeWindow('depth', 700, 500)
    
    cv2.imshow('normal', _vision.SAFE_IMAGE)
    cv2.resizeWindow('normal', 700, 500)

    cv2.waitKey(1)

# Run on program exit
@atexit.register
def close():
    pass
