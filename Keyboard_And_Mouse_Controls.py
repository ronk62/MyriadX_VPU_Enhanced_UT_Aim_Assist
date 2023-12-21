# no she-bang in Windows

import time
import keyboard
import math
from pynput.mouse import Button, Controller

mouse = Controller()

'''
ref. https://pynput.readthedocs.io/en/latest/mouse.html#controlling-the-mouse
ref. https://www.unknowncheats.me/forum/apex-legends/495561-calculate-mouse-movement-value.html
'''
gameScrnWidth = 1600
gameScrnHeight = 900
UT99FOV = 75
# UT99FOV = 90
# UT99FOV = 120
UT99sens = 1
UTfull360 = 16363.0 / UT99sens


def click():
    # Press and release
    mouse.press(Button.left)
    mouse.release(Button.left)


def RealFov(fov, width, height):
    raspectRatio = (width / height) / (4/3)
    rFovRad = 2 * math.atan(math.tan(math.radians(fov * 0.5)) * raspectRatio)
    rFovDeg = math.degrees(rFovRad)
    return rFovDeg

def coord2deg (delta, fov, width):
    coordRad = math.atan(((delta * 2) / width) * math.tan(math.radians(fov * 0.5)))
    coordDeg = math.degrees(coordRad)
    return coordDeg

'''
ref.
gameScrnWidth = 1600
gameScrnHeight = 900
UT99FOV = 90
UT99sens = 1
UTfull360 = 16363.0 / UT99sens
'''
def AimMouseAlt(target):
    offsetY, offsetX = target

    realUT99fov = RealFov(UT99FOV, gameScrnWidth, gameScrnHeight)

    yaw = coord2deg(offsetX - gameScrnWidth / 2, realUT99fov, gameScrnWidth)
    pitch = coord2deg(offsetY - gameScrnHeight / 2, realUT99fov, gameScrnWidth)

    y_distance = int((pitch * 1))
    x_distance = int((yaw * 1))

    print("y_distance, x_distance ", y_distance, x_distance)
    # Move pointer relative to current position
    mouse.move(x_distance, y_distance)
    return y_distance, x_distance
