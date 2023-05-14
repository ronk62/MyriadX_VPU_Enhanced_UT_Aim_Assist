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
UT99FOV = 90
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
def AimMouseAlt(target, xErrorSum, xErrorLast, yErrorSum, yErrorLast):
    offsetY, offsetX = target

    ## add pid controller
    # pid values
    Kp = 1.0
    Ki = 0.0
    Kd = 0.0
    iX = Ki * xErrorSum
    dX = Kd * xErrorLast
    iY = Ki * yErrorSum
    dY = Kd * yErrorLast

    offsetYpid = int((offsetY * Kp) + (offsetY * iY) + (offsetY * dY))  # add pid
    offsetXpid = int((offsetX * Kp) + (offsetX * iX) + (offsetX * dX))  # add pid

    realUT99fov = RealFov(UT99FOV, gameScrnWidth, gameScrnHeight)

    yaw = coord2deg(offsetXpid - gameScrnWidth / 2, realUT99fov, gameScrnWidth)
    pitch = coord2deg(offsetYpid - gameScrnHeight / 2, realUT99fov, gameScrnWidth)

    y_distance = int((pitch * 0.6))
    x_distance = int((yaw * 0.65))

    print("y_distance, x_distance ", y_distance, x_distance)
    # Move pointer relative to current position
    mouse.move(x_distance, y_distance)

'''
save for ref only

def AimMouseAlt(target, xErrorSum, xErrorLast, yErrorSum, yErrorLast):
    offsetY, offsetX = target
    realUT99fov = RealFov(UT99FOV, gameScrnWidth, gameScrnHeight)

    yaw = coord2deg(offsetX - gameScrnWidth / 2, realUT99fov, gameScrnWidth)
    pitch = coord2deg(offsetY - gameScrnHeight / 2, realUT99fov, gameScrnWidth)

    ## add pid controller
    # pid values
    Kp = 0.6
    Ki = 0.0
    Kd = 0.0
    iX = Ki * xErrorSum
    dX = Kd * xErrorLast
    iY = Ki * yErrorSum
    dY = Kd * yErrorLast
    # y_distance = int(pitch * 0.60)
    y_distance = int((pitch * Kp) + (pitch * iY) + (pitch * dY))      # add pid
    # x_distance = int(yaw * 0.65)
    x_distance = int((yaw * Kp) + (yaw * iX) + (yaw * dX))            # add pid

    print("y_distance, x_distance ", y_distance, x_distance)
    # Move pointer relative to current position
    mouse.move(x_distance, y_distance)
'''