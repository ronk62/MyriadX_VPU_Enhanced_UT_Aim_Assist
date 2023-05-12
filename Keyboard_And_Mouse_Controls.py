# no she-bang in Windows

# ref: C:\Users\ronk6\OneDrive\Documents\PythonProjects\winLocalPython\Halo_Aim_Assistant-master_g-make-it

import ctypes
import time
## unused?
# import win32com.client as comclt
import win32api, win32con, win32gui, win32ui
import keyboard
import math

'''
ref. https://www.unknowncheats.me/forum/apex-legends/495561-calculate-mouse-movement-value.html
'''
gameScrnWidth = 1600
gameScrnHeight = 900
UT99FOV = 90
UT99sens = 1
UTfull360 = 16363.0 / UT99sens

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def click():
    # win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
    time.sleep(0.001)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)

'''
def AimMouseAlt(target):
    offsetY, offsetX = target
    y_distance = int(((offsetY - 400)/400) * 40)
    x_distance = int(((offsetX - 750)/750) * 80)
    print(y_distance, x_distance)
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_distance, y_distance, 0, 0)
'''

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

    # y_distance = int(pitch * (UTfull360 / 360))
    y_distance = int(pitch)
    # x_distance = int(yaw * (UTfull360 / 360))
    x_distance = int(yaw)
    print(y_distance, x_distance)
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_distance, y_distance, 0, 0)