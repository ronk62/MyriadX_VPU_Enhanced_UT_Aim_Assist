# no she-bang in Windows

import time


'''
pidClass psuedo-code
--------------------
vars: Kp, Ki, Kd, errorCurrent, errorPrevious

instantiate pidY and pidX from pidClass
extract Y and X from 'target' var
pass current Y error to pidY
pass current X error to pidX
recombine Y and X results into 'target'
'''

class PID:
    def __init__(self):
        self.errorPrevious = 0
        self.errorSum = 0

    def computePidOut(self, Kp=1, Ki=0.1, Kd=0.1, errorCurrent=0, initialize=False):
        if initialize:
            self.errorPrevious = 0
            self.errorSum = 0
        

        self.errorSum = self.errorSum + errorCurrent
        if abs(self.errorSum) > 200:     # clamp the errorSum (integral) at 200
            self.errorSum = 200 * (abs(self.errorSum) /  self.errorSum)
        # print("self.errorSum -------------> ", self.errorSum)
        self.deltaError = errorCurrent - self.errorPrevious
        self.pidOut = (Kp * errorCurrent) + (Ki * self.errorSum) + (Kd * self.deltaError)
        self.errorPrevious = errorCurrent
        return self.pidOut

