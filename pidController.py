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
        pass

    def computePidOut(self, Kp=1, Ki=0.1, Kd=0.1, errorCurrent=0, initialize=False):
        if initialize:
            self.errorPrevious = 0
            self.errorSum = 0
            self.timePrevious = time.time() - 0.000001
        
        if self.timePrevious == None:
            self.timePrevious = time.time() - 0.000001
        
        # self.dT = time.time() - self.timePrevious
        self.dT = 0.051

        if self.dT < 0.05:
            self.dT = 0.05

        # print("self.dT is ------> ", self.dT)
        # self.errorSum = self.errorSum + (errorCurrent * self.dT)
        self.errorSum = self.errorSum + errorCurrent
        # self.deltaError = (errorCurrent - self.errorPrevious) / (self.dT + 0.000001)
        self.deltaError = errorCurrent - self.errorPrevious
        self.pidOut = (Kp * errorCurrent) + (Ki * self.errorSum) + (Kd * self.deltaError)
        self.errorPrevious = errorCurrent
        self.timePrevious = time.time()
        return self.pidOut

