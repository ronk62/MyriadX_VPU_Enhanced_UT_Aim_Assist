# no she-bang in Windows

import matplotlib.pyplot as plt
import numpy as np
import time

class PLOT:
    def __init__(self):
        # init np arrays
        self.timestampArray = np.array([], dtype=np.float32)       # array to hold timestamps
        
        self.targetYarray = np.array([], dtype=np.float32)         # array to hold raw screen targetY
        self.errorYarray = np.array([], dtype=np.float32)          # array to hold screen offset errorY
        self.pidTargetYarray = np.array([], dtype=np.float32)      # array to hold pidTargetY
        self.mouseMotionYarray = np.array([], dtype=np.float32)    # array to hold mouseMotionY

        self.targetXarray = np.array([], dtype=np.float32)         # array to hold raw screen targetX
        self.errorXarray = np.array([], dtype=np.float32)          # array to hold screen offset errorX
        self.pidTargetXarray = np.array([], dtype=np.float32)      # array to hold pidTargetX
        self.mouseMotionXarray = np.array([], dtype=np.float32)    # array to hold mouseMotionX


    def updatePlotArrays(currentTargetY, currentTargetX):
        currentTime = time.time()
        ## Update arrays
        timestampArray = np.append(timestampArray, currentTime)
        targetYarray = np.append(targetYarray, currentTargetY)
        targetXarray = np.append(targetXarray, currentTargetX)


    def plotData(timestampArray, targetYarray, targetXarray):
        # convert values in timestampArray to dT
        print("converting values in timestampArray to dT...")
        for i in len(timestampArray):
            timestampArray[i] = timestampArray[i] - timestampArray[0]

        ## Create plots
        # raw target data
        plt.figure(1)
        plt.plot(timestampArray,targetYarray, label='targetYarray')
        plt.plot(timestampArray,targetXarray, label='targetXarray')

        plt.xlabel('dT')
        plt.ylabel('raw screen target values')
        plt.title('target values  ')
        plt.legend()
        plt.show()