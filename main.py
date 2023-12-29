#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
from Keyboard_And_Mouse_Controls import *
from pidController import *
import matplotlib.pyplot as plt
import dxcam

'''
Significantly reduced latency (previously ~500 ms); did this by changing inputFrameShape from (1600, 900) to (416, 416)
- in theory, reducing inputFrameShape minimized the data sent to the oak-d over USB, while retaining the data size/shape
  needed for the yolo model input layer
- also changed the displayed final frame to be the current dxcam capture frame, overlayed with the tracker bbox
- derived the target location via same logic and data as before (trackletsData = track.tracklets)
- one remaining symptom, as of 12/20/2023, is that the tracker gets confused when there is a high-delta
  mouse/camera movement - the tracker bbox shifts in the same direction as mouse, causing overshoot and ocsilations
-- possible remedy is to reinstitute trajectory extrapolation, which averages the delta mouse/cam movements
   and then extrapolates expected target position

Next steps:
- Uncommented the work already done and retune gains -->  introduce process to get speed and trajectory of target (dx, dy, dt) and
  lead the target
- replace my custom and flakey PID controller with easyPID or some such existing lib
- Done --> perhaps add some addaptive error correction and gains (pid controller)
'''

'''
5/11/2023 - test 1 - redo (with correct nn settings for yolo):
- all queues set to maxSize=1, blocking=False
- also notice --> detectionNetwork.setNumInferenceThreads(2)

Results:
- dtCapFrame: 0.010992050170898438 eFPScapFrame: 90.97483121689879
- dtNNdetections: 0.012943744659423828 eFPSnnDetections: 77.25738950007408
- dtTrackletsData: 0.00700068473815918 eFPStrackletsData: 142.84314957165162
- dtImshow: 0.0030007362365722656 eFPSimshow: 333.251438283979
- fullLoopTime: 0.03393721580505371 eFPSfullLoopTime: 29.46617590194039


- UsbSpeed.SUPER
- Latency Det NN: 459.51 ms, Average latency: 459.51 ms, Std: 0.00
- Latency trackFrame: 531.44 ms, Average latency: 495.47 ms, Std: 35.97

'''

'''
ref. https://github.com/ra1nty/DXcam
'''
camera = dxcam.create(device_idx=0, output_idx=1)  # returns a DXCamera instance on primary monitor

pidY = PID()
pidX = PID()


## yolo v3 tiny label texts
labelMap = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

nnPathDefault = str((Path(__file__).parent / Path('./models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to detection network blob", default=nnPathDefault)

args = parser.parse_args()

# Create pipeline
pipeline = dai.Pipeline()

# This might improve reducing the latency on some systems
pipeline.setXLinkChunkSize(0)

# Define sources and outputs
manip = pipeline.create(dai.node.ImageManip)
objectTracker = pipeline.create(dai.node.ObjectTracker)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork) # yolo-v3-tiny-tf

manipOut = pipeline.create(dai.node.XLinkOut)
xinFrame = pipeline.create(dai.node.XLinkIn)
trackerOut = pipeline.create(dai.node.XLinkOut)
xlinkOut = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

manipOut.setStreamName("manip")
xinFrame.setStreamName("inFrame")
xlinkOut.setStreamName("trackerFrame")
trackerOut.setStreamName("tracklets")
nnOut.setStreamName("nn")

# Properties
xinFrame.setMaxDataSize(1920*1080*3)

manip.initialConfig.setResizeThumbnail(416, 416)    # change size to accomodate nn yolo-v3-tiny-tf
# The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manip.inputImage.setBlocking(True)

## Network specific settings for yolo-v3-tiny-tf
detectionNetwork.setBlobPath(args.nnPath)
detectionNetwork.setConfidenceThreshold(0.75)
detectionNetwork.input.setBlocking(True)

detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

objectTracker.inputTrackerFrame.setBlocking(True)
objectTracker.inputDetectionFrame.setBlocking(True)
objectTracker.inputDetections.setBlocking(True)
objectTracker.setDetectionLabelsToTrack([0])  # track only person - yolo-v3-tiny-tf
## possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
# objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)     # primary type used for all dev up to 12/20/2023
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)  # BEST! Low latency and min oscilations (12/22/2023); testing an alternatives to 'ZERO_TERM_COLOR_HISTOGRAM'
# objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_KCF)  # DON'T USE this one; WORST latency (12/22/2023); testing an alternatives to 'ZERO_TERM_COLOR_HISTOGRAM'
# objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)  # WORST oscilations (12/22/2023); testing an alternatives to 'ZERO_TERM_COLOR_HISTOGRAM'
## take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

# Linking
manip.out.link(manipOut.input)
manip.out.link(detectionNetwork.input)
xinFrame.out.link(manip.inputImage)
xinFrame.out.link(objectTracker.inputTrackerFrame)
detectionNetwork.out.link(nnOut.input)
detectionNetwork.out.link(objectTracker.inputDetections)
detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
objectTracker.out.link(trackerOut.input)
objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

# Connect and start the pipeline
with dai.Device(pipeline) as device:

    ## use the below two lines for logging
    # device.setLogLevel(dai.LogLevel.TRACE)
    # device.setLogOutputLevel(dai.LogLevel.TRACE)

    print(device.getUsbSpeed())

    ### blocking=False
    # qIn = device.getInputQueue(name="inFrame", maxSize=1, blocking=False)
    # trackerFrameQ = device.getOutputQueue(name="trackerFrame", maxSize=1, blocking=False)
    # tracklets = device.getOutputQueue(name="tracklets", maxSize=1, blocking=False)
    # qManip = device.getOutputQueue(name="manip", maxSize=1, blocking=False)
    # qDet = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

    ### blocking=True
    qIn = device.getInputQueue(name="inFrame", maxSize=1, blocking=True)
    trackerFrameQ = device.getOutputQueue(name="trackerFrame", maxSize=1, blocking=True)
    tracklets = device.getOutputQueue(name="tracklets", maxSize=1, blocking=True)
    qManip = device.getOutputQueue(name="manip", maxSize=1, blocking=True)
    qDet = device.getOutputQueue(name="nn", maxSize=1, blocking=True)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    detections = []
    frame = None

    
    # init np arrays
    timestampArray = np.array([], dtype=np.float32)       # array to hold timestamps

    targetYarray = np.array([], dtype=np.float32)         # array to hold raw screen targetY
    errorYarray = np.array([], dtype=np.float32)          # array to hold screen offset errorY
    pidTargetYarray = np.array([], dtype=np.float32)      # array to hold pidTargetY
    mouseMotionYarray = np.array([], dtype=np.float32)    # array to hold mouseMotionY
    vectorYarray = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)    # array to hold last n targetYdelta values
    targetYprev = None

    targetXarray = np.array([], dtype=np.float32)         # array to hold raw screen targetX
    errorXarray = np.array([], dtype=np.float32)          # array to hold screen offset errorX
    pidTargetXarray = np.array([], dtype=np.float32)      # array to hold pidTargetX
    mouseMotionXarray = np.array([], dtype=np.float32)    # array to hold mouseMotionX
    vectorXarray = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)    # array to hold last n targetXdelta values
    targetXprev = None


    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.imshow(name, frame)
    

    def capture_window_dxcam():
        # UT game in 1278 x 686 windowed mode
        image = np.array(camera.grab([0, 0, 1600, 900]))
        return image
    
    
    def deltaT(previous_time):
        dt = time.time() - previous_time
        previous_time = time.time()
        return dt, previous_time
        

    baseTs = time.monotonic()
    simulatedFps = 30
    # inputFrameShape = (1920, 1080)
    # inputFrameShape = (1600, 900)
    inputFrameShape = (416, 416)

    diffs = np.array([])

    ### initialize some targeting vars
    trackedTargFrameCount = 0


    while True:
    # for countIter in range(40):         ## for latency testing
        if keyboard.is_pressed(46):     # press and hold 'c' to exit
            print("breaking loop; plotting data...")
            break
        
        time.sleep(0.0333)    # limit to ~30 FPS
        # time.sleep(0.09)    # limit to 1/n FPS
        # time.sleep(1)    # limit to 1/n FPS
        
        previous_time = time.time()
        _, previous_time = deltaT(previous_time)
        initTime = previous_time
        frame = capture_window_dxcam()

        dtCapFrame, previous_time = deltaT(previous_time)
        eFPScapFrame = 1 / (dtCapFrame + 0.000000001)

        if frame.size < 2:  # stop processing this iteration when frame is (essentially) empty
            continue

        inframe = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.BGR888p)
        img.setData(to_planar(inframe, inputFrameShape))
        img.setTimestamp(baseTs)
        baseTs += 1/simulatedFps

        img.setWidth(inputFrameShape[0])
        img.setHeight(inputFrameShape[1])
        qIn.send(img)

        trackFrame = trackerFrameQ.tryGet()
        # trackFrame = trackerFrameQ.get()    ## for latency testing
        if trackFrame is None:
            continue

        track = tracklets.get()
        manip = qManip.get()
        inDet = qDet.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections
        manipFrame = manip.getCvFrame()

        ## for latency testing  <-- note: 12/22/2023, this test does not work correctly with 
        ##                                configurations with frame passed from host to oak-d
        # Latency in miliseconds 
        # latencyMs = (dai.Clock.now() - track.getTimestamp()).total_seconds() * 1000
        # latencyMs = (dai.Clock.now() - trackFrame.getTimestamp()).total_seconds() * 1000
        # diffs = np.append(diffs, latencyMs)
        # print('Latency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}'.format(latencyMs, np.average(diffs), np.std(diffs)))

        displayFrame("nn", manipFrame)
        dtNNdetections, previous_time = deltaT(previous_time)
        eFPSnnDetections = 1 / (dtNNdetections + 0.000000001)

        color = (255, 0, 0)
        trackerFrame = trackFrame.getCvFrame()      # orig/actual trackerFrame
        # trackerFrame = capture_window_dxcam()       # game frame reCapture
        if not np.any(trackerFrame):
            continue

        #####
        trackerFrame = frame
        #####

        trackletsData = track.tracklets

        trackedCount = 0

        for t in trackletsData:
            roi = t.roi.denormalize(trackerFrame.shape[1], trackerFrame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            try:
                label = labelMap[t.label]
            except:
                label = t.label

            tStatusName = t.status.name

            if tStatusName == 'TRACKED':
                trackedCount += 1
                # trackedTargFrameCount += 1

                cv2.putText(trackerFrame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(trackerFrame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(trackerFrame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(trackerFrame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                bbox_height = (y2 - y1)
                bbox_ycenter = y1 + bbox_height/2
                
                bbox_width = (x2 - x1)
                bbox_xcenter = x1 + bbox_width/2

                ## uncomment following 3 lines to ignore downed targets (based on aspect ratio)
                if bbox_height < 1.1 * bbox_width:
                    print("ignore downed targets based on aspect ratio", (bbox_height, 1.1 * bbox_width))
                    continue
                
                ## uncomment following 6 lines for ROI targeting use (else, full-screen targeting)
                # if bbox_ycenter < 200 or bbox_ycenter > 700:
                #     print("bbox_ycenter not withing RIO")
                #     continue
                # if bbox_xcenter < 450 or bbox_xcenter > 1150:
                #     print("bbox_xcenter not withing RIO")
                #     continue

                    


                target = (int(y1 + 0.25 * bbox_height), int(bbox_xcenter))      # target based on raw detection and track bbox

                print("target before pids applied = ", target)

                targetY, targetX = target

                #####################
                gameScrnWidth = 1600
                gameScrnHeight = 900

                ## extrapolation code, compensate for Obj detection and tracking lag
                
                # init prev vars for this run
                if targetYprev == None:
                    targetYprev = targetY
                
                if targetXprev == None:
                    targetXprev = targetX
                

                ######

                # # calculate deltas
                # targetYdelta =  targetY - targetYprev
                # targetXdelta = targetX - targetXprev
                # print("targetYdelta, targetXdelta before extrapolation ", targetYdelta, targetXdelta)

                # save current target values as previous for next cycle
                targetYprev = targetY
                targetXprev = targetX

                # # do rolling update of arrays
                # np.roll(vectorYarray, -1)
                # vectorYarray[50] = targetYdelta
                # np.roll(vectorXarray, -1)
                # vectorXarray[50] = targetXdelta
                
                # # calculate avg and extrapoate
                # targetYdeltaavg = np.average(vectorYarray)
                # # targetYdelta = targetYdelta + 0.45 * targetYdeltaavg
                # targetYdelta = targetYdelta + 5 * targetYdeltaavg
                # targetXdeltaavg = np.average(vectorXarray)
                # # targetXdelta = targetXdelta + 0.85 * targetXdeltaavg
                # targetXdelta = targetXdelta + 5 * targetXdeltaavg

                # # limit max Y delta to 0.4 of (gameScrnHeight / 2)
                # if targetYdelta > 0.4 * (gameScrnHeight / 2):
                #     targetYdelta = 0.4 * (gameScrnHeight / 2)
                
                # # limit max X delta to 0.55 of (gameScrnWidth / 2)
                # if targetXdelta > 0.55 * (gameScrnWidth / 2):
                #     targetXdelta = 0.55 * (gameScrnWidth / 2)
                
                # print("targetYdelta, targetXdelta AFTER extrapolation ", targetYdelta, targetXdelta)

                # errorY = targetY - ((gameScrnHeight / 2) + targetYdelta)
                # errorX = targetX - ((gameScrnWidth / 2) + targetXdelta)
                
                # print("calculated errorY = ", errorY)
                # print("calculated errorX = ", errorX)


                '''
                pidClass psuedo-code
                --------------------
                vars: Kp, Ki, Kd, error, previousError

                instantiate pidY and pidX from pidClass
                extract Y and X from 'target' var
                pass current Y error to pidY
                pass current X error to pidX
                recombine Y and X results into 'target'
                '''

                # ## best PID and scale tuning pre-11/11/2023
                # KpY = 0.55  # 0.8, 0.75, 0.85, 0.9, 0.85, 1
                # KiY = 0.02  # 0.18, 0.09, 0.07,  0.05
                # KdY = 0.02  # 0.03, 0.02, 0.009, 0, 0, 1

                # KpX = 0.55  # 0.8, 0.75, 0.85, 0.9, 0.85, 1
                # KiX = 0.02  # 0.18, 0.09, 0.07, 0.05
                # KdX = 0.02  # 0.03, 0.02, 0.009, 0, 0, 1

                # ScaleY = 0.035        # trial and error testing
                # ScaleX = 0.045        # trial and error testing

                # ## experimental PID and scale tuning post-11/11/2023
                # KpY = 0.55  # 0.8, 0.75, 0.85, 0.9, 0.85, 1
                # KiY = 0.01  # 0.18, 0.09, 0.07,  0.05
                # KdY = 0.01  # 0.03, 0.02, 0.009, 0, 0, 1

                # KpX = 0.55  # 0.8, 0.75, 0.85, 0.9, 0.85, 1
                # KiX = 0.01  # 0.18, 0.09, 0.07, 0.05
                # KdX = 0.01  # 0.03, 0.02, 0.009, 0, 0, 1

                # ScaleY = 0.0562           # trial and error testing
                # ScaleX = 0.0562           # trial and error testing

                
                if keyboard.is_pressed(45):     # press and hold 'x' to target and fire
                    trackedTargFrameCount += 1

                    # if trackedTargFrameCount == 1:
                    #     pidTargetY = pidY.computePidOut(KpY, KiY, KdY, errorY, True)
                    #     pidTargetX = pidX.computePidOut(KpX, KiX, KdX, errorX, True)
                    # else:
                    #     pidTargetY = pidY.computePidOut(KpY, KiY, KdY, errorY, False)
                    #     pidTargetX = pidX.computePidOut(KpX, KiX, KdX, errorX, False)
                    

                    ## Update arrays
                    currentTime = time.time()
                    timestampArray = np.append(timestampArray, currentTime)
                    targetYarray = np.append(targetYarray, targetY)
                    targetXarray = np.append(targetXarray, targetX)
                    # errorYarray = np.append(errorYarray, errorY)
                    # errorXarray = np.append(errorXarray, errorX)
                    # pidTargetYarray = np.append(pidTargetYarray, pidTargetY)
                    # pidTargetXarray = np.append(pidTargetXarray, pidTargetX)
                    
                    # target = (pidTargetY, pidTargetX)
                    # print("target after pids applied = ", target)
                    
                    # mouseMotionY = ScaleY * pidTargetY
                    # mouseMotionX = ScaleX * pidTargetX
                    # print("scaled mouseMotionY, mouseMotionX ", mouseMotionY, mouseMotionX)

                    # ## Update arrays
                    # mouseMotionYarray = np.append(mouseMotionYarray, mouseMotionY)
                    # mouseMotionXarray = np.append(mouseMotionXarray, mouseMotionX)


                    ## move mouse to point at target
                    mouseMotionY, mouseMotionX =AimMouseAlt(target)
                    # # Move pointer relative to current position
                    # mouse.move(mouseMotionX, mouseMotionY)
                    
                    if 400 < targetY < 500 and 750 < targetX < 850: # fire only when on-target
                        # fire at target 3 times
                        click()
                        click()
                        click()

                    ## Update vars and arrays
                    # previousTrackTime = time.time()     # use to calculate dT between track frames
                    mouseMotionYarray = np.append(mouseMotionYarray, mouseMotionY)
                    mouseMotionXarray = np.append(mouseMotionXarray, mouseMotionX)
                
                else:
                    trackedTargFrameCount = 0

        if trackedCount == 0:
            trackedTargFrameCount = 0
            vectorYarray = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)
            vectorXarray = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)
            targetYprev = None
            targetXprev = None


        cv2.putText(trackerFrame, "Fps: {:.2f}".format(fps), (2, trackerFrame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        dtTrackletsData, previous_time = deltaT(previous_time)
        eFPStrackletsData = 1 / (dtTrackletsData + 0.000000001)

        if frame.size > 1:
            # cv2.imshow("tracker", trackerFrame)
            trackerFrame = cv2.cvtColor(trackerFrame, cv2.COLOR_RGB2BGR)
            cv2.imshow("origFrameWithBBox", trackerFrame)

            if cv2.waitKey(1) == ord('q'):
                break

        dtImshow, previous_time = deltaT(previous_time)
        eFPSimshow = 1 / (dtImshow + 0.000000001)

        
        fullLoopTime = time.time() - initTime
        eFPSfullLoopTime = 1 / (fullLoopTime + 0.000000001)


        # print("dtCapFrame:", dtCapFrame, "eFPScapFrame:", eFPScapFrame)
        # print("dtNNdetections:", dtNNdetections, "eFPSnnDetections:", eFPSnnDetections)
        # print("dtTrackletsData:", dtTrackletsData, "eFPStrackletsData:", eFPStrackletsData)
        # print("dtImshow:", dtImshow, "eFPSimshow:", eFPSimshow)
        # print("fullLoopTime:", fullLoopTime, "eFPSfullLoopTime:", eFPSfullLoopTime)

        # ## break after first 25 tracks
        # if len(timestampArray) == 25:
        #     break
    

    ### plot the data, then exit
    # # convert values in timestampArray to dT
    # print("converting values in timestampArray to dT...")
    # for i in range(len(timestampArray)):
    #     timestampArray[i] = timestampArray[i] - timestampArray[0]

    ## Create plots
    # raw-target values
    plt.figure(1)
    plt.plot(timestampArray,targetYarray, label='targetYarray')
    plt.plot(timestampArray,targetXarray, label='targetXarray')
    # plt.plot(timestampArray,errorYarray, label='errorYarray')
    # plt.plot(timestampArray,errorXarray, label='errorXarray')

    plt.xlabel('dT')
    plt.ylabel('raw-target values')
    plt.title('target values over time')
    plt.legend()
    # plt.show()

    # # pid-target values
    # plt.figure(2)
    # plt.plot(timestampArray,pidTargetYarray, label='pidTargetYarray')
    # plt.plot(timestampArray,pidTargetXarray, label='pidTargetXarray')

    # plt.xlabel('dT')
    # plt.ylabel('pid-target values')
    # plt.title('pid-target values over time')
    # plt.legend()
    # # plt.show()

    # mouseMotion values
    plt.figure(3)
    plt.plot(timestampArray,mouseMotionYarray, label='mouseMotionYarray')
    plt.plot(timestampArray,mouseMotionXarray, label='mouseMotionXarray')

    plt.xlabel('dT')
    plt.ylabel('mouseMotion values')
    plt.title('mouseMotion values over time')
    plt.legend()
    plt.show()

    print("exiting...")
    exit()