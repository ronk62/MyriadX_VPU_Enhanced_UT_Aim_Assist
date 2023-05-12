#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
# from PIL import ImageGrab, Image
import dxcam

camera = dxcam.create(device_idx=0, output_idx=1)  # returns a DXCamera instance on primary monitor

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
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
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

    print(device.getUsbSpeed())

    qIn = device.getInputQueue(name="inFrame", maxSize=1, blocking=False)
    trackerFrameQ = device.getOutputQueue(name="trackerFrame", maxSize=1, blocking=False)
    tracklets = device.getOutputQueue(name="tracklets", maxSize=1, blocking=False)
    qManip = device.getOutputQueue(name="manip", maxSize=1, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    detections = []
    frame = None

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
    inputFrameShape = (1600, 900)

    diffs = np.array([])

    while True:
        previous_time = 0
        initTime, previous_time = deltaT(previous_time)
        # frame = capture_window_PIL()
        frame = capture_window_dxcam()
        dtCapFrame, previous_time = deltaT(previous_time)
        eFPScapFrame = 1 / (dtCapFrame + 0.000000001)

        if frame.size < 2:  # stop processing this iteration when frame is (essentially) empty
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.BGR888p)
        img.setData(to_planar(frame, inputFrameShape))
        img.setTimestamp(baseTs)
        baseTs += 1/simulatedFps

        img.setWidth(inputFrameShape[0])
        img.setHeight(inputFrameShape[1])
        qIn.send(img)

        trackFrame = trackerFrameQ.tryGet()
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

        ## Show Latency in miliseconds 
        # latencyMs = (dai.Clock.now() - manip.getTimestamp()).total_seconds() * 1000
        # diffs = np.append(diffs, latencyMs)
        # print('Latency Det NN: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}'.format(latencyMs, np.average(diffs), np.std(diffs)))
 
        displayFrame("nn", manipFrame)
        dtNNdetections, previous_time = deltaT(previous_time)
        eFPSnnDetections = 1 / (dtNNdetections + 0.000000001)

        color = (255, 0, 0)
        trackerFrame = trackFrame.getCvFrame()
        trackletsData = track.tracklets
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

            ####
            tStatusName = t.status.name

            if tStatusName == 'TRACKED':

                cv2.putText(trackerFrame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(trackerFrame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(trackerFrame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(trackerFrame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            ####


        cv2.putText(trackerFrame, "Fps: {:.2f}".format(fps), (2, trackerFrame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        dtTrackletsData, previous_time = deltaT(previous_time)
        eFPStrackletsData = 1 / (dtTrackletsData + 0.000000001)

        ## Show Latency in miliseconds 
        # latencyMs = (dai.Clock.now() - trackFrame.getTimestamp()).total_seconds() * 1000
        # diffs = np.append(diffs, latencyMs)
        # print('Latency trackFrame: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}'.format(latencyMs, np.average(diffs), np.std(diffs)))

        ## use with dxcam version
        if frame.size > 1:
            cv2.imshow("tracker", trackerFrame)

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