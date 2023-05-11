#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
from PIL import ImageGrab, Image
import dxcam

camera = dxcam.create(device_idx=0, output_idx=1)  # returns a DXCamera instance on primary monitor

'''
5/9/2023: 

performance while using person-detection-retail-0013_openvino_2021.4_6shave.blob

dtCapFrame: 0.050776004791259766 eFPScapFrame: 19.694341538225782
dtNNdetections: 0.013944864273071289 eFPSnnDetections: 71.71098324851401
dtTrackletsData: 0.009974956512451172 eFPStrackletsData: 100.25105357609362
dtImshow: 0.004987478256225586 eFPSimshow: 200.50208705164175
fullLoopTime: 0.07968330383300781 eFPSfullLoopTime: 12.549680446178519

-> approx 1.5-2 seconds latency (final 'imshow' frame is 1-2s behind game)

'''

'''
5/10/2023 - test 1:
- remove all steps in while loop other than...
    -- frame grab capture_window function (and timing capture)
    -- full loop timing capture

Results:
- dtCapFrame: 0.0466306209564209 eFPScapFrame: 21.4451353647944
- fullLoopTime: 0.0466306209564209 eFPSfullLoopTime: 21.44513582468824

'''

'''
5/10/2023 - test 2:
- remove all steps in while loop other than frame grab capture_window
- replace the PIL-based frame grab capture_window function with https://github.com/ra1nty/DXcam
    -- show new verion frame grab (capture_window_dxcam) timing capture
    -- full loop timing capture

Results:
- dtCapFrame: 0.0009975433349609375 eFPScapFrame: 1002.4617101746749
- fullLoopTime: 0.0009975433349609375 eFPSfullLoopTime: 1002.4617101746749

'''

'''
5/10/2023 - test 3:
- replace the PIL-based frame grab capture_window function with https://github.com/ra1nty/DXcam
- restore all steps in while loop
    -- show all timing capture elements

Results:
- dtCapFrame: 0.016281604766845703 eFPScapFrame: 61.41900340298768
- dtNNdetections: 0.025780200958251953 eFPSnnDetections: 38.789455630308254
- dtTrackletsData: 0.011970758438110352 eFPStrackletsData: 83.53688879724538
- dtImshow: 0.003987312316894531 eFPSimshow: 250.79544056970104
- fullLoopTime: 0.05801987648010254 eFPSfullLoopTime: 17.2354724524012

-> approx 1.5-2 seconds latency...no improvement (final 'imshow' frame is still 1-2s behind game)

'''

'''
5/11/2023 - test 1:
- all queues set to maxSize=1, blocking=False

Results:
- dtCapFrame: 0.013970613479614258 eFPScapFrame: 71.57881290469987
- dtNNdetections: 0.018477916717529297 eFPSnnDetections: 54.11865207362289
- dtTrackletsData: 0.008976936340332031 eFPStrackletsData: 111.39656678978366
- dtImshow: 0.004983663558959961 eFPSimshow: 200.65555941202624
- fullLoopTime: 0.04640913009643555 eFPSfullLoopTime: 21.547483789818358

- approx 0.6 seconds latency; better but still not good enough

'''

## mobilenet ssd label texts
# labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

## person-detection-retail-0013 label texts
labelMap = [ "person", "" ]

## yolo v3 tiny label texts
# labelMap = [    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",    "teddy bear",     "hair drier", "toothbrush"]


nnPathDefault = str((Path(__file__).parent / Path('./models/person-detection-retail-0013_openvino_2021.4_6shave.blob')).resolve().absolute())
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
detectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)

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

# manip.initialConfig.setResizeThumbnail(300, 300)    # change size to accomodate nn mobilenet-ssd
manip.initialConfig.setResizeThumbnail(544, 320)  # for nn person-detection-retail-0013
# manip.initialConfig.setResizeThumbnail(416, 416)    # change size to accomodate nn yolo-v3-tiny-tf
# manip.initialConfig.setResize(384, 384)
# manip.initialConfig.setKeepAspectRatio(False) #squash the image to not lose FOV
# The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manip.inputImage.setBlocking(True)

# setting node configs
detectionNetwork.setBlobPath(args.nnPath)
detectionNetwork.setConfidenceThreshold(0.45)
detectionNetwork.input.setBlocking(True)

objectTracker.inputTrackerFrame.setBlocking(True)
objectTracker.inputDetectionFrame.setBlocking(True)
objectTracker.inputDetections.setBlocking(True)
## select for correct model
# objectTracker.setDetectionLabelsToTrack([15])  # track only person - mobilenet-ssd 
objectTracker.setDetectionLabelsToTrack([1])  # track only person - person-detection-retail-0013
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
    

    def capture_window_PIL():
        # UT game in 1278 x 686 windowed mode
        image =  np.array(ImageGrab.grab(bbox=(0,0,1600,900)))
        return image
    
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
        latencyMs = (dai.Clock.now() - manip.getTimestamp()).total_seconds() * 1000
        diffs = np.append(diffs, latencyMs)
        print('Latency Det NN: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}'.format(latencyMs, np.average(diffs), np.std(diffs)))
 
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

            cv2.putText(trackerFrame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(trackerFrame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(trackerFrame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(trackerFrame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(trackerFrame, "Fps: {:.2f}".format(fps), (2, trackerFrame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        dtTrackletsData, previous_time = deltaT(previous_time)
        eFPStrackletsData = 1 / (dtTrackletsData + 0.000000001)

        ## Show Latency in miliseconds 
        latencyMs = (dai.Clock.now() - trackFrame.getTimestamp()).total_seconds() * 1000
        diffs = np.append(diffs, latencyMs)
        print('Latency trackFrame: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}'.format(latencyMs, np.average(diffs), np.std(diffs)))

        ## use with PIL version
        # cv2.imshow("tracker", trackerFrame)

        # if cv2.waitKey(1) == ord('q'):
        #     break
        
        ## use with dxcam version
        if frame.size > 1:
            cv2.imshow("tracker", trackerFrame)

            if cv2.waitKey(1) == ord('q'):
                break

        dtImshow, previous_time = deltaT(previous_time)
        eFPSimshow = 1 / (dtImshow + 0.000000001)

        
        fullLoopTime = time.time() - initTime
        eFPSfullLoopTime = 1 / (fullLoopTime + 0.000000001)


        print("dtCapFrame:", dtCapFrame, "eFPScapFrame:", eFPScapFrame)
        print("dtNNdetections:", dtNNdetections, "eFPSnnDetections:", eFPSnnDetections)
        print("dtTrackletsData:", dtTrackletsData, "eFPStrackletsData:", eFPStrackletsData)
        print("dtImshow:", dtImshow, "eFPSimshow:", eFPSimshow)
        print("fullLoopTime:", fullLoopTime, "eFPSfullLoopTime:", eFPSfullLoopTime)