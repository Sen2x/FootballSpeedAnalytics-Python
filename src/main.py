import csv
import json
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from field import applyHomography


videoPath = "videos/match.mp4"
savedVideo = "output/players_detected.mp4"
dataFile = "output/tracks.csv"
calibrationFile = "output/calibration.json"

modelN = "yolo26s.pt"
inputSize = 1280
trackerName = "src/botsort_football.yaml"
minConfidence = 0.05

displayVideo = True
displayWidth = 1100
windowName = "Football Analytics (press q to stop)"

trailLength = 30
gapLimit = 10
fieldMargin = 0.5
fieldPolygon = [(541, 357), (1252, 359), (1610, 1080), (262, 1080)]


def loadCalibration():
    if not os.path.exists(calibrationFile):
        return None
    with open(calibrationFile) as file:
        return json.load(file)


def isOnField(calibration, footX, footY):
    polygon = np.array(fieldPolygon, dtype=np.int32)
    if cv2.pointPolygonTest(polygon, (float(footX), float(footY)), False) < 0:
        return False
    if calibration is None:
        return True
    fieldX, fieldY = applyHomography(calibration["homography"], footX, footY)
    insideX = -fieldMargin <= fieldX <= calibration["fieldLength"] + fieldMargin
    insideY = -fieldMargin <= fieldY <= calibration["fieldWidth"] + fieldMargin
    return insideX and insideY


def getFieldCorners(calibration):
    inverse = np.linalg.inv(calibration["homography"])
    length = calibration["fieldLength"]
    width = calibration["fieldWidth"]
    corners = []
    for x, y in [(0, 0), (length, 0), (length, width), (0, width)]:
        pixelX, pixelY = applyHomography(inverse, x, y)
        corners.append((int(pixelX), int(pixelY)))
    return corners


def getFootPoint(x1, y1, x2, y2):
    footX = (x1 + x2) / 2
    footY = y2
    return footX, footY


def updateTrail(trails, lastSeen, playerId, point, frameNumber):
    if playerId not in trails:
        trails[playerId] = []
    elif frameNumber - lastSeen[playerId] > gapLimit:
        trails[playerId] = []

    trails[playerId].append(point)
    trails[playerId] = trails[playerId][-trailLength:]

    lastSeen[playerId] = frameNumber


def drawPlayer(frame, playerId, x1, y1, x2, y2, trail, fontScale):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 150, 0), 1)

    for i in range(1, len(trail)):
        cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), 1)

    cv2.circle(frame, trail[-1], 3, (0, 0, 255), -1)

    label = str(playerId)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (textW, textH), baseline = cv2.getTextSize(label, font, fontScale, thickness)

    labelX = int(x1)
    labelY = max(textH + 4, int(y1) - 4)

    cv2.rectangle(frame,
                  (labelX, labelY - textH - 4),
                  (labelX + textW + 4, labelY + 2),
                  (180, 60, 0), -1)
    cv2.putText(frame, label, (labelX + 2, labelY - 1),
                font, fontScale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    os.makedirs("output", exist_ok=True)

    if torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"
    print("Device:", device)
    if device == "cpu":
        print("WARNING: running on CPU, this will be slow. Check that PyTorch with CUDA is installed.")

    model = YOLO(modelN)

    calibration = loadCalibration()
    if calibration is None:
        print("No calibration: using fieldPolygon (pixels) to ignore spectators.")
        fieldCorners = fieldPolygon
    else:
        print("Field filter ON: people outside the field are ignored.")
        fieldCorners = getFieldCorners(calibration)
        if calibration.get("note"):
            print("NOTE:", calibration["note"])

    video = cv2.VideoCapture(videoPath)
    if not video.isOpened():
        print("Error: could not open", videoPath)
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if not (0 < fps < 1000):
        print("WARNING: video FPS is unknown, using 25. Times in the CSV may be wrong!")
        fps = 25

    print(f"Video: {width}x{height}, {fps:.2f} FPS, {totalFrames} frames")

    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(savedVideo, codec, fps, (width, height))
    if not writer.isOpened():
        print("Error: could not create", savedVideo)
        video.release()
        return

    fontScale = height / 2000

    trails = {}
    lastSeen = {}
    seenIds = set()
    rowsSaved = 0
    outsideCount = 0
    frameNumber = 0
    startTime = time.perf_counter()

    csvFile = open(dataFile, "w", newline="")
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(["frame", "timeSec", "playerId",
                        "x1", "y1", "x2", "y2",
                        "footX", "footY", "confidence"])

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break

            results = model.track(frame,
                                  persist=True,
                                  tracker=trackerName,
                                  classes=[0],
                                  conf=minConfidence,
                                  imgsz=inputSize,
                                  device=device,
                                  verbose=False)
            boxes = results[0].boxes

            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().tolist()
                coordinates = boxes.xyxy.tolist()
                confidences = boxes.conf.tolist()

                for playerId, box, confidence in zip(ids, coordinates, confidences):
                    x1, y1, x2, y2 = box
                    footX, footY = getFootPoint(x1, y1, x2, y2)

                    if not isOnField(calibration, footX, footY):
                        outsideCount += 1
                        continue

                    timeSec = frameNumber / fps

                    csvWriter.writerow([frameNumber, round(timeSec, 3), playerId,
                                        round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                                        round(footX, 1), round(footY, 1),
                                        round(confidence, 3)])
                    rowsSaved += 1
                    seenIds.add(playerId)

                    updateTrail(trails, lastSeen, playerId,
                                (int(footX), int(footY)), frameNumber)
                    drawPlayer(frame, playerId, x1, y1, x2, y2,
                               trails[playerId], fontScale)

            if fieldCorners is not None:
                for i in range(4):
                    cv2.line(frame, fieldCorners[i], fieldCorners[(i + 1) % 4], (255, 0, 255), 1)

            cv2.putText(frame, f"Frame {frameNumber}  |  {frameNumber / fps:.2f} s",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            writer.write(frame)

            if displayVideo:
                previewHeight = int(height * displayWidth / width)
                preview = cv2.resize(frame, (displayWidth, previewHeight))
                cv2.imshow(windowName, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("Stopped by user")
                    break
                if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed, stopping")
                    break

            frameNumber += 1

            if frameNumber % 100 == 0:
                elapsed = time.perf_counter() - startTime
                print(f"Frame {frameNumber}/{totalFrames}  |  "
                      f"processing speed: {frameNumber / elapsed:.1f} frames/s")

    except KeyboardInterrupt:
        print("Stopped with Ctrl+C")

    csvFile.close()
    writer.release()
    video.release()
    cv2.destroyAllWindows()

    elapsed = time.perf_counter() - startTime
    print()
    print("Done!")
    print(f"Frames processed: {frameNumber}")
    print(f"Video length:     {frameNumber / fps:.1f} s")
    print(f"Processing time:  {elapsed:.1f} s  ({frameNumber / max(elapsed, 0.001):.1f} frames/s)")
    print(f"Different IDs:    {len(seenIds)}")
    print(f"Rows in CSV:      {rowsSaved}")
    print(f"Ignored outside:  {outsideCount}")
    print("Video saved:", savedVideo)
    print("Data saved: ", dataFile)


if __name__ == "__main__":
    main()
