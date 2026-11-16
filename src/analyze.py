import csv
import json
import os

import cv2
import numpy as np

from field import applyHomography, fieldLines


tracksFile = "output/tracks.csv"
calibrationFile = "output/calibration.json"
playersFolder = "output/players"
statsFile = "output/stats.csv"

maxGapFrames = 10
blockSeconds = 0.4
speedBlockSeconds = 1.0
maxRealSpeed = 11.0
minTrackSeconds = 5.0
fieldMargin = 0.5

pixelsPerMeter = 10
imageMargin = 30
legendHeight = 100


def loadCalibration():
    if not os.path.exists(calibrationFile):
        return None
    with open(calibrationFile) as file:
        return json.load(file)


def loadTracks(matrix, fieldLength, fieldWidth):
    tracks = {}
    fps = None
    outsideCount = 0

    with open(tracksFile, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame = int(row["frame"])
            timeSec = float(row["timeSec"])
            playerId = int(row["playerId"])
            footX = float(row["footX"])
            footY = float(row["footY"])

            if timeSec > 0:
                fps = frame / timeSec

            fieldX, fieldY = applyHomography(matrix, footX, footY)

            insideX = -fieldMargin <= fieldX <= fieldLength + fieldMargin
            insideY = -fieldMargin <= fieldY <= fieldWidth + fieldMargin
            if not (insideX and insideY):
                outsideCount += 1
                continue

            if playerId not in tracks:
                tracks[playerId] = []
            tracks[playerId].append({"frame": frame, "time": timeSec,
                                     "x": fieldX, "y": fieldY})

    return tracks, fps, outsideCount


def splitIntoSegments(points):
    segments = []
    current = []
    for point in points:
        if current and point["frame"] - current[-1]["frame"] > maxGapFrames:
            segments.append(current)
            current = []
        current.append(point)
    if current:
        segments.append(current)
    return segments


def middleValue(values):
    values = sorted(values)
    return values[len(values) // 2]


def averageBlocks(segment, blockFrames):
    blocks = []
    for start in range(0, len(segment), blockFrames):
        part = segment[start:start + blockFrames]
        listX = []
        listY = []
        listTime = []
        for point in part:
            listX.append(point["x"])
            listY.append(point["y"])
            listTime.append(point["time"])
        blocks.append((middleValue(listTime), middleValue(listX), middleValue(listY)))
    return blocks


def calculateDistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    return (dx ** 2 + dy ** 2) ** 0.5


def analyzePlayer(points, blockFrames, speedBlockFrames):
    segments = splitIntoSegments(points)

    distance = 0.0
    movingTime = 0.0
    trackedTime = 0.0
    maxSpeed = 0.0
    rejectedSteps = 0
    allBlocks = []

    for segment in segments:
        trackedTime += segment[-1]["time"] - segment[0]["time"]
        blocks = averageBlocks(segment, blockFrames)
        allBlocks.append(blocks)

        for i in range(1, len(blocks)):
            time1, x1, y1 = blocks[i - 1]
            time2, x2, y2 = blocks[i]
            stepTime = time2 - time1
            if stepTime <= 0:
                continue

            step = calculateDistance((x1, y1), (x2, y2))
            speed = step / stepTime

            if speed > maxRealSpeed:
                rejectedSteps += 1
                continue

            distance += step
            movingTime += stepTime

        speedBlocks = averageBlocks(segment, speedBlockFrames)
        for i in range(1, len(speedBlocks)):
            time1, x1, y1 = speedBlocks[i - 1]
            time2, x2, y2 = speedBlocks[i]
            stepTime = time2 - time1
            if stepTime <= 0:
                continue
            speed = calculateDistance((x1, y1), (x2, y2)) / stepTime
            if speed <= maxRealSpeed and speed > maxSpeed:
                maxSpeed = speed

    firstFrame = points[0]["frame"]
    lastFrame = points[-1]["frame"]
    coverage = len(points) / (lastFrame - firstFrame + 1)

    return {
        "distance": distance,
        "movingTime": movingTime,
        "trackedTime": trackedTime,
        "maxSpeed": maxSpeed,
        "rejectedSteps": rejectedSteps,
        "segments": len(segments),
        "coverage": coverage,
        "blocks": allBlocks
    }


def toPixel(x, y):
    return (int(imageMargin + y * pixelsPerMeter), int(imageMargin + x * pixelsPerMeter))


def drawPitch(fieldLength, fieldWidth, lines):
    imageWidth = int(fieldWidth * pixelsPerMeter) + 2 * imageMargin
    imageHeight = int(fieldLength * pixelsPerMeter) + 2 * imageMargin + legendHeight
    picture = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)
    picture[:] = (60, 130, 60)
    drawPitchLines(picture, fieldLength, fieldWidth, lines)
    return picture


def drawPitchLines(picture, fieldLength, fieldWidth, lines):
    for start, end in fieldLines(fieldLength, fieldWidth, lines):
        cv2.line(picture, toPixel(start[0], start[1]), toPixel(end[0], end[1]),
                 (255, 255, 255), 2)


def writeText(picture, text, y):
    cv2.putText(picture, text, (imageMargin, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)


def makeHeatmap(points, playerId, fieldLength, fieldWidth, lines, label):
    picture = drawPitch(fieldLength, fieldWidth, lines)

    cellsX = int(fieldLength) + 1
    cellsY = int(fieldWidth) + 1
    grid = np.zeros((cellsX, cellsY), dtype=np.float32)

    for point in points:
        cellX = min(max(int(point["x"]), 0), cellsX - 1)
        cellY = min(max(int(point["y"]), 0), cellsY - 1)
        grid[cellX, cellY] += 1

    grid = cv2.GaussianBlur(grid, (0, 0), 2)
    grid = grid / grid.max() * 255
    gray = grid.astype(np.uint8)

    pitchWidth = int(fieldWidth * pixelsPerMeter)
    pitchHeight = int(fieldLength * pixelsPerMeter)
    gray = cv2.resize(gray, (pitchWidth, pitchHeight), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    top = imageMargin
    left = imageMargin
    area = picture[top:top + pitchHeight, left:left + pitchWidth]
    mixed = cv2.addWeighted(area, 0.35, colored, 0.65, 0)
    mask = gray > 15
    area[mask] = mixed[mask]

    drawPitchLines(picture, fieldLength, fieldWidth, lines)

    legendTop = imageMargin * 2 + pitchHeight
    gradient = np.arange(256, dtype=np.uint8).reshape(1, 256)
    gradient = cv2.resize(gradient, (300, 15), interpolation=cv2.INTER_LINEAR)
    bar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    picture[legendTop:legendTop + 15, left:left + 300] = bar
    cv2.putText(picture, "less time", (left, legendTop + 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(picture, "more time", (left + 220, legendTop + 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    writeText(picture, f"Track ID {playerId} - heatmap, {label}", legendTop + 70)
    return picture


def makeTrajectory(blocksList, playerId, fieldLength, fieldWidth, lines, label):
    picture = drawPitch(fieldLength, fieldWidth, lines)

    for blocks in blocksList:
        for i in range(1, len(blocks)):
            point1 = toPixel(blocks[i - 1][1], blocks[i - 1][2])
            point2 = toPixel(blocks[i][1], blocks[i][2])
            cv2.line(picture, point1, point2, (0, 255, 255), 2)

    firstBlock = blocksList[0][0]
    lastBlock = blocksList[-1][-1]
    cv2.circle(picture, toPixel(firstBlock[1], firstBlock[2]), 7, (0, 255, 0), -1)
    cv2.circle(picture, toPixel(lastBlock[1], lastBlock[2]), 7, (0, 0, 255), -1)

    legendTop = imageMargin * 2 + int(fieldLength * pixelsPerMeter)
    writeText(picture, f"Track ID {playerId} - trajectory, {label}", legendTop + 20)
    writeText(picture, "Green = start, red = end, breaks = tracking lost", legendTop + 45)
    return picture


def getDistance(row):
    return row["distanceMeters"]


def formatTime(seconds):
    minutes = int(seconds // 60)
    rest = int(seconds % 60)
    return f"{minutes:02d}:{rest:02d}"


def main():
    calibration = loadCalibration()
    if calibration is None:
        print("Distance in meters is NOT available: no calibration file.")
        print("Run: python src/calibrate.py")
        return

    if not os.path.exists(tracksFile):
        print("No", tracksFile, "- run python src/main.py first")
        return

    matrix = calibration["homography"]
    fieldLength = calibration["fieldLength"]
    fieldWidth = calibration["fieldWidth"]
    calibrationError = calibration["meanErrorMeters"]
    lines = calibration.get("lines")
    calibrationNote = calibration.get("note")
    if calibrationNote is None:
        label = "field meters, fixed camera calibration"
    else:
        label = "ESTIMATED meters (+-15%)"
        print("NOTE:", calibrationNote)

    tracks, fps, outsideCount = loadTracks(matrix, fieldLength, fieldWidth)
    if fps is None:
        print("Error: could not find FPS in", tracksFile)
        return

    blockFrames = max(1, round(fps * blockSeconds))
    speedBlockFrames = max(1, round(fps * speedBlockSeconds))
    print(f"FPS: {fps:.2f}, averaging blocks of {blockFrames} frames ({blockSeconds} s)")
    print(f"Detections outside the field (ignored): {outsideCount}")
    if calibrationError is None:
        print("Calibration error: unknown (only 4 points)")
    else:
        print(f"Calibration error: {calibrationError} m")

    os.makedirs(playersFolder, exist_ok=True)
    rows = []

    for playerId in sorted(tracks):
        points = tracks[playerId]
        result = analyzePlayer(points, blockFrames, speedBlockFrames)

        if result["trackedTime"] < minTrackSeconds:
            note = f"tracked less than {minTrackSeconds:.0f} s - speed N/A, no images"
            averageSpeed = "N/A"
            maxSpeed = "N/A"
        else:
            note = ""
            if result["movingTime"] > 0:
                averageSpeed = round(result["distance"] / result["movingTime"] * 3.6, 1)
            else:
                averageSpeed = "N/A"
            maxSpeed = round(result["maxSpeed"] * 3.6, 1)

            if result["rejectedSteps"] > 0:
                note = f"{result['rejectedSteps']} impossible jumps not counted"

            folder = os.path.join(playersFolder, f"player_{playerId}")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, "heatmap.png"),
                        makeHeatmap(points, playerId, fieldLength, fieldWidth, lines, label))
            cv2.imwrite(os.path.join(folder, "trajectory.png"),
                        makeTrajectory(result["blocks"], playerId, fieldLength, fieldWidth, lines, label))

        rows.append({
            "playerId": playerId,
            "trackedTime": formatTime(result["trackedTime"]),
            "distanceMeters": round(result["distance"]),
            "averageSpeedKmh": averageSpeed,
            "maxSpeedKmh": maxSpeed,
            "coveragePercent": round(result["coverage"] * 100),
            "segments": result["segments"],
            "note": note
        })

    with open(statsFile, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rows.sort(key=getDistance, reverse=True)

    print()
    print(f"{'ID':>5} {'time':>6} {'distance':>9} {'avg km/h':>9} {'max km/h':>9} {'cover':>6}  note")
    for row in rows:
        print(f"{row['playerId']:>5} {row['trackedTime']:>6} {row['distanceMeters']:>7} m "
              f"{str(row['averageSpeedKmh']):>9} {str(row['maxSpeedKmh']):>9} "
              f"{row['coveragePercent']:>5}%  {row['note']}")

    print()
    print("Stats saved:", statsFile)
    print("Images saved in:", playersFolder)
    print("Meters are correct ONLY if the camera did not move during the video.")


if __name__ == "__main__":
    main()
