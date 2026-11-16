import json
import os

import cv2
import numpy as np

from field import applyHomography, fieldLines


videoPath = "videos/match.mp4"
calibrationFile = "output/calibration.json"
checkImageFile = "output/calibration_check.png"

frameToUse = 0
fieldLength = 105
fieldWidth = 68
displayWidth = 1600
windowName = "Calibration"

clickedPoints = []
displayScale = 1.0


def onMouseClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        realX = x / displayScale
        realY = y / displayScale
        clickedPoints.append((realX, realY))
        print(f"Point {len(clickedPoints)}: pixel ({realX:.0f}, {realY:.0f})")


def drawClickedPoints(frame):
    picture = frame.copy()
    number = 1
    for x, y in clickedPoints:
        cv2.circle(picture, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(picture, str(number), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        number += 1
    return picture


def askFieldPoint(number):
    while True:
        text = input(f"Point {number}: field X Y in meters (example: 16.5 13.84): ")
        parts = text.split()
        if len(parts) != 2:
            print("Type two numbers separated by a space. Use a dot: 16.5")
            continue
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            print("Not a number. Use a dot, not a comma: 16.5")


def drawFieldOnFrame(frame, matrix):
    inverse = np.linalg.inv(matrix)
    picture = frame.copy()
    for start, end in fieldLines(fieldLength, fieldWidth, None):
        x1, y1 = applyHomography(inverse, start[0], start[1])
        x2, y2 = applyHomography(inverse, end[0], end[1])
        cv2.line(picture, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
    return picture


def main():
    global displayScale

    video = cv2.VideoCapture(videoPath)
    video.set(cv2.CAP_PROP_POS_FRAMES, frameToUse)
    ok, frame = video.read()
    video.release()
    if not ok:
        print("Error: could not read frame", frameToUse, "from", videoPath)
        return

    height, width = frame.shape[:2]
    displayScale = min(1.0, displayWidth / width)

    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(windowName, onMouseClick)

    print("Click field points (at least 4, better 6-8, spread over the field).")
    print("u = undo last point, Enter = done, Esc = cancel")

    while True:
        picture = drawClickedPoints(frame)
        small = cv2.resize(picture, (int(width * displayScale), int(height * displayScale)))
        cv2.imshow(windowName, small)
        key = cv2.waitKey(20) & 0xFF
        if key == 13:
            break
        if key == 27:
            print("Cancelled")
            cv2.destroyAllWindows()
            return
        if key == ord("u") and clickedPoints:
            clickedPoints.pop()
            print("Last point removed")

    cv2.destroyAllWindows()

    if len(clickedPoints) < 4:
        print("Error: need at least 4 points, you clicked", len(clickedPoints))
        return

    fieldPoints = []
    for number in range(1, len(clickedPoints) + 1):
        fieldPoints.append(askFieldPoint(number))

    imageArray = np.array(clickedPoints, dtype=np.float32)
    fieldArray = np.array(fieldPoints, dtype=np.float32)
    matrix, mask = cv2.findHomography(imageArray, fieldArray)

    if matrix is None:
        print("Error: calibration failed. Check that the points are not on one line.")
        return

    print()
    errors = []
    for number in range(len(clickedPoints)):
        imageX, imageY = clickedPoints[number]
        realX, realY = fieldPoints[number]
        guessX, guessY = applyHomography(matrix, imageX, imageY)
        error = ((guessX - realX) ** 2 + (guessY - realY) ** 2) ** 0.5
        errors.append(error)
        print(f"Point {number + 1}: real ({realX}, {realY})  "
              f"calculated ({guessX:.2f}, {guessY:.2f})  error {error:.2f} m")

    meanError = sum(errors) / len(errors)
    if len(clickedPoints) == 4:
        print("Only 4 points: error is always 0, so it cannot be checked. Use 6+ points.")
        meanErrorToSave = None
    else:
        print(f"Mean calibration error: {meanError:.2f} m")
        meanErrorToSave = round(meanError, 3)

    os.makedirs("output", exist_ok=True)
    calibration = {
        "videoPath": videoPath,
        "frame": frameToUse,
        "fieldLength": fieldLength,
        "fieldWidth": fieldWidth,
        "imagePoints": clickedPoints,
        "fieldPoints": fieldPoints,
        "homography": matrix.tolist(),
        "meanErrorMeters": meanErrorToSave
    }
    with open(calibrationFile, "w") as file:
        json.dump(calibration, file, indent=2)

    cv2.imwrite(checkImageFile, drawFieldOnFrame(frame, matrix))

    print("Calibration saved:", calibrationFile)
    print("Open", checkImageFile, "- pink lines must lie on the real field lines.")


if __name__ == "__main__":
    main()
