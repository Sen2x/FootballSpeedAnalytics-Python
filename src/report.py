import csv
import json
import os

import cv2
import numpy as np


videoPath = "videos/match.mp4"
tracksFile = "output/tracks.csv"
statsFile = "output/stats.csv"
calibrationFile = "output/calibration.json"
playersFolder = "output/players"
reportFile = "output/report.html"

edgeMargin = 20
cropPadding = 0.5
photoHeight = 300
colorEveryFrames = 10

shirtColors = [
    {"name": "Red team", "hueFrom": 0, "hueTo": 20},
    {"name": "Red team", "hueFrom": 165, "hueTo": 180},
    {"name": "Light blue team", "hueFrom": 80, "hueTo": 104},
    {"name": "Referee", "hueFrom": 104, "hueTo": 130},
]


def loadStats():
    stats = {}
    with open(statsFile, newline="") as file:
        for row in csv.DictReader(file):
            stats[int(row["playerId"])] = row
    return stats


def findBestFrames(playerIds, frameWidth, frameHeight):
    best = {}
    with open(tracksFile, newline="") as file:
        for row in csv.DictReader(file):
            playerId = int(row["playerId"])
            if playerId not in playerIds:
                continue

            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])

            nearEdge = (x1 < edgeMargin or y1 < edgeMargin or
                        x2 > frameWidth - edgeMargin or y2 > frameHeight - edgeMargin)
            if nearEdge:
                continue

            score = float(row["confidence"]) * (y2 - y1)
            if playerId not in best or score > best[playerId]["score"]:
                best[playerId] = {"score": score, "frame": int(row["frame"]),
                                  "box": (x1, y1, x2, y2)}
    return best


def findShirtHues(playerIds):
    rowsByFrame = {}
    with open(tracksFile, newline="") as file:
        for row in csv.DictReader(file):
            frameNumber = int(row["frame"])
            if int(row["playerId"]) in playerIds and frameNumber % colorEveryFrames == 0:
                if frameNumber not in rowsByFrame:
                    rowsByFrame[frameNumber] = []
                rowsByFrame[frameNumber].append(row)

    huesById = {}
    video = cv2.VideoCapture(videoPath)
    for frameNumber in sorted(rowsByFrame):
        video.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
        ok, frame = video.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for row in rowsByFrame[frameNumber]:
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            boxWidth = x2 - x1
            boxHeight = y2 - y1
            shirt = hsv[int(y1 + 0.2 * boxHeight):int(y1 + 0.5 * boxHeight),
                        int(x1 + 0.25 * boxWidth):int(x2 - 0.25 * boxWidth)]
            if shirt.size == 0:
                continue
            playerId = int(row["playerId"])
            if playerId not in huesById:
                huesById[playerId] = []
            huesById[playerId].append(np.median(shirt[:, :, 0]))
    video.release()

    shirtHues = {}
    for playerId in huesById:
        shirtHues[playerId] = float(np.median(huesById[playerId]))
    return shirtHues


def getTeam(hue):
    if hue is None:
        return "Unknown"
    for color in shirtColors:
        if color["hueFrom"] <= hue < color["hueTo"]:
            return color["name"]
    return "Unknown"


def cropPlayer(frame, box):
    x1, y1, x2, y2 = box
    boxWidth = x2 - x1
    boxHeight = y2 - y1
    left = max(0, int(x1 - boxWidth * cropPadding))
    right = min(frame.shape[1], int(x2 + boxWidth * cropPadding))
    top = max(0, int(y1 - boxHeight * cropPadding))
    bottom = min(frame.shape[0], int(y2 + boxHeight * cropPadding))
    crop = frame[top:bottom, left:right]
    scale = photoHeight / crop.shape[0]
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def drawPlayerOnFrame(frame, playerId, box):
    picture = frame.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(picture, (int(x1) - 4, int(y1) - 4), (int(x2) + 4, int(y2) + 4), (0, 0, 255), 3)
    cv2.putText(picture, f"ID {playerId}", (int(x1), int(y1) - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return cv2.resize(picture, (picture.shape[1] // 2, picture.shape[0] // 2))


def savePlayerImages(best):
    video = cv2.VideoCapture(videoPath)
    for playerId in best:
        video.set(cv2.CAP_PROP_POS_FRAMES, best[playerId]["frame"])
        ok, frame = video.read()
        if not ok:
            continue
        folder = os.path.join(playersFolder, f"player_{playerId}")
        os.makedirs(folder, exist_ok=True)
        box = best[playerId]["box"]
        cv2.imwrite(os.path.join(folder, "player.jpg"), cropPlayer(frame, box))
        cv2.imwrite(os.path.join(folder, "frame.jpg"), drawPlayerOnFrame(frame, playerId, box))
    video.release()


def makeHtml(stats, playerIds, teams, note):
    players = []
    for playerId in sorted(playerIds):
        row = stats[playerId]
        folder = f"players/player_{playerId}/"
        players.append({
            "id": playerId,
            "team": teams[playerId],
            "time": row["trackedTime"],
            "distance": row["distanceMeters"],
            "avgSpeed": row["averageSpeedKmh"],
            "maxSpeed": row["maxSpeedKmh"],
            "coverage": row["coveragePercent"],
            "note": row["note"],
            "photo": folder + "player.jpg",
            "frame": folder + "frame.jpg",
            "heatmap": folder + "heatmap.png",
            "trajectory": folder + "trajectory.png"
        })

    template = open(os.path.join(os.path.dirname(__file__), "report_template.html"),
                    encoding="utf-8").read()
    page = template.replace("PLAYERS_DATA", json.dumps(players))
    page = page.replace("CALIBRATION_NOTE", note)
    with open(reportFile, "w", encoding="utf-8") as file:
        file.write(page)


def main():
    stats = loadStats()

    playerIds = set()
    for playerId in stats:
        if os.path.exists(os.path.join(playersFolder, f"player_{playerId}", "heatmap.png")):
            playerIds.add(playerId)

    if not playerIds:
        print("No players with images. Run analyze.py first.")
        return

    video = cv2.VideoCapture(videoPath)
    frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    best = findBestFrames(playerIds, frameWidth, frameHeight)
    savePlayerImages(best)

    note = "Calibrated field meters."
    if os.path.exists(calibrationFile):
        with open(calibrationFile) as file:
            calibration = json.load(file)
        if calibration.get("note"):
            note = calibration["note"]

    shirtHues = findShirtHues(playerIds)
    teams = {}
    for playerId in sorted(playerIds):
        teams[playerId] = getTeam(shirtHues.get(playerId))
        print(f"Track ID {playerId}: {teams[playerId]}")

    makeHtml(stats, playerIds, teams, note)
    print("Players in report:", len(playerIds))
    print("Report saved:", reportFile, "- open it in a browser")


if __name__ == "__main__":
    main()
