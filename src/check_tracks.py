import csv

csvPath = "output/tracks.csv"
shortTrackFrames = 25

framesById = {}

with open(csvPath, newline="") as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        playerId = int(row["playerId"])
        frame = int(row["frame"])

        if playerId not in framesById:
            framesById[playerId] = []
        framesById[playerId].append(frame)

print(f"{'ID':>5} {'seen':>6} {'first':>6} {'last':>6} {'missing':>8} {'maxGap':>8}")

shortTracks = 0

for playerId in sorted(framesById):
    frames = framesById[playerId]

    first = frames[0]
    last = frames[-1]
    seen = len(frames)
    missing = (last - first + 1) - seen

    maxGap = 0
    for i in range(1, len(frames)):
        gap = frames[i] - frames[i - 1] - 1
        if gap > maxGap:
            maxGap = gap

    if seen < shortTrackFrames:
        shortTracks += 1

    print(f"{playerId:>5} {seen:>6} {first:>6} {last:>6} {missing:>8} {maxGap:>8}")

print()
print("Total IDs:", len(framesById))
print(f"Short IDs (< {shortTrackFrames} frames):", shortTracks)
print("Many short IDs usually mean ID switches or false detections.")