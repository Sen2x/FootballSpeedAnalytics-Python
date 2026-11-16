import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analyze import averageBlocks, calculateDistance, middleValue, splitIntoSegments
from field import applyHomography, fieldLines


def makePoints(frames):
    points = []
    for frame in frames:
        points.append({"frame": frame, "time": frame / 25, "x": float(frame), "y": 0.0})
    return points


class CalculateDistanceTest(unittest.TestCase):

    def testThreeFourFive(self):
        self.assertEqual(calculateDistance((0, 0), (3, 4)), 5)

    def testSamePoint(self):
        self.assertEqual(calculateDistance((2, 7), (2, 7)), 0)


class SplitIntoSegmentsTest(unittest.TestCase):

    def testNoGapGivesOneSegment(self):
        segments = splitIntoSegments(makePoints([0, 1, 2, 3, 4]))
        self.assertEqual(len(segments), 1)

    def testSmallGapIsKept(self):
        segments = splitIntoSegments(makePoints([0, 1, 5, 6]))
        self.assertEqual(len(segments), 1)

    def testBigGapSplitsTrack(self):
        segments = splitIntoSegments(makePoints([0, 1, 2, 50, 51]))
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1][0]["frame"], 50)


class AverageBlocksTest(unittest.TestCase):

    def testMedianIgnoresOneWrongPoint(self):
        points = makePoints([0, 1, 2, 3, 4])
        points[2]["x"] = 1000.0
        blocks = averageBlocks(points, 5)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][1], 3.0)

    def testMiddleValue(self):
        self.assertEqual(middleValue([5, 1, 3]), 3)


class FieldTest(unittest.TestCase):

    def testIdentityHomography(self):
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertEqual(applyHomography(identity, 12.5, 30), (12.5, 30))

    def testCustomLinesAreUsed(self):
        customLines = [((0, 0), (10, 0))]
        self.assertEqual(fieldLines(105, 68, customLines), customLines)


if __name__ == "__main__":
    unittest.main()
