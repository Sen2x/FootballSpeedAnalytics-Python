import math


def applyHomography(matrix, x, y):
    a = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
    b = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
    w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    return a / w, b / w


def rectangleLines(x1, y1, x2, y2):
    return [((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1))]


def circleLines(centerX, centerY, radius, parts=36):
    lines = []
    for i in range(parts):
        angle1 = 2 * math.pi * i / parts
        angle2 = 2 * math.pi * (i + 1) / parts
        point1 = (centerX + radius * math.cos(angle1), centerY + radius * math.sin(angle1))
        point2 = (centerX + radius * math.cos(angle2), centerY + radius * math.sin(angle2))
        lines.append((point1, point2))
    return lines


def fieldLines(length, width, customLines=None):
    if customLines is not None:
        return customLines
    middleY = width / 2
    lines = []
    lines += rectangleLines(0, 0, length, width)
    lines.append(((length / 2, 0), (length / 2, width)))
    lines += circleLines(length / 2, middleY, 9.15)
    lines += rectangleLines(0, middleY - 20.16, 16.5, middleY + 20.16)
    lines += rectangleLines(length - 16.5, middleY - 20.16, length, middleY + 20.16)
    lines += rectangleLines(0, middleY - 9.16, 5.5, middleY + 9.16)
    lines += rectangleLines(length - 5.5, middleY - 9.16, length, middleY + 9.16)
    return lines
