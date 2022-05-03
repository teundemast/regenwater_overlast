import numpy as np
import math

def getMiddleCoordinates(array: np.ndarray):
    half_x = int(array.shape[1]/2)
    half_y = int(array.shape[0]/2)
    return half_x, half_y


def distance(x1: int, y1: int, x2: int, y2: int):
    return math.ceil(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))


def inCircle(radius, middle_x, middle_y, x: int, y: int):
    distance_to_middle = distance(middle_x, middle_y, x, y)
    distance_radius = radius-distance_to_middle
    if distance_to_middle < radius:
        return distance_radius, True
    return distance_radius, False


def calcDepression(array: np.ndarray):
    score = 0
    middle_x, middle_y = getMiddleCoordinates(array)
    middle_value = array[middle_y][middle_x]
    radius = array.shape[1] - middle_x
    len_x = array.shape[1]
    len_y = array.shape[0]

    for y in range(len_y):
        for x in range(len_x):
            distance_radius, in_circle = inCircle(radius, middle_x, middle_y, x, y)
            if in_circle:
                value = array[y][x]
                score += -(middle_value - value) * distance_radius
    return score

if __name__ == '__main__':
    array = np.array([[3, 4, 3, 4, 3, 4, 5],[3, 4, 3, 4, 3, 4, 5,],[3, 4, 3, 4, 3, 4, 5,],[3, 4, 6, 1, 6, 4, 5,],[3, 4, 3, 4, 3, 4, 5,],[3, 4, 3, 4, 3, 4, 5,],[3, 4, 3, 4, 3, 4, 5,]])
    print(calcDepression(array))