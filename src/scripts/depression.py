# Not used
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

def scalar(array: np.ndarray, scale: float):
    
    len_x = array.shape[0]
    if len_x % 2 == 0:
        array = array[:-1]
        for list in array:
            list = list[:-1]
    len_x = array.shape[0]
    scaled_len = round(len_x * scale)
    if scaled_len % 2 == 0:
        scaled_len -= 1
    if scaled_len <= 0:
        return None

    result = np.zeros((scaled_len, scaled_len))
    decrease = round((len_x - scaled_len)/2)
    count = 0
    for index, value in enumerate(array):
        if index < decrease or index >= len_x-decrease:
            continue
        try:
            result[count] = value[decrease:len_x-decrease]
        except ValueError:
            try:
                result[count] = value[decrease + 1:len_x-decrease]
            except ValueError:
                result[count] = value[decrease - 1:len_x-decrease]
        count += 1
    return result


if __name__ == '__main__':
    array = np.array([[3, 4, 3, 4, 3, 4, 5],[3, 4, 3, 4, 3, 4, 5,],[3, 4, 3, 4, 3, 4, 5,],[3, 4, 6, 1, 6, 4, 5,],[3, 4, 3, 4, 3, 4, 5,],[3, 4, 3, 4, 3, 4, 5,],[3, 4, 3, 4, 3, 4, 5,]])
    print(calcDepression(array))
