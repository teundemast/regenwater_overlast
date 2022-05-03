import numpy as np


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




if __name__ == '__main__':
    array = np.array([[3, 4, 3, 4, 3, 4, 5, 7], [3, 4, 3, 4, 3, 4, 5, 3], [3, 4, 3, 4, 3, 4, 5, 3], [3, 4, 6, 1, 6, 4, 5, 5], [3, 4, 3, 4, 3, 4, 5, 5], [3, 4, 3, 4, 3, 4, 5, 4], [3, 4, 3, 4, 3, 4, 5, 1], [3, 4, 3, 4, 3, 4, 5, 5]])
    for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
        
        scalar(array, i)
