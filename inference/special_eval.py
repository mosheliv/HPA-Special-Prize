"""Handle special prize evaluation."""
import time

import numpy as np

from submission_predict import Model


def evaluate():
    """Evaluate special prize submissions."""
    test_data = load_test_data('dummy')
    result = {}
    model = Model()
    for test_id, (image_path, answer) in test_data.items():
        pixel_data = read_fov_images(image_path)
        start = time.time()
        prediction = model.predict(pixel_data)
        end = time.time()
        prediction_time = end - start
        result[test_id] = (answer, prediction, prediction_time)

    print(result)

    return result


def load_test_data(path):
    """Mock load dummy test data from csv file."""
    return {'1': ('image_path_1', {'0'}), '2': ('image_path_2', {'25', '0'})}


def read_fov_images(path):
    """Mock read all channels of a field of view and return numpy array."""
    red = np.random.rand(2048, 2048, 1)
    green = np.random.rand(2048, 2048, 1)
    blue = np.random.rand(2048, 2048, 1)
    yellow = np.random.rand(2048, 2048, 1)
    stack = np.stack([red, green, blue, yellow], axis=-1)
    return stack


if __name__ == '__main__':
    evaluate()

