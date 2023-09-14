import numpy as np

# Collection of random distributions not implemented in numpy


def bimodal(*, loc=[0, 1], scale=[0.5, 0.5]):
    peak1 = np.random.normal(loc=loc[0], scale=scale[0])
    peak2 = np.random.normal(loc=loc[1], scale=scale[1])
    # print(f"{peak1}")
    # print(f"{peak2}")
    return np.random.choice([peak1, peak2])
