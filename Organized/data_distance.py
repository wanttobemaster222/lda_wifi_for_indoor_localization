import numpy as np
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d

class Dis:
    def __init__(self, class_mean, tst_XX):
        self.class_mean = class_mean
        self.tst_XX = tst_XX
    def D_dis(self):
        eu_dis = np.zeros((len(self.class_mean[:, 0]), len(self.tst_XX[:, 0])))
        for i in range(0, len(self.class_mean[0, :])+1):
            for j in range(0, len(self.tst_XX[:, 0])):
                eu_dis[i, j] = distance.euclidean(self.class_mean[i, :], self.tst_XX[j, :])
        # Smoothing
        eu_dis = gaussian_filter1d(eu_dis, sigma=2)
        return eu_dis
        