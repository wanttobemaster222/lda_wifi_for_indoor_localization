import statistics
import numpy as np

class Data_mean:
    def __init__(self, train_XX, wifi_Y):
        self.train_XX = train_XX
        self.wifi_Y = wifi_Y
        
    def data_mean(self):
        e_class = sorted(np.unique(self.wifi_Y))
        L_mean = np.zeros((len(self.train_XX[0, :])))
        L_Mean = np.zeros((1, len(self.train_XX[0, :])))
        for i in range(0, len(e_class)):
            e_index = np.where(e_class[i] == self.wifi_Y)
            for j in range(0, len(self.train_XX[0, :])):
                L_mean[j] = statistics.mean(self.train_XX[(e_index[0])[0]:(e_index[0])[-1]+1, j])
            L_Mean = np.vstack((L_Mean, L_mean))
        return L_Mean[1:,]
        