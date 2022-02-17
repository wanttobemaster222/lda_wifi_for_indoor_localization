import numpy as np
import os

data_path = 'C:/Users/yjh54/Desktop/Coex/coex_dataset/'
test_path = 'C:/Users/yjh54/Desktop/Coex/coex_test_data/0_1636001185913/wifi.txt'

class data_load:
    def __init__(self, wifi, num, tst_wifi):
        self.wifi = wifi
        self.num = num
        self.tst_wifi = tst_wifi
        
    def load_wifi(self):
        file_list = sorted(os.listdir(data_path), key=int)
        for i in range(0, len(file_list)):
            with open(data_path + file_list[i] + '/wifi.txt', 'r', encoding='UTF-8') as f1_0:
                wf = np.genfromtxt(f1_0, str)
                self.num.append(len(wf))
                self.wifi = np.vstack((self.wifi, wf))    
        with open(test_path, 'r', encoding='UTF-8') as f1_0:
            self.tst_wifi = np.genfromtxt(f1_0, str)
            
        return self.wifi, self.num, self.tst_wifi