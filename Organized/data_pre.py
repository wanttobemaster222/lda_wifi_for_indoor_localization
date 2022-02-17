import numpy as np

class Data_pre:
    def __init__(self, wifi, num, tst_wifi):
        self.wifi = wifi
        self.num = num
        self.tst_wifi = tst_wifi
        
    def replace(self):
        uni_bssid = np.unique(self.wifi[1:, 1])
        LLand = self.wifi[1:, :]
        re_X = np.ones((1, len(uni_bssid)))
        re_Y = np.zeros((1, 1))
        q = 0
        for i in range(0, len(self.num)):
            Land = LLand[q:q+self.num[i], :]
            tm = list(set(Land[:, 0]))
            place = (-100)*(np.ones((len(tm), uni_bssid.shape[0])))
            Label = (i+1)*(np.ones((len(tm), 1)))
            for j in range(0, len(tm)):
                p = np.where(tm[j] == Land[:, 0])
                for k in range(0, len(p[0])):
                    pla = np.where(uni_bssid == Land[(p[0])[k], 1])
                    place[j, (pla[0])[0]] = Land[(p[0])[k], 2]
            re_X = np.vstack((re_X, place))
            re_Y = np.vstack((re_Y, Label))
            q += self.num[i]
        
        Landmark = self.tst_wifi[1:, :]
        tst_k = list(sorted(set(Landmark[:, 0])))
        tst_Replace = (-100)*(np.ones((len(tst_k), uni_bssid.shape[0])))
        for i in range(0, len(tst_k)):
            tst_p = np.where(tst_k[i] == Landmark[:, 0])
            for j in range(0, len(tst_p[0])):
                place = np.where(uni_bssid == Landmark[(tst_p[0])[j], 1])
                if len(place[0]):
                    tst_Replace[i, (place[0])[0]] = Landmark[(tst_p[0])[j], 2]        
        return re_X[1:, :], re_Y[1:, :], tst_Replace