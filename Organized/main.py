import data_load
import data_pre
import data_mean
import data_distance
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kfda import Kfda
import matplotlib.pyplot as plt
import numpy as np

wifi = np.zeros((1, 3))
num = []    # number of each class
tst_wifi=[]

# Data load
load_data = data_load.data_load(wifi, num, tst_wifi)
wifi, num, tst_wifi = load_data.load_wifi()
# Data preprocessing
pre_data = data_pre.Data_pre(wifi, num, tst_wifi)
wifi_X, wifi_Y, tst_X = pre_data.replace()
# Data normalization
std = StandardScaler()
train_X = std.fit_transform(wifi_X)
tst_X = std.transform(tst_X)



#### Normal LDA ####
component = len(np.unique(wifi_Y)) - 1      # Largest Eigenvalue Ratio
lda = LinearDiscriminantAnalysis(n_components=component)
train_XX = lda.fit_transform(train_X, wifi_Y)
tst_XX = lda.transform(tst_X)

# Average of each class
c_mean = data_mean.Data_mean(train_XX, wifi_Y)
class_mean = c_mean.data_mean()

# Distance between class_mean(train_mean) and tst_XX
dis = data_distance.Dis(class_mean, tst_XX)
data_dis = dis.D_dis()

L=len(class_mean[0, :])
for i in range(0, L+1):
    plt.subplot(4, 4, i+1)
    plt.plot(data_dis[i, :], color='black', label=i+1)
    plt.legend(ncol=14)
plt.suptitle('Euclid_Distance between Landmark and Test_data(Smoothing)', fontsize=16)
plt.show()

# (sum of n eigenvalues)/(sum of all eigenvalues)
# a = lda.explained_variance_ratio_
# x = range(len(a)+1)
# j = [0]
# for i in range(len(a)):
#     j.append(j[-1]+a[i])
# plt.plot(x, j, label='ratio')
# plt.legend()
# plt.show()



#### Kernel LDA ####
# RBF
klda = Kfda(kernel='rbf', n_components=component)
k_train_XX = klda.fit_transform(train_X, wifi_Y)
k_tst_XX = klda.transform(tst_X)

# Average of each class
k_c_mean = data_mean.Data_mean(k_train_XX, wifi_Y)
k_class_mean = k_c_mean.data_mean()

# Distance between class_mean(train_mean) and tst_XX
k_dis = data_distance.Dis(k_class_mean, k_tst_XX)
k_data_dis = k_dis.D_dis()

for i in range(0, len(k_class_mean[0, :])):
    plt.subplot(4, 4, i+1)
    plt.plot(k_data_dis[i, :], color='black', label=i+1)
    plt.legend(ncol=14)
plt.suptitle('Distance between Landmark and Test_data(Smoothing)', fontsize=16)
plt.show()