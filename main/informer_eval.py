import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path2 = "results/Test 1/pred.npy"
file_path3 = "results/Test 1/true.npy"
# 使用NumPy加载.npy文件
true_value = []
pred_value = []
test_value = []


data2 = np.load(file_path2)
data3 = np.load(file_path3)
print(data3)
print(data2.shape, data3.shape)
for i in range(data2.shape[0]):
    # for j in range(data2.shape[1]):
    true_value.append(data3[i][0][0])
    pred_value.append(data2[i][0][0])
test_value = data3.flatten().tolist()
plt.figure(figsize=(20, 12))
plt.plot(true_value, label="true")
plt.plot(pred_value, label="pred")
plt.axvline(x=0.7*len(true_value))
plt.legend()
plt.show()