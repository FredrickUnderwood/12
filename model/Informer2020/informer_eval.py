import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 指定.npy文件路径
file_path1 = "results/informer_custom_ftMS_sl100_ll50_pl1_dm512_nh20_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/real_prediction.npy"
file_path2 = "results/informer_custom_ftMS_sl100_ll50_pl1_dm512_nh20_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/pred.npy"
file_path3 = "results/informer_custom_ftMS_sl100_ll50_pl1_dm512_nh20_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/true.npy"
# 使用NumPy加载.npy文件
true_value = []
pred_value = []
test_value = []

data1 = np.load(file_path1)
data2 = np.load(file_path2)
data3 = np.load(file_path3)
print(data3)
print(data1.shape, data2.shape, data3.shape)
for i in range(data2.shape[0]):
    # for j in range(data2.shape[1]):
    true_value.append(data3[i][0][0])
    pred_value.append(data2[i][0][0])
test_value = data3.flatten().tolist()
plt.plot(true_value, label="true")
plt.plot(pred_value, label="pred")
plt.legend()
plt.show()