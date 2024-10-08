from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/train_data.csv')

data = []
for i in range(int(len(df)/100)):
    data.append(df.iloc[i, :].values)

data = np.stack(data, axis=0)

print(data.shape)
vr = data[5:, 0]
vl = data[5:, 1]

hr = data[5:, 2]
hl = data[5:, 3]

# dv = data[5:, 4]
# dh = data[5:, 5]

pos_x = data[5:, 6]
pos_y = data[5:, 7]

print(vr.shape)
print(pos_x.shape)
# plt.plot(range(len(vr)), vr)
# plt.show()
results_vr = stats.pearsonr(vr, pos_x)
results_vl = stats.pearsonr(vl, pos_x)
results_hr = stats.pearsonr(hr, pos_y)
results_hl = stats.pearsonr(hl, pos_y)

print(f'左边眼睛水平： {results_vr}')
print(f'右边眼睛水平： {results_vl}')
print(f'左边眼睛竖直： {results_hr}')
print(f'右边眼睛竖直： {results_hl}')
