import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from Eyedata import EyeData
        
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

df = pd.read_csv('./data/latest data/std/continuous_train_std2.csv')
print(df.shape)
def plot_signal(start, end):
    end *= 250
    time = np.arange(start, end+1)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df.loc[start:end, 'vl'], label='左眼水平偏置')
    plt.plot(df.loc[start:end, 'vr'], label='右眼水平偏置')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(df.loc[start:end, 'hl'], label='左眼垂直偏置')

    plt.plot(df.loc[start:end, 'hr'], label='右眼垂直偏置')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    plot_signal(0, 30)