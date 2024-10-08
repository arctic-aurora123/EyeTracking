import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

df = pd.read_csv('./data/continuous/continuous_train.csv')

def plot_signal(start, end):
    end *= 250
    time = np.arange(start, end+1)
    plt.figure(figsize=(20, 10))
    plt.title('右眼偏置信号')
    plt.plot(time, df.loc[start:end, 'vr'], label='右眼水平偏置')
    plt.plot(time, df.loc[start:end, 'hr'], label='右眼垂直偏置')
    plt.show()
    
if __name__ == '__main__':
    plot_signal(0, 30)