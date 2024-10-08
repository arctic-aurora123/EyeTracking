import torch
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.signal import butter, filtfilt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

data_path = "C:/Users/Arctic/Documents/ads1299/saved/"
data_file = [
    # "data0_test_circle.csv",
    # "data0_test_circle_original.csv",
    "volts0.csv",
    "volts1.csv",
    "volts2.csv",
    # "volts3.csv",
    # "volts4.csv",
    # "volts5.csv",
]

count_file = [
    "pos_count0.csv",
    "pos_count1.csv",
    "pos_count2.csv",
    # "pos_count3.csv",
    # "pos_count4.csv",
    # "pos_count5.csv",
]
sample_rate = 250

# print(df.head())

def standardize(data):
    return (data - np.mean(data))/np.std(data)

def minmax_normalize(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))

def lpf(data, lowcut, order=2):
    b, a = butter(order, 2*lowcut/sample_rate, 'lowpass')
    return filtfilt(b, a, data)


def process_data(i, start = 0):
    
    df = pd.read_csv(data_path+data_file[i])
    df_pos = pd.read_csv(data_path+count_file[i])
    
    # vr = standardize(df['CH1'][start:].values)
    # vl = standardize(df['CH3'][start:].values)
    # hr = standardize(df['CH2'][start:].values)
    # hl = standardize(-df['CH4'][start:].values)
    # delta_v = vr - vl
    # delta_h = hr - hl
    
    start_count = df_pos['start_cnt']
    end_count = df_pos['end_cnt']
    
    vr = minmax_normalize(df['CH1'].values)
    vl = minmax_normalize(df['CH3'].values)
    hr = minmax_normalize(df['CH2'].values)
    hl = minmax_normalize(-df['CH4'].values)
    
    delta_v = vr - vl
    delta_h = hr - hl
    value = {'vr':vr, 'vl':vl, 'hr':hr, 'hl':hl, 'dv':delta_v, 'dh':delta_h}
    offset = {'vr':vr, 'vl':vl, 'hr':hr, 'hl':hl, 'dv':delta_v, 'dh':delta_h}
    std = {'vr':vr, 'vl':vl, 'hr':hr, 'hl':hl, 'dv':delta_v, 'dh':delta_h}
    
    for k, m in value.items(): 
        value[k] = lpf(m, 5, 3)
        offset[k] = lpf(m, 0.08, 3)
        std[k] = value[k] - offset[k]

    df_target = pd.DataFrame(columns=['vr','vl','hr','hl', 'pos_x', 'pos_y', 'label'])
    
    for j in range(df_pos.shape[0]):
        # row_start = int(df_pos.loc[j, 'start_cnt'])
        row_end = int(df_pos.loc[j, 'end_cnt'])
        
        df_target.loc[row_end, 'pos_x'] = (df_pos.loc[j, 'pos_x'] - 1280) / 2560 
        df_target.loc[row_end, 'pos_y'] = (df_pos.loc[j, 'pos_y'] - 720) / 1440
        df_target.loc[row_end, 'label'] = 1
        df_target.loc[row_end, 'cnt'] = df_pos.loc[j, 'end_cnt']
    df_target.to_csv(f'data_target{i}.csv', index=False)
    
    time = np.arange(0, len(df)/250, 1/250)
    
    plt.figure(figsize=(10, 12))
    
    plt.subplot(4, 1 ,1)
    # plt.title('右眼原始信号')
    # plt.plot(time, value['vr'], label='右眼水平')
    # plt.plot(time, value['hr'], label='右眼垂直')
    # plt.plot(time, offset['vr'], label='右眼水平趋势', color='y')
    # plt.plot(time, offset['hr'], label='右眼水平趋势', color='g')
    # plt.legend()
    # plt.xlabel('时间(s)')
    # plt.ylabel('归一化电压')
    
    plt.title('左眼原始信号')
    plt.plot(time, value['vl'], label='左眼水平')
    plt.plot(time, value['hl'], label='左眼垂直')
    plt.plot(time, offset['vl'], label='左眼水平趋势', color='y')
    plt.plot(time, offset['hl'], label='左眼水平趋势', color='g')
    plt.legend()
    plt.xlabel('时间(s)')
    plt.ylabel('归一化电压')
    
    plt.subplot(4, 1 ,2)
    # plt.title('右眼偏置信号')
    # plt.plot(time, std['vr'], label='右眼水平偏置')
    # plt.plot(time, std['hr'], label='右眼垂直偏置')
    
    plt.title('左眼偏置信号')
    plt.plot(time, std['vl'], label='左眼水平偏置')
    plt.plot(time, std['hl'], label='左眼垂直偏置')
    
    plt.legend()
    plt.xlabel('时间(s)')
    plt.ylabel('归一化电压')
    
    plt.subplot(4, 1,3)
    # plt.title('右眼微分信号')
    # plt.plot(time, np.gradient(std['vr']), label='右眼水平微分')
    # plt.plot(time, np.gradient(std['hr']), label='右眼垂直微分')
    
    plt.title('左眼微分信号')
    plt.plot(time, np.gradient(std['vl']), label='左眼水平微分')
    plt.plot(time, np.gradient(std['hl']), label='左眼垂直微分')
    
    plt.legend()
    plt.xlabel('时间(s)')
    plt.ylabel('归一化电压')
    
    plt.subplot(4, 1,4)
    plt.title('差分信号')
    plt.plot(time, value['dv'], label='水平差分')
    plt.plot(time, value['dh'], label='垂直差分')
    
    plt.xlabel('时间(s)')
    plt.ylabel('归一化电压')
    
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    return df_target, df, value, offset, std
def regression(X, y, max_depth, random_state):
    model = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X, y)
    return model


def update(frame): 
    point.set_data(data_points[frame:frame+1, 0], data_points[frame:frame+1, 1])
    point_target.set_data(test_data.loc[frame:frame+1, 'pos_x'].values, test_data.loc[frame:frame+1, 'pos_y'].values)
    return point, point_target

if __name__ == '__main__':
    
    s = 2
    # j = 0
    _, signal, raw, _, std = process_data(s)
    
    # target = pd.read_csv(f'data_target{s}.csv')
    # raw = pd.DataFrame(raw)
    # std = pd.DataFrame(std)
    # idx = target['cnt'].values.astype(np.int32)

    # m = 0

    # for n in range(len(std)):
    #     if n <= idx[m]:
    #         std.loc[n, 'pos_x'] = target.loc[m, 'pos_x']
    #         std.loc[n, 'pos_y'] = target.loc[m, 'pos_y']
    #     else:
    #         m += 1
    #         if m > len(idx)-1:
    #             break
    #     print(n)
    # std.to_csv(f'std_data{s}.csv', index=False)
    
    # train_data =  pd.DataFrame(columns=['vr','vl','hr','hl', 'dv', 'dh'])
    # for i in idx:
    #     train_data = pd.concat([train_data, std.loc[i:i+99]], ignore_index=True)
    #     train_data.loc[j:j+99,'pos_x'] = target['pos_x'][i]
    #     train_data.loc[j:j+99,'pos_y'] = target['pos_y'][i]
    #     j += 100
    
    # train_data = train_data.dropna()
    # train_data.to_csv(f'train_data{s}.csv', index=False)
    
    train_data0 = pd.read_csv('train_data0.csv')
    train_data1 = pd.read_csv('train_data1.csv')
    train_data2 = pd.read_csv('train_data2.csv')
    
    train_data = pd.concat([train_data0, train_data1, train_data2], ignore_index=True)
    
    train_data.to_csv('train_data.csv', index=False)
    data = np.split(train_data.values, len(train_data)/25)
    print(torch.tensor(data).shape)
    
    test_data = pd.read_csv('std_data2.csv')
    
    target0 = pd.read_csv('data_target0.csv')
    target1 = pd.read_csv('data_target1.csv')
    target2 = pd.read_csv('data_target2.csv')
    
    target = pd.concat([target0, target1, target2], ignore_index=True).to_csv(
        'target.csv', index=False
    )
    
    X = train_data.loc[:, ['vr','vl','hr','hl']].values
    y = train_data.loc[:, ['pos_x', 'pos_y']].values
    test= test_data.loc[:, ['vr','vl','hr','hl']].values

    model = regression(X, y, max_depth=50, random_state=0)
    print(f'dataset score: {model.score(X, y)}')
    
    data_points = model.predict(test)
    
    # print(data_points)
    fig, ax = plt.subplots()

    point, = ax.plot(data_points[:,0], data_points[:,1], 'r.')
    point_target, = ax.plot(test_data.loc[:,'pos_x'].values, test_data.loc[:,'pos_y'].values, 'bx')
    anim = animation.FuncAnimation(fig, update, frames=data_points.shape[0], interval=2, blit=True)
    # plt.show()
    # anim.save(f'plot.mp4', writer='ffmpeg', fps=60)
    