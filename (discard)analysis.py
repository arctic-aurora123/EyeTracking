from types import FrameType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.signal import butter, filtfilt

import os
from tqdm import *

file_path = os.path.abspath('C:/Users/Arctic/Documents/ads1299/saved')

pos_list = [
    'pos_count0.csv',
    'pos_count1.csv',
    'pos_count2.csv',
    'pos_count3.csv',
    'pos_count4.csv',
    'pos_count5.csv',
    'pos_count6.csv',
    # 'pos_count7.csv',
    # 'pos_count8.csv',
    # 'pos_count9.csv',
]

volts_list = [
    'Device_0_Volts.csv',
    'Device_1_Volts.csv',
    'Device_2_Volts.csv',
    'Device_3_Volts.csv',
    'Device_4_Volts.csv',
    'Device_5_Volts.csv',
    'Device_6_Volts.csv',
    # 'Device_7_Volts.csv',
    # 'Device_8_Volts.csv',
    # 'Device_9_Volts.csv',
]

data_list = [
    'data0.csv',
    'data1.csv',
    'data2.csv',
    'data3.csv',
    'data4.csv',
    'data5.csv',
    'data6csv',
]

def discrete_diff(signal, dt=1):
    return [(signal[i+1] - signal[i]) / dt for i in range(len(signal) - 1)]
def data_preprocess(pos_list, volts_list):
    fs = 250
    lpf = 10
    # m_lpf = 0.1
    LPFb, LPFa = butter(N=3, Wn=lpf/(fs/2), btype='low', analog=False)
    # LPFb_m, LPFa_m = butter(N=1, Wn=m_lpf/(fs/2), btype='low', analog=False)
    scaler = MinMaxScaler()
    
    data = pd.DataFrame(columns=['vertical_left', 'vertical_right', 'horizontal', 'pos_x', 'pos_y', 'label', 'cnt'])
    for i in range(len(volts_list)):
        df_pos = pd.read_csv(os.path.join(file_path, pos_list[i]))
        df_volts = pd.read_csv(os.path.join(file_path, volts_list[i]))

        df = pd.DataFrame(columns=['vertical_left', 'vertical_right', 'horizontal', 'pos_x', 'pos_y', 'label', 'cnt'])

        df['vertical_left'] = filtfilt(LPFb, LPFa, df_volts['CH1']) 
        df['vertical_right'] = filtfilt(LPFb, LPFa, df_volts['CH2']) 
        df['horizontal'] = filtfilt(LPFb, LPFa, df_volts['CH3']) 
        
        # df['vertical_left'] = df_volts['CH1']
        # df['vertical_right'] = df_volts['CH2']
        # df['horizontal'] = df_volts['CH3'] 
        
        idx = 0
        for j in range(df_pos.shape[0]):
            row_start = int(df_pos.loc[j, 'start_cnt'])
            row_end = int(df_pos.loc[j, 'end_cnt'])
            
            df.loc[row_end, 'pos_x'] = (df_pos.loc[j, 'pos_x'] - 1280) / 2560 
            df.loc[row_end, 'pos_y'] = (df_pos.loc[j, 'pos_y'] - 720) / 1440
            df.loc[row_end, 'label'] = 1
            df.loc[row_end, 'cnt'] = df_pos.loc[j, 'end_cnt']
        # for j in range(df.shape[0]):
        #     if j == df_pos.loc[idx, 'end_cnt'] and idx < df_pos.shape[0]-1:
        #         print(idx)
        #         df.loc[int(df_pos.loc[idx, 'end_cnt']), 'label'] = 1
        #         idx += 1
        #     else:
        #         df.loc[j, 'pos_x'] = (df_pos.loc[idx, 'pos_x'] - 1280) / 2560 
        #         df.loc[j, 'pos_y'] = (df_pos.loc[idx, 'pos_y'] - 720) / 1440
        #         df.loc[j, 'label'] = 1
        df[col_scale] *= 1000
        # df.to_csv(f'data_raw{i}.csv', index=False)
        df = df.dropna(axis=0, how='any')

        df.to_csv(f'data{i}.csv', index=False)
        data = pd.concat([data, df], ignore_index=True)
    # data[col_scale] = scaler.fit_transform(data[col_scale])
    
    data.to_csv(f'data.csv', index=False)
    
def regression(X, y, max_depth, random_state):
    model = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X, y)
    return model
    
# def update(frame): 
#     point.set_data(y_hat[frame-10:frame, 0], y_hat[frame-10:frame, 1])
#     point_target.set_data(y_target[frame-10:frame, 0], y_target[frame-10:frame, 1])
#     return point, point_target
    
if __name__ == '__main__':
    col_scale = ['vertical_left', 'vertical_right', 'horizontal']
    data_preprocess(pos_list, volts_list)
    fig, ax = plt.subplots()
    scaler = MinMaxScaler()
    df = pd.read_csv('data5.csv')
    
    X = df[['vertical_left', 'vertical_right','horizontal']].values
    y = df[['pos_x', 'pos_y']].values
    
    model = regression(X, y, max_depth=30, random_state=0)
    print(X.shape)
    print(f'dataset score: {model.score(X, y)}')
    
    # for j in range(len(volts_list)):
    j = 6
    df_raw = pd.read_csv(f'data_raw{j}.csv')
    df_target = pd.read_csv(f'data{j}.csv')
    # df[col_scale] = scaler.fit_transform(df[col_scale])
    idx = 0
    
    df_plot = pd.DataFrame(columns=['vertical_left', 'vertical_right', 'horizontal', 'pos_x', 'pos_y', 'label', 'cnt'])
    x = np.arange(0, len(df_target)*50)
    y = np.arange(0, len(df_target)*50)
    # x = np.arange(0, len(df_target))
    # y = np.arange(0, len(df_target))
    m = np.arange(0, len(df_target)*50-1, 50)
    
    for seg in df_target['cnt']:
        seg = int(seg)
        df_plot= pd.concat([df_plot, df_raw[(seg-50):(seg)]], ignore_index=True)
    vl = df_plot['vertical_left'].values
    vr = df_plot['vertical_right'].values
    h = df_plot['horizontal'].values
    
    v_diff = np.gradient(df_raw['vertical_left'].values)
    # vl = df_raw['vertical_left'].values * 1000
    # vr = df_raw['vertical_right'].values * 1000
    # h = df_raw['horizontal'].values * 1000
    
    # signal = discrete_diff(vl)
    signal_x = v_diff
    signal_y = h
    
    x = np.arange(0, len(df_raw))
    plt.plot(x, signal_x)
    # plt.plot(x, df_target['pos_x']/2, 'b-')
    
    # plt.plot(y, -signal_y/10, 'r.')
    # plt.plot(y, df_target['pos_y'], 'b-')
    
    # plt.plot(y, signal_y)
    for m in df_target['cnt']:
        plt.scatter(m, 0)
    # plt.scatter(m, np.zeros_like(m))
    plt.show()
    
    X_hat = df_raw[['vertical_left', 'vertical_right','horizontal']].values
    y_target = df_raw[['pos_x','pos_y']].values
    # X_target = df_target[['vertical_left', 'vertical_right','horizontal']].values
    # y_target = df_target[['pos_x', 'pos_y']].valuess
    y_hat = model.predict(X_hat)
    
    
    
    # # pbar = tqdm(X_hat.shape[0]) 
    # point, = ax.plot(y_hat[:,0], y_hat[:,1], 'r.')
    # point_target, = ax.plot(y_target[:,0], y_target[:,1], 'bx')
    # # target, = ax.plot(y_target[:,0], y_target[:,1], 'bx')
    # # plt.scatter(y_target[:, 0], y_target[:, 1], c='b', marker='x')
    # idx = 0
    # anim = animation.FuncAnimation(fig, update, frames=X_hat.shape[0], interval=1, blit=True)
    # # animation.ArtistAnimation(fig, frames, interval=1, blit=True)
    # plt.show()
    # anim.save(f'plot.mp4', writer='ffmpeg', fps=60)