import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from Eyedata import EyeData

sample_rate = 250
start = 0
min_blink_period = 0.1
blink_period, press_delay = 0.4, 0.1
    
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def lpf(data, lowcut, order=2):
    b, a = butter(order, 2*lowcut/sample_rate, 'lowpass')
    return filtfilt(b, a, data)

def minmax_normalize(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))

def cosine_similarity(vector1, vector2):
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    similarity = np.dot(unit_vector1, unit_vector2)
    return similarity
    
def preprocess(csv_file, pkl_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    
    with open(pkl_file, 'rb') as f:
        pkl = pickle.load(f)
    temp = EyeData()
    for i in range(len(pkl)-1):
        temp = pkl[i]
        start, end = temp.start_cnt, temp.end_cnt
        
        pos_x = np.linspace(temp.pos_x[0], temp.pos_x[-1], end-start)
        g = interp1d(temp.pos_x, temp.pos_y, kind ="linear")
        pos_y = g(pos_x)
        pos_x = (pos_x - 1280) / 2560 
        pos_y = (pos_y - 720) / 1440
        
        # blink_label = np.zeros_like(pos_x)
        
        # for label in temp.blink_time:
        #     start_blink = int(label-press_delay*250) - start
        #     end_blink = int(label-(press_delay-blink_period)*250) - start
        #     blink_label[start_blink:end_blink] = 1
    
    vl = minmax_normalize(data[:, 0])
    vr = minmax_normalize(data[:, 1])
    hl = minmax_normalize(data[:, 2])
    hr = minmax_normalize(-data[:, 3])
        
    vl = lpf(vl, 5, 3)
    hl = lpf(hl, 5, 3)
    vr = lpf(vr, 5, 3)
    hr = lpf(hr, 5, 3)

    mean = np.mean((vl + vr) / 2)
    vl_peaks_id, properties = find_peaks(vl, distance=sample_rate * min_blink_period, height=mean, prominence=(0.1, 1))
    
    vr_peaks_id, _ = find_peaks(vr)

    vl_peaks = vl[vl_peaks_id]
    vr_peaks = vr[vr_peaks_id]

    offset_vl = lpf(vl, 0.08, 3)
    offset_vr = lpf(vr, 0.08, 3)
    offset_hl = lpf(hl, 0.08, 3)
    offset_hr = lpf(hr, 0.08, 3)

    std_vl = vl - offset_vl
    std_vr = vr - offset_vr
    std_hl = hl - offset_hl
    std_hr = hr - offset_hr
    values = [vl, vr, hl, hr]
    offsets = [offset_vl, offset_vr, offset_hl, offset_hr]
    std = [std_vl, std_vr, std_hl, std_hr]
    return [values, offsets, std]

    # with open('C:/Users/Arctic/Desktop/similarity.csv', 'a') as file:
    #     print(f"{idx}, 水平相似度(value), {cosine_similarity(vl, vr)}", file=file)
    #     print(f"{idx}, 垂直相似度(value), {cosine_similarity(hl, hr)}", file=file)
    #     print(f"{idx}, 水平相似度(std), {cosine_similarity(std_vl, std_vr)}", file=file)
    #     print(f"{idx}, 垂直相似度(std), {cosine_similarity(std_hl, std_hr)}", file=file)
    #     # print(f"{idx}, 水平峰值相似度(value), {cosine_similarity(vl_peaks_id, vr_peaks_id)}", file=file)

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data.iloc[:, 0:4].values

def peaks_intep(std):
    delete_idx = []
    original_idx = np.arange(len(std))
    
    def peaks_finding(std):
        std_vl_peaks_id, _ = find_peaks(std[:, 0], distance = sample_rate * min_blink_period, prominence=(0.04, 1))
        print(std_vl_peaks_id)
        return std_vl_peaks_id

    std_vl_peaks_id = peaks_finding(std)
    
    for i, peaks_id in enumerate(std_vl_peaks_id):
        start_blink = int(peaks_id - blink_period * sample_rate / 2)
        end_blink = int(peaks_id + blink_period * sample_rate / 2)
        delete_idx.append(np.arange(start_blink, end_blink))
    
    valid_idx = np.delete(original_idx, delete_idx)
    interp_func_vl = interp1d(valid_idx, std[valid_idx, 0], kind='linear')
    interp_func_vr = interp1d(valid_idx, std[valid_idx, 1], kind='linear')
    
    std_interpolated = std.copy()
    
    std_interpolated[delete_idx, 0] = interp_func_vl(delete_idx) 
    std_interpolated[delete_idx, 1] = interp_func_vr(delete_idx) 
    
    return std_interpolated, std_vl_peaks_id
                
def export_plots(std, idx):
    # fig, axs = plt.subplots(2, 1, figsize=(15,12))
    plt.figure(figsize = (8, 6))
    labels = ['左眼垂直', '右眼垂直', '左眼水平', '右眼水平']
    plt.title(f'消除drift的信号{idx}')
    
    plt.plot(std[:, 0], label='左眼垂直')
    # plt.plot(std[:, 1], label='右眼垂直')
    
    std_interpolated, std_vl_peaks_id = peaks_intep(std)
    
    plt.plot(std_interpolated[:, 0], label='左眼垂直插值', linestyle='--')
    # plt.plot(std_interpolated[:, 1], label='右眼垂直插值', linestyle='--')
    
    for peaks_id in std_vl_peaks_id:
        start_blink = int(peaks_id - blink_period * sample_rate / 2)
        end_blink = int(peaks_id + blink_period * sample_rate / 2)
        plt.fill_between(np.arange(start_blink, end_blink), np.ones(end_blink-start_blink)/2, np.zeros(end_blink-start_blink), color='yellow')
        
        print(f'{idx}: {start_blink} - {end_blink}')
        
    
    std_peaks = std[std_vl_peaks_id, 0]
    # std_peaks = std[std_vl_peaks_id, 1]
    
    plt.scatter(std_vl_peaks_id, std_peaks, color='blue', marker='x')
    plt.legend()
    # axs[1].set_title(f'原始信号{idx}（归一化+基本滤波）')
    # for signal, label in zip(values, labels):
    #     axs[1].plot(signal, label=label)
    
    # for offset, label in zip(offsets, labels):
    #     axs[1].plot(offset, label=label+'trend', linestyle='--')
        
    # vl_peaks = values[0, vl_peaks_id]
    # axs[1].scatter(vl_peaks_id, vl_peaks, color='blue', marker='x')
    # print(f'{idx} blink list: {temp.blink_time}')
    # prominences = np.sort(properties['prominences'])[-9:-1].min()
    # print(f"{idx} peak prominences: {prominences}")
    # for peaks_id in vl_peaks_id:
    #     start_blink = int(peaks_id - blink_period * sample_rate / 2)
    #     end_blink = int(peaks_id + blink_period * sample_rate / 2)
    #     axs[1].fill_between(np.arange(start_blink, end_blink), np.ones(end_blink-start_blink), np.zeros(end_blink-start_blink), color='yellow')
    #     print(f'{idx}: {start_blink} - {end_blink}')
        
    # axs[1].legend()

    plt.savefig(f'C:/Users/Arctic/Desktop/viz_{idx}.png')
    
if __name__ == '__main__':
    csv_path = 'data/latest data/std/'
    for i in range(20):
        std = load_data(csv_path + f'continuous_train_std{i}.csv') 
        export_plots(std, i)
    