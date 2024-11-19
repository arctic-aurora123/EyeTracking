import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter
from scipy.signal import periodogram

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

seq = ['左眼垂直', '右眼垂直', '左眼水平', '右眼水平']
sample_rate = 250
start = 500
end = 2900
def read_data(file_path):
    data = pd.read_csv(file_path).values
    data = lpf(data, 10)
    data = data[start:end]
    return data.values

def minmax_normalize(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))

def lpf(data, lowcut, order=2):
    b, a = butter(order, 2*lowcut/sample_rate, 'lowpass')
    return filtfilt(b, a, data)

if __name__ == '__main__':
    for s in range(3):
        if s == 1: 
            continue
        file_path = f'./repeatibility/csv/stick{s}/'
        print(f'./repeatibility/csv/stick{s}/')
        for m, label in enumerate(seq):
            fig = plt.figure(figsize=(10, 6))
            d = []
            for idx in range(6):
                file_name = f'rep_{idx}.CSV'
                data = pd.read_csv(file_path + file_name).values
                print(data.shape)
                data = data[:, m]
                data = lpf(data, 10)
                data = data[start:end]
                d.append(data)
            data = np.array(d)
            
            for idx in range(len(data)):
                t = np.arange(d[idx].shape[0])
                plt.plot(t, d[idx], label=f'{idx * 10} mins')
            plt.title(f'{label}原始信号佩戴1h, 过10min检测')
            plt.xlabel('Time (count)')
            plt.ylabel('Voltage (V)')
            plt.legend()
            plt.savefig(f'./repeatibility/img/stick{s}/{label}原始信号佩戴1h, 过10min检测.png')
            
            mean = []
            fig = plt.figure(figsize=(10, 6))
            for idx in range(len(data)):
                t = np.arange(d[idx].shape[0])
                m = np.ones(d[idx].shape[0])*np.mean(d[idx])
                mean.append(np.mean(d[idx]))
                plt.plot(t, m, label=f'{idx * 10} mins')
            mean = np.array(mean)
            plt.title(f'{label}原始信号均值佩戴1h, 过10min检测')
            plt.xlabel('Time (count)')
            plt.ylabel('Voltage (V)')
            plt.legend()
            plt.savefig(f'./repeatibility/img/stick{s}/{label}原始信号均值佩戴1h, 过10min检测.png')
            print(f'6次原始信号均值标准差：{np.std(mean)}')
            
            fig = plt.figure(figsize=(10, 6))
            r = []
            for idx in range(len(data)):
                t = np.arange(len(data))
                r.append(np.max(d[idx]) - np.min(d[idx]))
            r = np.array(r)
            plt.scatter(t*5, r)
            plt.title(f'{label}原始信号范围佩戴1h, 过10min检测')
            plt.xlabel('Time (min)')
            plt.ylabel('Voltage (V)')
            plt.savefig(f'./repeatibility/img/stick{s}/{label}原始信号范围佩戴1h, 过10min检测')
            print(f'6次原始信号幅值标准差：{np.std(r)}')
            
            for idx in range(len(d)):
                d[idx] = minmax_normalize(d[idx])
            print(d)
            
            fig = plt.figure(figsize=(10, 6))
            for idx in range(len(d)):
                t = np.arange(d[idx].shape[0])
                plt.plot(t, d[idx], label=f'{idx * 10} mins')
            plt.title(f'{label}归一化信号佩戴1h, 过10min检测')
            plt.xlabel('Time (count)')
            plt.ylabel('Voltage (V)')
            plt.legend()
            plt.savefig(f'./repeatibility/img/stick{s}/{label}归一化信号佩戴1h, 过10min检测.png')
            
            mean = []
            fig = plt.figure(figsize=(10, 6))
            for idx in range(len(d)):
                t = np.arange(d[idx].shape[0])
                m = np.ones(d[idx].shape[0])*np.mean(d[idx])
                mean.append(np.mean(d[idx]))
                plt.plot(t, m, label=f'{idx * 10} mins')
            mean = np.array(mean)
            plt.title(f'{label}归一化信号均值佩戴1h, 过10min检测')
            plt.xlabel('Time (count)')
            plt.ylabel('Voltage (V)')
            plt.legend()
            plt.savefig(f'./repeatibility/img/stick{s}/{label}归一化信号均值佩戴1h, 过10min检测.png')
            print(f'6次归一化信号均值标准差：{np.std(mean)}')
            
        plt.close()
            # fig = plt.figure(figsize=(10, 6))
            # angle = []
            # for idx in range(len(d)):
            #     freqs, psd = periodogram(d[idx], sample_rate)
            #     plt.plot(freqs[0:20], psd[0:20], label=f'{idx * 10} mins')
            #     angle.append(np.angle(psd[3]))
            # plt.title(f'{label}归一化信号频谱佩戴1h, 过10min检测')
            # plt.xlabel('Freq')
            # plt.ylabel('M')
            # plt.legend()
            # plt.savefig(f'./repeatibility/img/stick{s}/stick{s}/{label}归一化信号功率谱佩戴1h, 过10min检测.png')
            # # 找到频率为5Hz的索引
            # idx = np.where(freqs == f)[0][0]

            # # 幅值是功率谱密度（psd）的最大值
            # magnitude = np.max(psd)

            # # 相位是对应频率的相位角
            # phase = np.angle(psd[idx])

            # print(f"信号幅值: {magnitude:.2f}")
            # print(f"信号相位: {phase:.2f} rad")