import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from EyeNet import EyeTrackNet_MLP
from Eyedata import EyeDataset, EyeDataset_continuous
from torch.utils.data import DataLoader
from tqdm import tqdm

def update(frame): 
    point.set_data(predict_points[frame:frame+1, 0], predict_points[frame:frame+1, 1])
    point_target.set_data(targets[frame-delay:frame-delay+1, 0], targets[frame-delay:frame-delay+1, 1])
    return point, point_target

def moving_average(x, w):
    dim = x.shape[-1]
    results = []
    for i in range(dim):
        temp = np.convolve(x[:,i], np.ones(w), "valid") / w
        results.append(temp)
    return np.stack(results, axis=-1) 

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = './data/latest data/std/intered_train_std11.csv'
    model_path = './model/1119_model_3_small.pth'
    video_path = './animation/plot_1119_model_3-1.mp4'
    pred_export_path = './data/latest data/predicted_data.csv'
    delay = 10
    window_size = 50
    step_size = 5
    ignore_first_sec = 3
    moving_window = 10
    batch_size = 32
    
    dataset = EyeDataset_continuous(data_path, overlap = True, window_size = window_size,
                        step_size = step_size, ignore_first_sec = ignore_first_sec)
    
    print(f"Test Dataset size: {len(dataset)}")
    
    test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    model = torch.load(model_path, weights_only=False)
    
    predict_points = []
    targets = []
    
    model.eval()
    
    for batch_idx, (inputs, pos_target, blink_target) in enumerate(test_data):
        inputs = inputs.to(device)
        pos_target = pos_target[:, -1, :]
        val = model(inputs).cpu().detach().numpy()
        predict_points.append(val[:, :2])
        targets.append(pos_target)

    predict_points = np.stack(predict_points, axis=0).reshape(-1, 2)
    targets = np.stack(targets, axis=0).reshape(-1, 2)
    print(predict_points.shape, targets.shape)
    
    # np.savetxt(pred_export_path, predict_points, delimiter=",")
    predict_points = moving_average(predict_points, moving_window)
    targets = moving_average(targets, moving_window)
    # print(predict_points.shape)
    # print(targets.shape)
    
    fig, ax = plt.subplots()

    point, = ax.plot(predict_points[:,0], predict_points[:,1], 'r.')
    point_target, = ax.plot(targets[:,0], targets[:,1], 'bx')
    anim = animation.FuncAnimation(fig, update, frames=targets.shape[0], interval=2, blit=True)
    plt.show()
    plt.close()
    anim.save(video_path, writer='ffmpeg', fps=60)
    
