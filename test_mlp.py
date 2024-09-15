import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from EyeNet import EyeTrackNet_MLP
from Eyedata import EyeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def update(frame): 
    point.set_data(predict_points[frame:frame+1, 0], predict_points[frame:frame+1, 1])
    point_target.set_data(targets[frame-50:frame-49, 0], targets[frame-50:frame-49, 1])
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
    data_path = './data/std_data1.csv'
    model_path = './model/mlp_model.pth'
    
    window_size = 10
    step_size = 5
    ignore_first_sec = 5
    moving_window = 5
    batch_size = 64
    
    dataset = EyeDataset(data_path, overlap = True, window_size = window_size,
                        step_size = step_size, ignore_first_sec = ignore_first_sec)
    
    print(f"Test Dataset size: {len(dataset)}")
    
    test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = torch.load(model_path, weights_only=False)
    
    predict_points = []
    targets = []
    
    model.eval()
    
    for batch_idx, (inputs, gt) in enumerate(test_data):
        inputs = inputs.to(device)
        inputs = torch.flatten(inputs, start_dim=1)
        points = model(inputs).cpu().detach().numpy()
        predict_points.append(points)

        target = gt[:,-1,:].cpu().detach().numpy()
        targets.append(target)

    predict_points = np.stack(predict_points, axis=0).reshape(-1, 2)
    targets = np.stack(targets, axis=0).reshape(-1, 2)
    
    predict_points = moving_average(predict_points, moving_window)
    targets = moving_average(targets, moving_window)
    print(predict_points.shape)
    print(targets.shape)
    
    df = pd.DataFrame(predict_points)
    df.to_csv('./data/predict_points.csv', index=False)
    df = pd.DataFrame(targets)
    df.to_csv('./data/targets.csv', index=False)
    
    
    fig, ax = plt.subplots()

    point, = ax.plot(predict_points[:,0], predict_points[:,1], 'r.')
    point_target, = ax.plot(targets[:,0], targets[:,1], 'bx')
    
    anim = animation.FuncAnimation(fig, update, frames=targets.shape[0], interval=2, blit=True)
    # plt.show()
    anim.save(f'plot3.mp4', writer='ffmpeg', fps=60)
    
