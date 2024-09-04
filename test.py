import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update(frame): 
    point.set_data(predict_points[frame:frame+1, 0], predict_points[frame:frame+1, 1])
    point_target.set_data(test_data.loc[frame:frame+1, 'pos_x'].values, test_data.loc[frame:frame+1, 'pos_y'].values)
    return point, point_target

if __name__ == '__main__':
    
    window_size = 100
    step_size = 10
    
    test_data = pd.read_csv('train_data.csv')
    
    model = torch.load("./seq_model.pth", weights_only=False).to('cpu')
    # model = torch.load("./point_model.pth", weights_only=False).to('cpu')
    
    test= test_data.loc[:, ['vr','vl','hr','hl']].values
    test = torch.tensor(test, dtype=torch.float32)
    test = test.unfold(0, window_size, step_size)
    test = torch.permute(test, (0, 2, 1))
    
    predict_points = model(test)
    
    # predict_points = predict_points.detach().numpy()
    predict_points = torch.flatten(predict_points,0,1).detach().numpy()
    # print(predict_points.shape)
    fig, ax = plt.subplots()

    point, = ax.plot(predict_points[:,0], predict_points[:,1], 'r.')
    point_target, = ax.plot(test_data.loc[:,'pos_x'].values, test_data.loc[:,'pos_y'].values, 'bx')
    anim = animation.FuncAnimation(fig, update, frames=predict_points.shape[0], interval=1, blit=True)
    plt.show()
    anim.save(f'plot.mp4', writer='ffmpeg', fps=60)
    
