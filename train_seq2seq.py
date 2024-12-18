import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Eyedata import EyeDataset
from EyeNet import EyeTrackNet_seq, EyeTrackNet_point
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data_path = './data/single_point/train_data.csv'
    data_path = './data/continuous/continuous_train.csv'
    dataset = EyeDataset(data_path, overlap = True, window_size = 25, step_size = 5, ignore_first_sec = 5)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    print(f"Train Dataset size: {len(train_dataset.indices)}")
    print(f"Test Dataset size: {len(test_dataset.indices)}")

    train_data = DataLoader(train_dataset,batch_size=64, shuffle=False, drop_last=True)
    test_data = DataLoader(test_dataset,batch_size=64, shuffle=False, drop_last=True)

    model = EyeTrackNet_seq(output_size=3).to(device)
    
    criterion = nn.MSELoss()
    lr = 0.005
    num_epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_data), desc=f"Train epoch {epoch}: ") as pbar:
            sum = 0
            for batch_idx, (inputs, pos_target, blink_target) in enumerate(train_data):
                
                print(inputs.shape, pos_target.shape, blink_target.shape)
                inputs = inputs.to(device)
                pos_target = pos_target.to(device)
                blink_target = blink_target.to(device)
                
                optimizer.zero_grad()
                (pos_x, pos_y), blink = model(inputs)
                pos = torch.cat([pos_x, pos_y], dim=1)
                loss = nn.MSELoss(pos, pos_target) + nn.MSELoss(blink, blink_target)
                # loss = criterion(outputs, targets)
                sum += loss.item()
                loss.backward()
                optimizer.step()
                pbar.update(1) 
        pbar.close()   
        print(f"Train loss: {sum/len(train_data)}")    
            
        model.eval()  
        with torch.no_grad():
            with tqdm(total=len(test_data), desc=f"Test epoch {epoch}: ") as pbar:
                test_loss = 0
                for inputs, targets in test_data:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item()
                    pbar.update(1)    
            pbar.close()
            print(f"Test loss: {test_loss / len(test_data)}")

    torch.save(model, 'continuous_seq_model.pth')
    print("model saved at: ./continuous_seq_model.pth")