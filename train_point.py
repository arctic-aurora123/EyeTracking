import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Eyedata import EyeDataset
from EyeNet import EyeTrackNet_seq, EyeTrackNet_point
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    
    writer = SummaryWriter("logs/0915_rnn")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = './data/train_data.csv'
    dataset = EyeDataset(data_path, overlap = True, window_size = 10, step_size = 5, ignore_first_sec = 5)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    print(f"Train Dataset size: {len(train_dataset.indices)}")
    print(f"Test Dataset size: {len(test_dataset.indices)}")

    train_data = DataLoader(train_dataset,batch_size=64, shuffle=True, drop_last=True)
    test_data = DataLoader(test_dataset,batch_size=64, shuffle=True, drop_last=True)

    model = EyeTrackNet_point().to(device)
    
    criterion = nn.MSELoss()
    lr = 0.005
    num_epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_data), desc=f"Train epoch {epoch}: ") as pbar:
            sum = 0
            for batch_idx, (inputs, targets) in enumerate(train_data):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                targets = torch.mean(targets, dim=1)

                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                sum += loss.item()
                loss.backward()
                optimizer.step()
                pbar.update(1) 
        pbar.close()   
        print(f"Train loss: {sum/len(train_data)}")    
        writer.add_scalar("Loss/train", loss.item(), epoch) 
            
        model.eval()  
        with torch.no_grad():
            with tqdm(total=len(test_data), desc=f"Test epoch {epoch}: ") as pbar:
                test_loss = 0
                for inputs, targets in test_data:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    targets = torch.mean(targets, dim=1)
                    
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item()
                    pbar.update(1)    
            pbar.close()
            print(f"Test loss: {test_loss / len(test_data)}")
            writer.add_scalar("Loss/test", test_loss / len(test_data), epoch)

    torch.save(model, 'point_model.pth')
    print("model saved at: ./point_model.pth")
    writer.close()