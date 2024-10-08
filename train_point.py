import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Eyedata import EyeDataset, EyeDataset_continuous
from EyeNet import EyeTrackNet_seq, EyeTrackNet_point
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    
    writer = SummaryWriter("logs/1008_rnn_con")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = './data/continuous/continuous_train.csv'
    dataset = EyeDataset_continuous(data_path, overlap = True, window_size = 50, step_size = 5, ignore_first_sec = 5)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    print(f"Train Dataset size: {len(train_dataset.indices)}")
    print(f"Test Dataset size: {len(test_dataset.indices)}")

    train_data = DataLoader(train_dataset,batch_size=64, shuffle=True, drop_last=True)
    test_data = DataLoader(test_dataset,batch_size=64, shuffle=True, drop_last=True)

    model = EyeTrackNet_point().to(device)

    lr = 0.005
    num_epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_data), desc=f"Train epoch {epoch}: ") as pbar:
            sum = 0
            for batch_idx, (inputs, pos_target, blink_target) in enumerate(train_data):
                inputs = inputs.to(device)
                pos_target = pos_target.to(device)
                blink_target = blink_target.to(device)
                
                optimizer.zero_grad()
                (pos_x, pos_y), blink = model(inputs)
                pos = torch.cat([pos_x, pos_y], dim=1)
                train_pos_loss = nn.MSELoss(pos, pos_target)
                train_blink_loss = nn.MSELoss(blink, blink_target)
                train_loss = train_pos_loss + train_blink_loss
                sum += train_loss.item()
                train_loss.backward()
                optimizer.step()
                pbar.update(1) 
        pbar.close()   
        print(f"Train loss: {sum/len(train_data)}")
        if epoch % 5 == 0:    
            writer.add_scalar("Loss/train/pos", train_pos_loss.item() / len(train_data), epoch)
            writer.add_scalar("Loss/train/blink", train_pos_loss.item() / len(train_data), epoch)
            writer.add_scalar("Loss/train/train_loss", train_loss.item() / len(train_data), epoch)
            
        model.eval()  
        with torch.no_grad():
            with tqdm(total=len(test_data), desc=f"Test epoch {epoch}: ") as pbar:
                test_loss = 0
                for inputs, targets in test_data:
                    inputs = inputs.to(device)
                    pos_target = pos_target.to(device)
                    blink_target = blink_target.to(device)
                    (pos_x, pos_y), blink = model(inputs)
                    pos = torch.cat([pos_x, pos_y], dim=1)
                    test_pos_loss = nn.MSELoss(pos, pos_target).item() 
                    test_blink_loss = nn.MSELoss(blink, blink_target).item()
                    test_loss = test_pos_loss + test_blink_loss
                    pbar.update(1)    
            pbar.close()
            print(f"Test loss: {test_loss / len(test_data)}")
            if epoch % 5 == 0: 
                writer.add_scalar("Loss/test/pos", test_pos_loss / len(test_data), epoch)
                writer.add_scalar("Loss/test/blink", test_blink_loss/ len(test_data), epoch)
                writer.add_scalar("Loss/test/test_loss", test_loss/ len(test_data), epoch)

    torch.save(model, './model/continuous_model_1.pth')
    print("model saved at: ./model/continuous_model_1.pth")
    writer.close()