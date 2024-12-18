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
    window_size = 50
    for idx in range(1):
        batch_size = 64
        writer = SummaryWriter(f"logs/1119_model_3_small")
        model_path = './model/1119_model_3_small.pth'
        data_path = f'./data/latest data/std/intered_train_std_small.csv'
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        dataset = EyeDataset_continuous(data_path, overlap = True, window_size = window_size, step_size = 5, ignore_first_sec = 3)

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

        print(f"Train Dataset size: {len(train_dataset.indices)}")
        print(f"Test Dataset size: {len(test_dataset.indices)}")

        train_data = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
        test_data = DataLoader(test_dataset,batch_size=batch_size, shuffle=True, drop_last=True)

        criterion = nn.MSELoss()
        model = EyeTrackNet_point().to(device)

        lr = 2e-4
        num_epochs = 200
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train()
            with tqdm(total=len(train_data), desc=f"Train epoch {epoch}: ") as pbar:
                sum = 0
                for batch_idx, (inputs, pos_target, blink_target) in enumerate(train_data):
                    inputs = inputs[:, :window_size-1, :].to(device)
                    pos_target = pos_target[:, -1, :].to(device)
                    # blink_temp = torch.mean(blink_target, dim=1).to(device)
                    # blink_target = blink_temp.masked_fill(blink_temp != 0, 1)
                    # # print(blink_target)
                    optimizer.zero_grad()
                    val = model(inputs)
                    pos = val[:, :2]
                    # blink = val[:, 2]
                    
                    train_pos_loss = criterion(pos, pos_target)
                    # train_blink_loss = criterion(blink, blink_target)
                    # train_loss = (train_pos_loss  * 1000 + train_blink_loss)
                    train_loss = train_pos_loss * 100
                    sum += train_loss.item()
                    train_loss.backward()
                    optimizer.step()
                    pbar.update(1) 
                    
            pbar.close()   
            print(f"Train loss: {sum/len(train_data)}")
            if epoch % 3 == 0:    
                writer.add_scalar("Loss/train_pos", sum/len(train_data) / len(train_data), epoch)
                writer.add_scalar("Loss/train", train_loss.item() / len(train_data), epoch)
                
            model.eval()  
            with torch.no_grad():
                with tqdm(total=len(test_data), desc=f"Test epoch {epoch}: ") as pbar:
                    test_loss = 0
                    for batch_idx, (inputs, pos_target, blink_target) in enumerate(test_data):
                        inputs = inputs.to(device)
                        pos_target = pos_target[:, -1, :].to(device)
                        # blink_target = torch.mean(blink_target, dim=1).to(device)
                        
                        val = model(inputs)
                        pos = val[:, :2]
                        # blink = val[:, 2]
                        
                        test_pos_loss = criterion(pos, pos_target).item()
                        # test_blink_loss = criterion(blink, blink_target).item()
                        # test_loss += (test_pos_loss* 1000 + test_blink_loss ) 
                        test_loss += test_pos_loss * 100
                        pbar.update(1)    
                pbar.close()
                print(f"Test loss: {test_loss / len(test_data)}")
                if epoch % 3 == 0: 
                    writer.add_scalar("Loss/test_pos", test_pos_loss/ len(test_data), epoch)
                    # writer.add_scalar("Loss/test_blink", test_blink_loss/ len(test_data), epoch)
                    writer.add_scalar("Loss/test", test_loss/ len(test_data), epoch)
                if epoch % 50 == 0: 
                    torch.save(model, model_path)
                    print(f"model saved at: {model_path}")
                    writer.close()