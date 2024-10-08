import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Eyedata import EyeDataset, EyeDataset_continuous
from EyeNet import EyeTrackNet_MLP
from tqdm import tqdm
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    window_size = 100
    batch_size = 32
    writer = SummaryWriter("logs/1008_mlp_con_5")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = './data/continuous/continuous_train.csv'
    dataset = EyeDataset_continuous(data_path, overlap = True, window_size = window_size, step_size = 5, ignore_first_sec = 5)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    print(f"Train Dataset size: {len(train_dataset.indices)}")
    print(f"Test Dataset size: {len(test_dataset.indices)}")

    train_data = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    test_data = DataLoader(test_dataset,batch_size=batch_size, shuffle=True, drop_last=True)

    model = EyeTrackNet_MLP(input_channel=4, input_seq_len=window_size).to(device)
    
    criterion = nn.MSELoss()
    lr = 5e-2
    num_epochs = 300
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_data), desc=f"Train epoch {epoch}: ") as pbar:
            sum = 0
            for batch_idx, (inputs, pos_target, blink_target) in enumerate(train_data):
                inputs = inputs.to(device)
                inputs = torch.flatten(inputs, 1)
                pos_target = pos_target[:, -1, :].unsqueeze(1).to(device)
                blink_temp = torch.mean(blink_target, dim=1, keepdim=True).to(device)
                blink_target = blink_temp.masked_fill(blink_temp != 0, 1)
                optimizer.zero_grad()
                val = model(inputs)
                pos = val[:, :2]
                blink = val[:, -1]

                train_pos_loss = criterion(pos, pos_target)
                train_blink_loss = criterion(blink, blink_target)
                train_loss = train_pos_loss + train_blink_loss
                sum += train_loss.item()
                train_loss.backward()
                optimizer.step()
                pbar.update(1) 
        pbar.close()   
        print(f"Train loss: {sum/len(train_data)}")   
        if epoch % 3 == 0:    
            writer.add_scalar("Loss/train_pos", train_pos_loss.item() / len(train_data), epoch)
            writer.add_scalar("Loss/train_blink", train_pos_loss.item() / len(train_data), epoch)
            writer.add_scalar("Loss/train", train_loss.item() / len(train_data), epoch)
            
        model.eval()  
        with torch.no_grad():
            with tqdm(total=len(test_data), desc=f"Test epoch {epoch}: ") as pbar:
                test_loss = 0
                for batch_idx, (inputs, pos_target, blink_target) in enumerate(test_data):
                    inputs = inputs.to(device)
                    inputs = torch.flatten(inputs, 1)
                    pos_target = pos_target[:, -1, :].unsqueeze(1).to(device)
                    blink_temp = torch.mean(blink_target, dim=1, keepdim=True).to(device)
                    blink_target = blink_temp.masked_fill(blink_temp != 0, 1)
                    
                    val = model(inputs)
                    pos = val[:, :2]
                    blink = val[:, -1]
                    
                    test_pos_loss = criterion(pos, pos_target).item()
                    test_blink_loss = criterion(blink, blink_target).item()
                    test_loss = test_pos_loss  + test_blink_loss
                    pbar.update(1)    
            pbar.close()
            print(f"Test loss: {test_loss / len(test_data)}")
            if epoch % 3 == 0: 
                writer.add_scalar("Loss/test_pos", test_pos_loss/ len(test_data), epoch)
                writer.add_scalar("Loss/test_blink", test_blink_loss/ len(test_data), epoch)
                writer.add_scalar("Loss/test", test_loss/ len(test_data), epoch)

    torch.save(model, 'model/mlp_con_model_5.pth')
    print("model saved at: ./model/mlp_con_model_4.pth")
    writer.close()