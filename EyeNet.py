import torch
import torch.nn as nn
from torchsummary import summary

class EyeTrackNet_seq(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=2, num_layers=4):
        super(EyeTrackNet_seq, self).__init__()
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size=input_size, 
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    batch_first=True)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(input_size=hidden_size, 
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    batch_first=True)
        
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        _, (hidden, cell) = self.encoder_lstm(x)  
    
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.shape[1], 1)  # (batch_size, sequence_length, hidden_size)
        
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        
        output = self.dense(decoder_output) 
        return output


class EyeTrackNet_point(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=2, num_layers=4):
        super(EyeTrackNet_point, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        out = self.dense(output)  # [batch_size, output_size]
        return out

class EyeTrackNet_MLP(nn.Module):
    def __init__(self):
        super(EyeTrackNet_MLP, self).__init__()
        self.input_size = 400
        self.hidden_size_1 = 512
        self.hidden_size_2 = 256
        self.output_size = 2
        self.mlp = nn.Sequential(nn.Linear(self.input_size, self.hidden_size_1), 
                                nn.ReLU(),
                                nn.Linear(self.hidden_size_1, self.hidden_size_1), 
                                nn.ReLU(), 
                                nn.Linear(self.hidden_size_1, self.hidden_size_2), 
                                nn.ReLU(), 
                                nn.Linear(self.hidden_size_2, self.output_size))
    def forward(self, x):
        return self.mlp(x)