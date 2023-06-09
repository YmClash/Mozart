import torch
import torch.nn as nn
import torch.optim as optim



class LSTMModel(nn.Module)

    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(LSTMModel,self).__init__()
        self.hidden_size = hidden_size
        self.num-layers = num_layers

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        ho = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        c0 = torch. zeros(self.num_layers,x.size(0),self.hidden_size)

        out, _ = self.lstm
        out =self.fc(out[])

