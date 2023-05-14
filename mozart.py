import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pretty_midi
import numpy as np
import matplotlib as plt
import mido
import pickle
import json
from torchvision.datasets import MNIST
import midi2np




train = MNIST(root="data", download=True,)

# on  class  une classe d'reseau
class LSMTModel(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers, output_size) :
        super(LSMTModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x) :
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# on  ajoute une  fichie  pour servir d'entrainement
midi_file = mido.MidiFile(r'MIDI/Caribbean-Blue.mid',clip=True)
result_array = midi2np.result_array(midi_file)
print(midi_file)

#
# with open('JSB-Chorales-dataset-master/jsb-chorales-quarter.pkl','rb') as x:
#     data = pickle.load(x,encoding="latin1")
#
#
# x_train = data['train']
# y_train = data['train']
# x_test = data['test']
# y_test = data['valid']



#donne    pour   l'eintrainement
x_train = torch.randn(x_train)
y_train = torch.randn(y_train)
x_test = torch.randn(x_test)
y_test = torch.randn(y_test)

# x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(0)
# y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(0)
# x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(0)
#
# y_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(0)



print(x_train)
print(y_train)

# on  initiliase le model


input_size = 128
hidden_size = 32
num_layers = 4
output_size = 1

mozart = LSMTModel(input_size, hidden_size, num_layers, output_size)

perte = nn.MSELoss()
optimizer = optim.Adam(mozart.parameters(), lr=0.001)

num_epoch = 10
batch_size = 32

loss_values = []

train_loss_values = []
val_loss_values = []

for epoch in range(num_epoch) :
    epoch_train_loss = 0
    for i in range(0, len(x_train), batch_size) :
        x_batch = x_train[i :i + batch_size]
        y_batch = y_train[i :i + batch_size]

        outputs = mozart(x_batch)
        loss = perte(outputs, y_batch)
        # loss_values.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    train_loss_values.append(epoch_train_loss / (len(x_train) // batch_size))

    with torch.no_grad() :
        val_outputs = mozart(x_test)
        val_loss = perte(val_outputs, y_test)
        val_loss_values.append(val_loss.item())

    print(
        f'epoch[{epoch + 1}/{num_epoch}], Train Loss: {train_loss_values[-1]:.4f}, Validation Loss: {val_loss_values[-1]:.4f}')

# loss_dataframe = pd.DataFrame(loss_values, columns=['loss'])

perte_dataframe = pd.DataFrame({'Train Loss' : train_loss_values, 'Validation Loss' : val_loss_values})

# plt.figure()
# loss_dataframe.plot()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()

plt.figure()
perte_dataframe.plot()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss ')
plt.legend(['Train Loss', 'Validation Loss'])
plt.show()

print(f'Fin du test ')