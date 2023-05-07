import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import load
import torch.optim as optim
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available() :
    print("CUDA")
else :
    print("CPU")


class LSTMModel(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers, num_classes) :
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x) :
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1])

        return out


def predict_single_image(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.reshape(-1, sequence_length,input_size).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class =torch.max(output,1)

    return  predicted_class.item()


def plot_learning_curve(loss_values):
    plt.figure(figsize=(10,5))
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend(['Train Loss', 'and Test Loss'])
    plt.show()





input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100

train_data = datasets.MNIST(root='Data', download=True, train=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='Data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

lucie = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(lucie.parameters(), lr=0.001)
# #
# loss_values = []
# train_loss_values = []
# val_loss_values = []
# num_epoch = 5
# epoch_train_loss = 0
#
# for epoch in range(num_epoch) :
#     epoch_train_loss = 0
#     for i, (images, labels) in enumerate(train_loader) :
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#
#         # Forward
#         outputs = lucie(images)
#         loss = criterion(outputs, labels).to(device)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_train_loss += loss.item()
#
#     train_loss_values.append(epoch_train_loss / len(train_loader) // batch_size)
#
#     if (i + 1) % 100 == 0 :
#         print(f'Epoch [{epoch + 1}/{num_epoch}], Step [{i + 1}/{len(train_loader)}], Train loss: {train_loss_values[-1]:4f} Loss: {loss.item():.4f}')
#
#
# with torch.no_grad() :
#     correct = 0
#     total = 0
#
#     for images, labels in test_loader :
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
#         outputs = lucie(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print(f'Test Accurancy of the model on the test images: {100 * correct / total:.2f}%')
#
with open('mozart_lstm_model_1.pt','wb') as f:
    torch.save(lucie.state_dict(),f)
def appel_model():
    with open('mozart_lstm_model_1.pt','rb') as model_1:
        lucie.load_state_dict(load(model_1))

# perte_dataframe = pd.DataFrame({'Train loss' : train_loss_values, 'Validation loss': val_loss_values})
#
# plt.Figure()
# perte_dataframe.plot()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Test Loss')
# # plt.legend(['Train Loss','Vakidation Loss'])
# # plt.show()
#
# plot_learning_curve(loss)

image_tensor , label = test_data[0]
preduction = predict_single_image(lucie,image_tensor)

for image_tensor,label in range(10):
    predict_single_image(lucie, label)
    print(f'predict class : {preduction} actual label : {label}')
