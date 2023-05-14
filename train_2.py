import torch
from torch import nn


activation = nn.ReLU()
input = torch.randn(2)
input_2 = torch.randn(2).unsqueeze(0)
print('tensor input')
print(input)
print('tensor 2')
print(input_2)
output =activation(input)
output_2 = torch.cat((activation(input_2), activation(-input_2)))
print()
print()

print('tensor apres activation RELU')
print(output)
print()
print('tensor2 apres activation ')
print(output_2)