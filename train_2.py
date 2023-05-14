import torch
from torch import nn


activation = nn.ReLU()
input = torch.randn(2)
input_2 = torch.randn(2).unsqueeze(0)
print(input)
output =activation(input)
print()
print()
output_2 = torch.cat((activation(input_2), activation(-input_2)))

print(output)
print()
print(output_2)