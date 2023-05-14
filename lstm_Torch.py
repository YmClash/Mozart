import torch


#neuronne  LSTM  simple  en  torch
# Définition des entrées
x = torch.tensor([0.5, 0.8])

# Définition des poids et des biais
w = torch.tensor([0.2, 0.4])
b = torch.tensor(0.1)

# Calcul de la sortie du neurone
z = torch.dot(x, w) + b
y = torch.sigmoid(z)

# Affichage de la sortie
print(y)
