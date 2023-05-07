import torch
from lstm import lucie
from torchvision.datasets import MNIST
from torch import load

def predict_single_image(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.reshape(-1, sequence_length,input_size).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class =torch.max(output,1)

    return  predicted_class.item()
#
# with open('mozart_lstm_model_1.pt','rb') as model_1:
#     lucie.load_state_dict(load(model_1))


image_tensor, label = MNIST.test_data[0]


class_predict = predict_single_image(lucie,image_tensor)

print(f'class predict : {class_predict},Actual Label : {label}')



