import torch
from datasets import load_dataset

from le_net import LeNet
from main import get_labels, transform

dataset = load_dataset("DeadPixels/DPhi_Sprint_25_Flowers")
test_dataset = dataset['test']
test_dataset.set_transform(transform)

model = LeNet(len(get_labels(test_dataset)))
flower_ai = torch.load('./flower_ai.pth')
model.load_state_dict(flower_ai)
model.eval()

# Get the first image from the dataset
prediction = model(test_dataset[0]['image'])
print(prediction)
