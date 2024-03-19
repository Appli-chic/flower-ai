import torch
from datasets import load_dataset

from le_net import LeNet
from main import get_labels, transform_image
from torch.utils.data import DataLoader

# Set GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("DeadPixels/DPhi_Sprint_25_Flowers")
test_dataset = dataset['test']
test_dataset.set_transform(transform_image)
test_loader = DataLoader(
    test_dataset,
    batch_size=3,
    shuffle=True,
    num_workers=12,
)

model = LeNet(len(get_labels(test_dataset)))
flower_ai = torch.load('./flower_ai.pth')
model.load_state_dict(flower_ai)
model.to(device)

with (torch.no_grad()):
    model.eval()

    correct = 0
    total = 0

    for data in test_loader:
        images = data['image'].to(device)
        labels = data['label'].to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct // total} %')

