import os
import glob

import torch
from datasets import load_dataset
from torchvision.transforms import ToPILImage, Compose
from torchvision import transforms

from main import classes
from torch.utils.data import DataLoader

from res_net import ResNet


def transform_image(entry):
    transformer = Compose(
        [
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    entry['image'] = [transformer(img) for img in entry['image']]
    return entry

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

model = ResNet(classes=5)
flower_ai = torch.load('./flower_ai.pth')
model.load_state_dict(flower_ai)
model.to(device)

with (torch.no_grad()):
    model.eval()

    incorrect_images = []
    correct = 0
    total = 0

    for data in test_loader:
        images = data['image'].to(device)
        labels = data['label'].to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        mask = (predicted != labels)
        incorrect_images.extend(
            [(img, pred, actual) for img, pred, actual in zip(images[mask], predicted[mask], labels[mask])])

    print(f'Accuracy on test images: {100 * correct // total} %')

    # Visualize the mistakes
    folder_path = "image_mistakes"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        files = glob.glob(f'{folder_path}/*')
        for file in files:
            os.remove(file)

    for index, (img, pred, actual) in enumerate(incorrect_images):
        pil_img = ToPILImage()(img.cpu())
        pil_img.save(os.path.join(folder_path, f"mistake_{index}.png"))
        print(f'Index: {index}, Predicted: {classes[pred.item()]}, Actual: {classes[actual.item()]}')
