import torch
import torch.optim as optim
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.mobile_optimizer import optimize_for_mobile

from le_net import LeNet


def get_labels(dataset):
    labels = []
    for index in range(len(dataset)):
        label = dataset[index]['label']
        if label not in labels:
            labels.append(label)

    return labels


def transform(examples):
    transformer = Compose(
        [
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    examples['image'] = [transformer(img) for img in examples['image']]
    return examples


if __name__ == "__main__":
    # Load data set
    dataset = load_dataset("DeadPixels/DPhi_Sprint_25_Flowers")

    # Clean up the cache files
    # dataset.cleanup_cache_files()

    train_dataset = dataset['train']
    labels = get_labels(train_dataset)

    train_dataset.set_transform(transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=12,
    )

    le_net = LeNet(classes=len(labels))

    # Optimizer
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(le_net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(4):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image']
            label = data['label']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = le_net(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 99:.3f}')
                running_loss = 0.0

    torch.save(le_net.state_dict(), './flower_ai.pth')
    print('Finished Training')
