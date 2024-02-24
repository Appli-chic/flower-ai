import torch.optim as optim
import torchvision.transforms as transforms
from datasets import load_dataset
from numpy import array
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from le_net import LeNet


def get_labels():
    labels = []
    for i in range(len(train_dataset)):
        label = train_dataset[i]['label']
        if label not in labels:
            labels.append(label)

    return labels


if __name__ == "__main__":
    # Load data set
    dataset = load_dataset("DeadPixels/DPhi_Sprint_25_Flowers")

    # Clean up the cache files
    # dataset.cleanup_cache_files()

    train_dataset = dataset['test']
    labels = get_labels()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = train_dataset.map(
        lambda x: {
            "image": transform(x["image"]), "label": x["label"]
        }
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
    )

    le_net = LeNet(classes=len(labels))

    # Optimizer
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(le_net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

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
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
