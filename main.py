import torch
import torch.optim as optim
from datasets import load_dataset
from torch import no_grad
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip
from torch.utils.mobile_optimizer import optimize_for_mobile

from res_net import ResNet

classes = {
    0: "daisy",
    1: "dandelion",
    2: "rose",
    3: "sunflower",
    4: "tulip",
}


class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return {'image': img, 'label': label}


def transform_image_from_extra_database(entry):
    transformer = Compose(
        [
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
        ]
    )
    entry['image'] = [transformer(img) for img in entry[0]]
    return entry


def transform_image(entry):
    transformer = Compose(
        [
            transforms.Resize((192, 192)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    entry['image'] = [transformer(img) for img in entry['image']]
    return entry


if __name__ == "__main__":
    # Set GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load data set
    dataset = load_dataset("DeadPixels/DPhi_Sprint_25_Flowers")

    # Extra dataset built
    # additional_dataset = CustomImageFolder(
    #     'flowers',
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize((192, 192)),
    #             transforms.ToTensor(),
    #         ]
    #     )
    # )

    train_dataset_from_huggingface = dataset['train']
    validation_dataset = dataset['validation']

    model = ResNet(classes=5)
    model.to(device)

    # Optimizer
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005)

    # Scheduler for learning rate reduction
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # Early stopping initialization
    min_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    # loop over the dataset multiple times
    for epoch in range(1000):

        train_dataset_from_huggingface.set_transform(transform_image)
        train_dataset = ConcatDataset([train_dataset_from_huggingface])
        # train_dataset = ConcatDataset([train_dataset_from_huggingface, additional_dataset])

        train_loader = DataLoader(
            train_dataset_from_huggingface,
            batch_size=3,
            shuffle=True,
            num_workers=12,
        )

        validation_dataset.set_transform(transform_image)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=3,
            shuffle=True,
            num_workers=12,
        )

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 99:.3f}')
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        correct = 0
        total = 0
        with no_grad():
            for data in validation_loader:
                images = data['image'].to(device)
                labels = data['label'].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(validation_loader)
        accuracy = 100 * correct / total
        print(f'Validation loss: {val_loss:.3f}, Accuracy: {accuracy:.2f}%')

        # Check for early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')

                break

        # Adjust learning rate
        scheduler.step(val_loss)

    print('Finished Training')

    # Save the model
    torch.save(model.state_dict(), './flower_ai.pth')

    # Optimize for mobile
    scripted_model = torch.jit.script(model)
    optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
    optimized_model._save_for_lite_interpreter('./flower_ai_optimized_lite.ptl')
    torch.jit.save(optimized_model, './flower_ai_optimized.pth')
