import torch
import torch.optim as optim
from datasets import load_dataset
from torch import no_grad
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip
from torch.utils.mobile_optimizer import optimize_for_mobile

from le_net import LeNet


def get_labels(dataset):
    labels = []
    for index in range(len(dataset)):
        label = dataset[index]['label']
        if label not in labels:
            labels.append(label)

    return labels


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

    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    labels = get_labels(train_dataset)

    train_dataset.set_transform(transform_image)
    train_loader = DataLoader(
        train_dataset,
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

    le_net = LeNet(classes=len(labels))
    le_net.to(device)

    # Optimizer
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(le_net.parameters(), lr=0.001, momentum=0.9)

    # Scheduler for learning rate reduction
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # Early stopping initialization
    min_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    # loop over the dataset multiple times
    for epoch in range(1000):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = le_net(inputs)
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
                outputs = le_net(images)
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
    torch.save(le_net.state_dict(), './flower_ai.pth')

    # Optimize for mobile
    scripted_model = torch.jit.script(le_net)
    optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
    optimized_model._save_for_lite_interpreter('./flower_ai_optimized_lite.ptl')
    torch.jit.save(optimized_model, './flower_ai_optimized.pth')
