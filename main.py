import torch
from datasets import load_dataset
import pandas as pd


def get_labels():
    labels = []
    for i in range(len(train_dataset)):
        label = train_dataset[i]['label']
        if label not in labels:
            labels.append(label)

    return labels


# Load data set
dataset = load_dataset("DeadPixels/DPhi_Sprint_25_Flowers")

# Clean up the cache files
# dataset.cleanup_cache_files()

train_dataset = dataset['train']

dataframe = train_dataset.to_pandas()
print(train_dataset[1])

labels = get_labels()
print(labels)
