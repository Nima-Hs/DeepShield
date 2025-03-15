import torch
import torch.nn as nn
from transformers import Dinov2ForImageClassification, ResNetForImageClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from deepshield.dataset.vhd11k import VHD11K, get_transforms


batch_size = 32
num_epochs = 10
learning_rate = 1e-3

transforms = get_transforms()

dataset = VHD11K(
    data_root="/home/bahar/voxel_hackathon/dataset",
    transform=transforms,
)

# Split the dataset into training, validation and test sets
dataset_size = len(dataset)
train_fraction = 0.8
val_fraction = 0.1

train_size = int(train_fraction * dataset_size)
val_size = int(val_fraction * dataset_size)
test_size = dataset_size - train_size - val_size


train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get number of labels
num_labels = 2

# Load Pretrained DINOv2
# model = Dinov2ForImageClassification.from_pretrained(
#     "facebook/dinov2-small-imagenet1k-1-layer",
#     num_labels=num_labels,
#     ignore_mismatched_sizes=True,
# )
model = ResNetForImageClassification.from_pretrained(
    "microsoft/resnet-50",
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
)

# # Freeze all layers except classifier
# for param in model.parameters():
#     param.requires_grad = False  # Freeze backbone

# # Unfreeze only the classifier head
# for param in model.classifier.parameters():
#     param.requires_grad = True

# # Replace classifier with sigmoid for multi-label classification
# model.classifier = nn.Sequential(
#     nn.Linear(model.classifier.in_features, num_labels),
#     nn.Sigmoid()  # Sigmoid for multi-label classification
# )

model.to("cuda")
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.BCEWithLogitsLoss()

training_args = TrainingArguments(
    output_dir="./outputs",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2
)

# Train using Hugging Face Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()