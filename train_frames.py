import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models

from frame_dataset import FrameDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

num_epochs = 10
torch.set_num_threads(8)

train_loader = DataLoader(
    FrameDataset("train_sample_frames"),
    batch_size=8,
    shuffle=True
)
# Training is very slow currently since vgg16 is single-threaded on CPU??
# TODO: Add support for face extraction to dataset loader
vgg16 = torchvision.models.vgg16(pretrained=True)
for param in vgg16.features.parameters():
    param.requires_grad = False

# Just something really simple for now
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(512 * 120, 1024),
    nn.ReLU(), # TODO: Add BatchNorm
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

# TODO: Add weights to the real/fake labels in accordance with
# their prevalences.
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def get_accuracy():
    correct_preds = 0
    total_preds = 0
    for minibatch_data, minibatch_labels in tqdm(train_loader, desc="Evaluating accuracy"):
        output = model(vgg16.features(minibatch_data))
        total_preds += output.shape[0]
        correct_preds += torch.sum(torch.argmax(output, dim=1, keepdims=True) == minibatch_labels)
    return 1. * correct_preds / total_preds

print("Before training, accuracy is {}.".format(get_accuracy()))
for epoch in range(num_epochs):
    losses = []
    for minibatch_data, minibatch_labels in tqdm(train_loader, desc="Epoch {}".format(epoch)):
        if len(minibatch_labels) > 1:
            minibatch_labels = torch.squeeze(minibatch_labels, dim=1)
        optimizer.zero_grad()
        output = model(vgg16.features(minibatch_data))
        loss = loss_fn(output, minibatch_labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print("Epoch {}: Loss is {}".format(epoch, torch.mean(torch.tensor(losses))))
print("After training, accuracy is {}.".format(get_accuracy()))
