import os
import pickle
import time
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models

from video_dataset import VideoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def create_parser():
    parser = argparse.ArgumentParser(description="Train NN model on video data.")
    parser.add_argument("data_dir",
                        help="Directory of source video files.")
    parser.add_argument("--learning_rate", default=0.0005)
    parser.add_argument("--num_epochs", default=20)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--preprocess", action="store_true",
                        help="Preprocess the dataset given by data_dir to speed up training.")
    return parser

def main(args):
    if args.preprocess:
        preprocess_dataset(args)

    train_tensor_file = os.path.join(args.data_dir, "train_tensors.pkl")
    assert os.path.exists(train_tensor_file), ("Training tensors not found! "
        "Be sure to run this script first with the --preprocess flag.")
    validation_tensor_file = os.path.join(args.data_dir, "validation_tensors.pkl")
    with open(train_tensor_file, "rb") as f_in:
        train_tensors = pickle.load(f_in)
    with open(validation_tensor_file, "rb") as f_in:
        validation_tensors = pickle.load(f_in)
    print("Loaded {} training examples and {} validation examples.".format(
        len(train_tensors), len(validation_tensors)))
    
    n_channels, filter_size = 512, 7
    def he_initialized_linear(in_channels, out_channels):
        layer = nn.Linear(in_channels, out_channels)
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        return layer
    # Just something really simple for now
    model = nn.Sequential(
        nn.Flatten(),
        he_initialized_linear(n_channels * filter_size * filter_size, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        he_initialized_linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        he_initialized_linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 2)
    )
    fake_loss_weight = 1./6.5 # approx. 4/5 of the dataset are fake videos
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., fake_loss_weight]))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    def get_accuracy():
        real_preds = 0
        fake_preds = 0
        correct_preds = 0
        total_preds = 0
        for x, y in validation_tensors:
            output = nn.functional.log_softmax(model(x), dim=1) # output is (50, 2)
            total_preds += 1
            cur_prediction = 1 if torch.mean(torch.argmax(output, dim=1).double()) > 0.5 else 0
            print(torch.mean(torch.argmax(output, dim=1).double()).item(), end=', ')
            if cur_prediction == 0:
                real_preds += 1
            else:
                fake_preds += 1
            if cur_prediction == y.item():
                correct_preds += 1
        print ("{} real predictions and {} fake predictions.".format(real_preds, fake_preds))
        return 1. * correct_preds / total_preds

    for epoch in range(args.num_epochs):
        # Make a pass through the training set
        train_perm = np.random.permutation(len(train_tensors))
        cur_loss = 0.
        for cur_batch_start in tqdm(range(0, len(train_tensors), args.batch_size), desc="Epoch {}".format(epoch)):
            cur_minibatch_x = torch.cat([torch.unsqueeze(train_tensors[idx][0], dim=0)
                for idx in train_perm[cur_batch_start:cur_batch_start + args.batch_size]])
            cur_minibatch_y = torch.cat([torch.unsqueeze(train_tensors[idx][1], dim=0)
                for idx in train_perm[cur_batch_start:cur_batch_start + args.batch_size]])
            m, n_frames, _, _, _ = cur_minibatch_x.shape
            cur_minibatch_x = cur_minibatch_x.reshape((-1, n_channels, filter_size, filter_size))

            optimizer.zero_grad()
            output = model(cur_minibatch_x)
            cur_minibatch_y = cur_minibatch_y.repeat_interleave(n_frames)
            loss = loss_fn(output, cur_minibatch_y)
            loss.backward()
            optimizer.step()
            cur_loss += loss.item() * m
        print("Epoch {}: avg. loss is {}. Accuracy is {}.".format(epoch, cur_loss / len(train_tensors), get_accuracy()))
    torch.save(model, "model.pkl")
        #prev_time = time.time()
        #for minibatch_data, minibatch_labels in tqdm(train_loader):
            #cur_time = time.time()
            #print("Time taken: {} seconds, data = ({}, {})".format(cur_time - prev_time, minibatch_data.shape, minibatch_labels))
            #prev_time = cur_time
            #pass
    """
    for i in range(4, len(dataset)):
        x, y = dataset[i]
        import torchvision.transforms as transforms
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        img_transform = transforms.ToPILImage()
        fig = plt.figure()
        img_ani = []
        for img in x:
            pil_img = img_transform(img)
            img_ani.append([plt.imshow(pil_img, animated=True)])
        ani = animation.ArtistAnimation(fig, img_ani, interval=200, blit=True,
                                        repeat_delay=1000)
        fig.show()
        if input() == "":
            break
    """

def preprocess_dataset(args):
    frames_per_example = 300 / 6 # should be 50 frames per example but some have 49 :(
    train_dataset = VideoDataset(args.data_dir, split="train")
    validation_dataset = VideoDataset(args.data_dir, split="validation")
    train_outfile = os.path.join(args.data_dir, "train_tensors.pkl")
    train_examples = []
    for i in tqdm(range(len(train_dataset)), desc="Preprocessing training set"):
        x, y = train_dataset[i]
        while x.shape[0] < frames_per_example:
            x = torch.cat((x, torch.unsqueeze(x[-1], dim=0)))
        train_examples.append((x, y))
    with open(train_outfile, "wb") as f_out:
        pickle.dump(train_examples, f_out)
    print("Successfully dumped preprocessed training tensors to {}.".format(train_outfile))

    validation_outfile = os.path.join(args.data_dir, "validation_tensors.pkl")
    validation_examples = []
    for i in tqdm(range(len(validation_dataset)), desc="Preprocessing validation set"):
        x, y = validation_dataset[i]
        while x.shape[0] < frames_per_example:
            x = torch.cat((x, torch.unsqueeze(x[-1], dim=0)))
        validation_examples.append((x, y))
    with open(validation_outfile, "wb") as f_out:
        pickle.dump(validation_examples, f_out)
    print("Successfully dumped preprocessed validation tensors to {}.".format(validation_outfile))

def foo():
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

if __name__=='__main__':
    main(create_parser().parse_args())