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
from model import DeepPepegaNet

def create_parser():
    parser = argparse.ArgumentParser(description="Train NN model on video data.")
    parser.add_argument("data_dir",
                        help="Directory of source video files.")
    parser.add_argument("--learning_rate", default=0.0005, type=float)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_dataloaders", default=2, type=int)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--load_from_checkpoint",
                        help="Path to model checkpoint")
    parser.add_argument("--evaluate_checkpoints", action="store_true",
                        help="Evaluate model checkpoints in checkpoint_dir on validation set.")
    return parser

def main(args):
    if args.evaluate_checkpoints:
        evaluate_checkpoints(args.checkpoint_dir, args.data_dir, args.batch_size)
        return

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Ensure model checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print("Model checkpoints will be saved to local directory {}".format(args.checkpoint_dir))

    train_loader = DataLoader(
        VideoDataset(args.data_dir, split="train"),
        batch_size=args.batch_size,
        shuffle=True#,
        #num_workers=args.num_dataloaders
    )
    validation_loader = DataLoader(
        VideoDataset(args.data_dir, split="validation"),
        batch_size=args.batch_size,
        shuffle=True#,
        #num_workers=args.num_dataloaders
    )
    model = DeepPepegaNet().to(device)
    if args.load_from_checkpoint:
        print("Loading saved model checkpoint from {}".format(args.load_from_checkpoint))
        model.load_state_dict(torch.load(args.load_from_checkpoint))
    model.train()
    #fake_loss_weight = 1./6.5 # approx. 4/5 of the dataset are fake videos
    #loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., fake_loss_weight]))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    i = 0
    for epoch in range(args.num_epochs):
        losses = []
        with tqdm(train_loader, desc="Epoch {}".format(epoch)) as pbar:
            cur_progress = {"status": "Loading data"}
            pbar.set_postfix(cur_progress)
            for minibatch_data, minibatch_labels in pbar:
                cur_progress["status"] = "Training model"
                pbar.set_postfix(cur_progress)
                m, n_frames, _, _, _ = minibatch_data.shape
                if m == 1:
                    # Inception model crashes if batch size is 1
                    # so if we're at the end of an epoch, just keep going
                    continue
                if len(minibatch_labels) > 1:
                    minibatch_labels = torch.squeeze(minibatch_labels, dim=1)
                optimizer.zero_grad()
                output = model(minibatch_data.to(device))
                loss = loss_fn(output, minibatch_labels.to(device))
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().item())
                cur_progress["loss"] = losses[-1]
                i += 1
                if i % 200 == 0:
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_epoch_{}_iter_{}.pth".format(epoch, i)))
                    print("Epoch {}, iter {}: Loss is {}".format(epoch, i, torch.mean(torch.tensor(losses))))
                cur_progress["status"] = "Loading data"
                pbar.set_postfix(cur_progress)
            cur_progress["status"] = "Done"
            cur_progress["loss"] = torch.mean(torch.tensor(losses))
            pbar.set_postfix(cur_progress)
        print("Epoch {}: Loss is {}".format(epoch, torch.mean(torch.tensor(losses))))
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_after_training_epoch_{}_iter_{}.pth".format(args.num_epochs, i)))
    model.eval()
    print("After training, validation accuracy is {}.".format(get_accuracy(model, validation_loader, device)))
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

def get_accuracy(model, validation_loader, device):
    # There are 120 real and 120 fake videos in the dataset
    real_preds = 0
    fake_preds = 0
    correct_preds = 0
    correct_real_preds = 0
    correct_fake_preds = 0
    total_preds = 0
    for x, y in tqdm(validation_loader, desc="Evaluating model"):
        output = nn.functional.log_softmax(model(x.to(device)), dim=1) # output is (batch_size, 2)
        total_preds += x.shape[0]
        cur_predictions = torch.argmax(output, dim=1).cpu()
        real_preds += torch.sum(cur_predictions == 0)
        fake_preds += torch.sum(cur_predictions == 1)
        correct_real_preds += torch.sum((cur_predictions == y.squeeze()) == (cur_predictions == 0)).item()
        correct_fake_preds += torch.sum((cur_predictions == y.squeeze()) == (cur_predictions == 1)).item()
        correct_preds += torch.sum(cur_predictions == y.squeeze()).item()
    print("{} real predictions and {} fake predictions.".format(real_preds, fake_preds))
    print("For real videos, precision: {}, recall: {}".format(correct_real_preds / real_preds, correct_real_preds / 120.))
    print("For fake videos, precision: {}, recall: {}".format(correct_fake_preds / fake_preds, correct_fake_preds / 120.))
    return 1. * correct_preds / total_preds

def evaluate_checkpoints(checkpoint_dir, data_dir, batch_size):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DeepPepegaNet().to(device)
    validation_loader = DataLoader(
        VideoDataset(data_dir, split="validation"),
        batch_size=batch_size,
        shuffle=True
    )
    checkpoint_files = [filename for filename in os.listdir(checkpoint_dir) if filename.endswith(".pth")]
    for checkpoint_filename in checkpoint_files:
        print("Evaluating checkpoint file {}".format(checkpoint_filename))
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_filename)))
        model.eval()
        accuracy = get_accuracy(model, validation_loader, device)
        print("Model accuracy is {}".format(accuracy))
        print("")
    print("All done!")

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

    def get_accuracy_old():
        correct_preds = 0
        total_preds = 0
        for minibatch_data, minibatch_labels in tqdm(train_loader, desc="Evaluating accuracy"):
            output = model(vgg16.features(minibatch_data))
            total_preds += output.shape[0]
            correct_preds += torch.sum(torch.argmax(output, dim=1, keepdims=True) == minibatch_labels)
        return 1. * correct_preds / total_preds

    print("Before training, accuracy is {}.".format(get_accuracy_old()))
    for epoch in range(num_epochs):
        losses = []
        i = 0
        for minibatch_data, minibatch_labels in tqdm(train_loader, desc="Epoch {}".format(epoch)):
            if len(minibatch_labels) > 1:
                minibatch_labels = torch.squeeze(minibatch_labels, dim=1)
            optimizer.zero_grad()
            output = model(vgg16.features(minibatch_data))
            loss = loss_fn(output, minibatch_labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            i += 1
            if i % 1000 == 0:
                torch.save(model.state_dict(), "model_epoch_{}_iter_{}".format(epoch, i))
        print("Epoch {}: Loss is {}".format(epoch, torch.mean(torch.tensor(losses))))
    print("After training, accuracy is {}.".format(get_accuracy_old()))

if __name__=='__main__':
    main(create_parser().parse_args())
