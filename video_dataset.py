"""
Usage:
Import this dataset by executing:
    from video_dataset import VideoDataset
Create the dataset by executing:
    train_dataset = VideoDataset("train_sample_videos")
The dataset can be passed to a DataLoader for training/eval:
    train_loader = torch.utils.data.DataLoader(train_dataset,
                     batch_size=8, shuffle=True)

Note that the dataset cannot load test directories, since the videos
do not have label metadata. These will have to be loaded manually
in case we wish to submit our model's guesses to the Kaggle leaderboards.
"""

import cv2
import os
import json
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, video_directory, split="train",
                 num_validation_videos=20, process_every_n_frames=6):
        assert(os.path.exists(video_directory))
        # Store metadata about which videos are real/fake
        self.basedir = video_directory
        with open(os.path.join(video_directory, "metadata.json")) as f_in:
            self.metadata = json.load(f_in)

        # Store the actual real and fake videos
        self.real_videos = []
        self.fake_videos = []
        for path in os.listdir(video_directory):
            if path.endswith(".mp4"):
                if self.metadata[path]['label'] == "FAKE":
                    self.fake_videos.append(path)
                else:
                    self.real_videos.append(path)

        # Train: leave 20 real and 20 fake videos for validation set
        if split == "train":
            self.real_videos = self.real_videos[:-num_validation_videos]
            self.fake_videos = self.fake_videos[:-num_validation_videos]
        else:
            self.real_videos = self.real_videos[-num_validation_videos:]
            self.fake_videos = self.fake_videos[-num_validation_videos:]

        self.process_every_n_frames = process_every_n_frames

        # This is the transform recommended by PyTorch docs at
        # https://pytorch.org/docs/stable/torchvision/models.html
        # The VGG16 model requires this transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Initialize the face detector
        #self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    def get_target_from_path(self, path):
        label = 1 if self.metadata[path]["label"] == "FAKE" else 0
        return torch.tensor([label])

    # This function runs through every self.process_every_n_frames
    # frames of video, applies the face detector to the frame,
    # crops the frame around the face and resizes it to (224, 224)
    # and stacks all the faces together in a torch tensor,
    # which is returned.
    # Dimension should be (300 / self.process_every_n_frames, 3, 224, 224)
    # (since all videos are 10s at 30fps, each video contains 300 frames)
    def get_cropped_faces(self, video_path):
        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        vid = cv2.VideoCapture(os.path.join(self.basedir, video_path))
        cropped_faces = []

        last_bbox = np.array([])
        mask = np.array([[1, 1], [0.5, 0.5]]) # to find center of bounding box
        count = 0
        while True:
            success, frame = vid.read()
            if not success:
                break
            count += 1
            if count % self.process_every_n_frames != 0:
                continue

            # Convert image to grayscale and detect faces on it
            # (the face detector was trained on grayscale images)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces, _, confidences = self.detector.detectMultiScale3(gray, 1.1, 5, outputRejectLevels=True)
            
            # Is this the first frame?
            if len(last_bbox) == 0:
                if len(faces) == 0:
                    # Yikes, no faces detected! Use center of frame for now
                    cur_face = np.array([
                        (frame.shape[0] - 224) / 2,
                        (frame.shape[1] - 224) / 2,
                        224,
                        224
                    ])
                else:
                    # Use the model's best guess
                    cur_face = faces[np.argmax(confidences)]
            else:
                # Not the first frame -- we want to keep the prediction
                # of the face location consistent with the previous frames
                if len(faces) == 0:
                    # Easy, just reuse the last bounding box
                    cur_face = last_bbox
                else:
                    # For each face, compute the L2 distance between the
                    # center of its bounding box and the center of the
                    # bounding box for the last frame.
                    # Divide this norm by the confidence of the model,
                    # and take the argmin to get the index of our new
                    # face bounding box for this frame
                    # Note: Does this work reasonably well? Who knows! :P
                    last_center = np.sum(last_bbox.reshape(1, 2, 2) * mask, axis=1)
                    face_bbox_l2_norms = np.linalg.norm(
                        np.sum(faces.reshape(-1, 2, 2) * mask, axis=1) - last_center,
                        axis=1
                    )
                    face_bbox_scores = face_bbox_l2_norms / np.squeeze(confidences, axis=1) # lower score is better
                    cur_face = faces[np.argmin(face_bbox_scores)]

            # Now we have the current bounding box!
            # Convert image to PIL to crop and resize more easily.
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            left_x, upper_y, width, height = cur_face
            pil_frame = pil_frame.crop((left_x, upper_y, left_x + width, upper_y + height))
            pil_frame = pil_frame.resize((299, 299), Image.LANCZOS)

            # Convert PIL image to torch tensor
            cropped_faces.append(torch.unsqueeze(self.transform(pil_frame), dim=0))
            #cropped_faces.append(torch.unsqueeze(transforms.ToTensor()(pil_frame), dim=0))
            last_bbox = cur_face
        return torch.cat(cropped_faces)

    def __getitem__(self, index):
        #x = Image.open(os.path.join(self.basedir, self.image_paths[index]))
        if index < len(self.real_videos):
            video_name = self.real_videos[index]
        else:
            video_name = self.fake_videos[index - len(self.real_videos)]
        x = self.get_cropped_faces(video_name)
        y = self.get_target_from_path(video_name)
        return x, y
    
    def __len__(self):
        return len(self.real_videos) + len(self.fake_videos)
