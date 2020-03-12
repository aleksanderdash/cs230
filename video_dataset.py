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
                 num_validation_videos=240, num_test_videos=240, process_every_n_frames=30):
        assert(os.path.exists(video_directory))

        self.basedir = video_directory
        # Recursively walk through self.basedir and all subdirectories
        all_metadata_filenames = []
        all_video_filenames = []
        for dirpath, _, filenames in os.walk(self.basedir):
            for filename in filenames:
                if filename.endswith(".json"):
                    all_metadata_filenames.append(os.path.join(dirpath, filename))
                elif filename.endswith(".mp4"):
                    all_video_filenames.append(os.path.join(dirpath, filename))

        # Store metadata about which videos are real/fake
        self.metadata = {}
        for cur_metadata_filename in all_metadata_filenames:
            if split == "debug":
                print("Processing metadata in file {}".format(cur_metadata_filename))
            with open(cur_metadata_filename) as f_in:
                cur_metadata = json.load(f_in)
                for cur_key in cur_metadata:
                    # Store expanded versions of paths of videos in metadata
                    full_filename = os.path.join(os.path.dirname(cur_metadata_filename), cur_key)
                    self.metadata[full_filename] = {'label': cur_metadata[cur_key]['label']}
                    if 'original' in cur_metadata[cur_key] and cur_metadata[cur_key]['original']:
                        full_orig_filename = os.path.join(os.path.dirname(cur_metadata_filename), cur_metadata[cur_key]['original'])
                        self.metadata[full_filename]['original'] = full_orig_filename

        # Store the actual real and fake videos' full pathnames
        self.real_videos = []
        self.fake_videos = []
        for cur_video_filename in all_video_filenames:
            if self.metadata[cur_video_filename]['label'] == "FAKE":
                self.fake_videos.append(cur_video_filename)
            else:
                self.real_videos.append(cur_video_filename)
        # Sort the pathnames since consecutive runs of os.walk aren't guaranteed
        # to return all the data in the same order
        self.real_videos.sort()
        self.fake_videos.sort()

        # Train: leave 20 real and 20 fake videos for validation set
        if split == "debug":
            print("DEBUG: There are {} real videos and {} fake videos.".format(len(self.real_videos), len(self.fake_videos)))
        elif split == "train":
            # Hardcoded to 2400 samples of each class for now
            self.real_videos = self.real_videos[:2400]
            self.fake_videos = self.fake_videos[:2400]
            #self.real_videos = self.real_videos[:-(num_validation_videos + num_test_videos)]
            #self.fake_videos = self.fake_videos[:-(num_validation_videos + num_test_videos)]
            # Make sure we only keep same number of real and fake videos
            #self.fake_videos = self.fake_videos[:len(self.real_videos)]
        elif split == "validation":
            self.real_videos = self.real_videos[-(num_validation_videos + num_test_videos):-num_test_videos]
            self.fake_videos = self.fake_videos[-(num_validation_videos + num_test_videos):-num_test_videos]
        elif split == "test":
            self.real_videos = self.real_videos[-num_test_videos:]
            self.fake_videos = self.fake_videos[-num_test_videos:]
        else:
            raise ValueError("Invalid split: must be debug, train, validation, or test")

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
        self.memoized = {}
    
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
        vid = cv2.VideoCapture(video_path)
        cropped_faces = []

        last_bbox = np.array([])
        mask = np.array([[1, 1], [0.5, 0.5]]) # to find center of bounding box
        count = 0
        while True:
            success, frame = vid.read()
            if not success:
                break

            # If we have all the frames we need, stop!
            if len(cropped_faces) == 300 / self.process_every_n_frames:
                break
            # Change the order here since sometimes videos have 299 frames instead of 300
            if count % self.process_every_n_frames != 0:
                count += 1
                continue
            count += 1

            # Convert image to grayscale and detect faces on it
            # (the face detector was trained on grayscale images)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces, _, confidences = self.detector.detectMultiScale3(gray, 1.1, 5, outputRejectLevels=True)
            
            # Is this the first frame?
            if len(last_bbox) == 0:
                if len(faces) == 0:
                    # Yikes, no faces detected! Use center of frame for now
                    cur_face = np.array([
                        (frame.shape[0] - 299) / 2,
                        (frame.shape[1] - 299) / 2,
                        299,
                        299
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
        vid.release()
        return torch.cat(cropped_faces)

    def __getitem__(self, index):
        #x = Image.open(os.path.join(self.basedir, self.image_paths[index]))
        if index < len(self.real_videos):
            video_name = self.real_videos[index]
        else:
            video_name = self.fake_videos[index - len(self.real_videos)]
        y = self.get_target_from_path(video_name)
        if video_name in self.memoized:
            return self.memoized[video_name], y
        x = self.get_cropped_faces(video_name)
        if (x.shape != (10, 3, 299, 299)):
            print("ERROR: item at index {} (path {}) returned shape {}!".format(index, video_name, x.shape))
            if len(x.shape) == 3:
                x = torch.cat([x.unsqueeze(0) for _ in range(10)])
            if x.shape[0] != 10:
                x = torch.cat([x for _ in range(10)])[:10]
        self.memoized[video_name] = x
        return x, y
    
    def __len__(self):
        return len(self.real_videos) + len(self.fake_videos)
