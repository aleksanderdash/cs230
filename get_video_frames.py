import argparse
import cv2
import json
import os

from tqdm import tqdm

def create_parser():
    parser = argparse.ArgumentParser(description="Grab first frames of videos in given directory.")
    parser.add_argument("source_dir",
                        help="Directory of source video files.")
    parser.add_argument("destination_dir",
                        help="Directory to save destination video files.")
    return parser

def process_video(video_path, destination_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if not success:
        print("Could not read frame from video {}.".format(video_path))
    success = cv2.imwrite(destination_path, image)
    if not success:
        print("Could not save frame from video {} to path {}.".format(video_path, destination_path))

def main(args):
    files = os.listdir(args.source_dir)
    if not os.path.exists(args.destination_dir):
        os.makedirs(args.destination_dir)

    # First, process all the video files
    for video_filename in tqdm(files, desc="Converting videos"):
        if video_filename.endswith(".mp4"):
            video_basename = video_filename[:-4]
            image_filename = "{}.jpg".format(video_basename)
            video_path = os.path.join(args.source_dir, video_filename)
            image_path = os.path.join(args.destination_dir, image_filename)
            process_video(video_path, image_path)
    
    # Then, process metadata.json, which only exists
    # for the training data, not the test data
    metadata_source_path = os.path.join(args.source_dir, "metadata.json")
    if os.path.exists(metadata_source_path):
        with open(metadata_source_path) as f_in:
            metadata = json.load(f_in)
        image_metadata = {}
        for video_path in metadata:
            image_path = video_path.replace(".mp4", ".jpg")
            image_metadata[image_path] = metadata[video_path]
            del image_metadata[image_path]["original"]
        metadata_destination_path = os.path.join(args.destination_dir, "metadata.json")
        with open(metadata_destination_path, "w") as f_out:
            json.dump(image_metadata, f_out)
        print("Finished dumping metadata to {}.".format(metadata_destination_path))

if __name__=='__main__':
    main(create_parser().parse_args())
