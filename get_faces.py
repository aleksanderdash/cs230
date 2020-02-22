import numpy as np
import cv2
import os
import time

start_time = time.time()
video_path = "train_sample_videos/eggbjzxnmg.mp4"
#video_path = "train_sample_videos/aagfhgtpmv.mp4"
process_every_n_frames = 1
output_path = "faces"
min_bbox_size = (224, 224)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(video_path)
out_vid = cv2.VideoWriter("faces.mp4", cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 6, (1920, 1080))

last_bbox = np.array([])
mask = np.array([[1, 1], [0.5, 0.5]]) # to find center of bounding box
total_count = 0
processed_count = 0
while True:
    success, img = vid.read()
    if not success:
        break
    total_count += 1
    if total_count % process_every_n_frames != 0:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) == 0:
        if len(last_bbox) == 0:
            # Can only happen if the first frame of video doesn't detect a face.
            # Just use the center part of the image
            min_w, min_h = min_bbox_size
            last_bbox = np.array([(img.shape[0] - min_w) / 2, (img.shape[1] - min_h) / 2,
                min_w, min_h])
        cur_face = last_bbox
    else:
        if len(last_bbox) == 0:
            # Pick the most confident one
            faces, _, confidences = detector.detectMultiScale3(gray, 1.1, 5, minSize=(100, 100), outputRejectLevels=True)
            cur_face = faces[np.argmax(confidences)]
        else:
            # Take the face whose bbox center is closest to our last bbox
            last_center = np.sum(last_bbox.reshape(1, 2, 2) * mask, axis=1)
            next_face_idx = np.argmin(np.linalg.norm(np.sum(faces.reshape(-1, 2, 2) * mask, axis=1) - last_center, axis=1))
            cur_face = faces[next_face_idx]

    # Draw a box around the face we detected
    # Later we might want to crop the image to just this section.
    x, y, w, h = tuple(int(item) for item in cur_face)
    print(cur_face)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    img_out = os.path.join(output_path, "{:03d}.jpg".format(processed_count))
    #cv2.imwrite(img_out, img)
    out_vid.write(img)
    # PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    last_bbox = cur_face
    processed_count += 1

end_time = time.time()

print("Processed {} frames (out of {} total). Took {} seconds.".format(processed_count, total_count, end_time - start_time))
vid.release()
out_vid.release()
