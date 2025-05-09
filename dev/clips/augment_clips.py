import cv2
import os
import random
import numpy as np


def flip_frame(frame):
    flipped_frame = cv2.flip(frame, 1)
    return flipped_frame


def rotate_frame(frame, angle):
    height, width = frame.shape[:2]
    center_x, center_y = (width // 2, height // 2)

    # get rotation matrix
    m = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # rotate frame
    rotated_frame = cv2.warpAffine(frame, m, (width, height))

    return rotated_frame


def saturate_frame(frame, multiplier):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * multiplier
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return bgr_image


def noise_frame(frame, percent):
    row, col, _ = frame.shape
    num_noise_pixels = int(percent * row * col)
    for _ in range(num_noise_pixels):
        x, y = random.randint(0, row - 1), random.randint(0, col - 1)
        frame[x, y] = np.random.randint(0, 256, 3)
    return frame


def augment_video(input_path, output_path, aug):
    # Open the original video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec for .mp4

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the beginning of the video

    # set parameters of augmentation
    random_angle = random.randint(-15, 15)
    saturation_multiplier = random.uniform(0.8, 1.2)
    noise_percentage = random.uniform(0, 0.0156)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the specific augmentation to the frame
        if aug == 'flip':
            augmented_frame = flip_frame(frame)
        elif aug == 'rotate':
            augmented_frame = rotate_frame(frame, random_angle)
        elif aug == 'saturation':
            augmented_frame = saturate_frame(frame, saturation_multiplier)
        elif aug == 'noise':
            augmented_frame = noise_frame(frame, noise_percentage)
        else:
            print(f"\tNon existent augmentation for {os.path.split(input_path)[1]}")
            break

        # Write the augmented frame to the output video
        out.write(augmented_frame)

    # Release the VideoWriter for this augmented version
    out.release()
    print(f"Saved video with {aug} augmentation to {output_path}")

    # Release the video capture
    cap.release()


def process_videos(from_dir, to_dir):
    # Ensure output directory exists
    os.makedirs(to_dir, exist_ok=True)

    subdirs = ['goal', 'save', 'pass', 'tackle', 'none']

    for name in subdirs:
        dir = os.path.join(to_dir, name)
        os.makedirs(dir, exist_ok=True)

    augmentations = ['flip', 'rotate', 'saturation', 'noise']

    for name in subdirs:
        input_dir = os.path.join(from_dir, name)
        output_dir = os.path.join(to_dir, name)
        for filename in os.listdir(input_dir):
            if filename.endswith(".mp4"):
                for aug in augmentations:
                    input_video_path = os.path.join(input_dir, filename)
                    new_name = f"{filename.split('.')[0]}_{aug}.mp4"
                    output_video_path = os.path.join(output_dir, new_name)

                    augment_video(input_video_path, output_video_path, aug)



if __name__ == '__main__':
    output_dir = r'/home/kyrylo/dev/clips/clips_dataset'
    input_dir = r'/home/kyrylo/dev/clips/clips_dataset'

    process_videos(input_dir, output_dir)

