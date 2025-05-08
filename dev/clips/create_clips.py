import pandas as pd
import cv2
import os
import random

def create_clips_from_dataframe(dataframe: pd.DataFrame, video_directory: str, output_directory: str, num_clips_per_action: dict):
    """
    Generates video clips from source videos based on action annotations in a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame with columns:
                                  'VidInfo' (video file name, e.g., 'video1.mp4'),
                                  'Action' (folder name for output clips),
                                  'StartFrame' (0-indexed start frame of action),
                                  'EndFrame' (0-indexed end frame of action),
                                  'Len' (length of action in frames).
        video_directory (str): Path to the directory containing the source video files.
        output_directory (str): Path to the root directory where action folders and clips will be saved.
        num_clips_per_action (dict): The number of 100-frame clips to generate for each
                                    action instance in the DataFrame.
    """

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through each action instance in the DataFrame
    for index, row in dataframe.iterrows():
        video_filename = row['VidInfo']
        action_name = row['Action']
        action_start_frame = row['StartFrame']
        action_end_frame = row['EndFrame']
        action_length = row['Len'] # This should ideally be action_end_frame - action_start_frame + 1

        # Construct the full path to the source video
        video_path = os.path.join(video_directory, video_filename)

        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}. Skipping action at index {index}.")
            continue

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not cap.isOpened():
            print(f"Warning: Could not open video file: {video_path}. Skipping action at index {index}.")
            continue

        # Get video properties
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the required clip length
        clip_length_frames = 100

        # Check if the video is long enough to extract a 100-frame clip
        if total_video_frames < clip_length_frames:
            print(f"Warning: Video {video_filename} is shorter than {clip_length_frames} frames ({total_video_frames} frames). Cannot create 100-frame clips. Skipping action at index {index}.")
            cap.release()
            continue

        # Calculate the valid range for the start frame of the 100-frame clip
        # The clip_start_frame must satisfy:
        # 1. clip_start_frame <= action_start_frame (the clip must start at or before the action)
        # 2. clip_start_frame + clip_length_frames - 1 >= action_end_frame (the clip must end at or after the action)
        # 3. clip_start_frame >= 0 (the clip must not start before the beginning of the video)
        # 4. clip_start_frame + clip_length_frames - 1 < total_video_frames (the clip must not end after the end of the video)

        # From 1: clip_start_frame <= action_start_frame
        # From 2: clip_start_frame >= action_end_frame - clip_length_frames + 1
        # From 3: clip_start_frame >= 0
        # From 4: clip_start_frame <= total_video_frames - clip_length_frames

        # Combining these, the valid range for clip_start_frame is:
        valid_start_min = max(0, action_end_frame - clip_length_frames + 1)
        valid_start_max = min(action_start_frame, total_video_frames - clip_length_frames)

        # Check if the valid range is valid (min <= max)
        if valid_start_min > valid_start_max:
            print(f"Warning: Cannot create a {clip_length_frames}-frame clip containing action from frame {action_start_frame} to {action_end_frame} in video {video_filename} ({total_video_frames} frames). Range is invalid: [{valid_start_min}, {valid_start_max}]. Skipping action at index {index}.")
            cap.release()
            continue

        # Create the output directory for this action if it doesn't exist
        action_output_dir = os.path.join(output_directory, action_name)
        os.makedirs(action_output_dir, exist_ok=True)

        if action_name == 'none':
            output_clip_filename = f"{os.path.splitext(video_filename)[0]}_{action_name}_id{index:04}_{action_start_frame}-{action_end_frame}.mp4"
            output_clip_path = os.path.join(action_output_dir, output_clip_filename)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_clip_path, fourcc, fps, (640, 640))

            cap.set(cv2.CAP_PROP_POS_FRAMES, action_start_frame)
            frames_read_count = 0
            while frames_read_count < clip_length_frames:
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} from {video_path}. Stopping clip writing for {output_clip_filename}.")
                    break # Stop if frame reading fails
                resized_frame = cv2.resize(frame, (640, 640))

                names_to_rotate = ("vid2", "vid3", "vid4", "vid5")
                if video_filename.startswith(names_to_rotate):
                    resized_frame = cv2.rotate(resized_frame, cv2.ROTATE_180)
                out.write(resized_frame)
                frames_read_count += 1

            # Release the video writer for the current clip
            out.release()
            print(f"Generated clip: {output_clip_path} - {action_start_frame} - {action_length}")
        else:
            for clip_num in range(num_clips_per_action[f'{action_name}_clones']):
                # Randomly select a start frame for the 100-frame clip within the valid range
                clip_start_frame = random.randint(valid_start_min, valid_start_max)
                clip_end_frame = clip_start_frame + clip_length_frames - 1

                # Construct the output clip filename
                # Using index and clip_num to ensure unique filenames for multiple clips from the same row
                output_clip_filename = f"{os.path.splitext(video_filename)[0]}_{action_name}_id{index:04}_{clip_start_frame}-{clip_end_frame}_{clip_num}.mp4"
                output_clip_path = os.path.join(action_output_dir, output_clip_filename)

                # Define the video writer
                # Using 'mp4v' codec, you might need to change this based on your system and desired output format
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_clip_path, fourcc, fps, (640, 640))

                # Set the video capture position to the start of the clip
                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)

                # Read and write the 100 frames for the clip
                frames_read_count = 0
                while frames_read_count < clip_length_frames:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Warning: Could not read frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} from {video_path}. Stopping clip writing for {output_clip_filename}.")
                        break # Stop if frame reading fails
                    
                    resized_frame = cv2.resize(frame, (640, 640))
                    names_to_rotate = ("vid2", "vid3", "vid4", "vid5")
                    if video_filename.startswith(names_to_rotate):
                        resized_frame = cv2.rotate(resized_frame, cv2.ROTATE_180)

                    out.write(resized_frame)
                    frames_read_count += 1

                # Release the video writer for the current clip
                out.release()
                print(f"Generated clip: {output_clip_path} - {clip_start_frame} - {action_length}")

            # Release the video capture after processing all clips for this row
            cap.release()

# --- Example Usage ---

df = pd.read_csv("/home/kyrylo/dev/actionsLast.csv")
to_replace = {"vid1": "vid1.mp4",
              "vid2": "vid2.mp4",
              "vid3": "vid3.mp4",
              "vid4": "vid4.mp4",
              "vid5": "vid5.mp4",
              "yt1": "youtube_1.mp4",
              "yt2": "youtube_2.mp4",
              "yt3": "youtube_3.mp4",
              "yt5": "youtube_5.mp4"}


for key, val in to_replace.items():
    df = df.replace(key, val)


# Define your video directory (where video1.mp4, video2.mp4, etc. are located)
# Replace with the actual path to your video files
your_video_directory = "/home/kyrylo/bucket-72907_mount/football-files/full-videos/" # Example: './my_videos/'

# Define your desired output directory (where action folders and clips will be created)
# Replace with the actual path where you want to save the clips
your_output_directory = '/home/kyrylo/dev/clips/clips_dataset/' # Example: './generated_clips/'

# Define how many clips to generate for each action instance

clone_dict = {"pass_clones": 1,
              "goal_clones":8,
              "save_clones": 10,
              "tackle_clones": 7,
              "none_clones": 1}
# clips_per_action = 3 # Generate 3 different 100-frame clips for each row in the DataFrame

# Note: For this example to work, you need to have dummy video files named
# 'video1.mp4', 'video2.mp4', 'video3.mp4' in the 'your_video_directory'.
# These videos should be long enough (at least 100 frames) and ideally
# contain content around the specified StartFrame and EndFrame.

# Call the function to create the clips
create_clips_from_dataframe(df, your_video_directory, your_output_directory, clone_dict)

# print("Clip generation process finished.")

# To run the example:
# 1. Make sure you have pandas and opencv-python installed (`pip install pandas opencv-python`).
# 2. Create the directories specified by `your_video_directory` and `your_output_directory`.
# 3. Place some dummy video files (e.g., video1.mp4, video2.mp4, video3.mp4) in `your_video_directory`.
#    Ensure these videos are long enough (at least 100 frames).
# 4. Uncomment the line `create_clips_from_dataframe(...)` above.
# 5. Run the Python script. The clips will be saved in subdirectories within `your_output_directory`.
