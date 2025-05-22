import argparse
from utils import read_video, save_video
from tracker import PlayerBallTracker
from action_detector import ActionDetector
from team_assigner import TeamAssigner
from drawers import PlayerTracksDrawer, BallTracksDrawer

from configs import (OUTPUT_VIDEO_PATH, 
                      TRACKER_VERSION, 
                      ACTION_MODEL_VERSION, 
                      TRACKER_PATH, 
                      ACTION_MODEL_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Football Analysis App")
    parser.add_argument("input_video", type=str, help="Path to video to analize")
    parser.add_argument("--output_video", type=str, default=OUTPUT_VIDEO_PATH,
                        help="Path where the output video will be saved")
    parser.add_argument("--detection_model_version", type=str, default=TRACKER_VERSION, 
                        help="'n' for nano (smaller but faster), 'l' for large (slower but better)")
    parser.add_argument("--action_model_version", type=str, default=ACTION_MODEL_VERSION,
                        help="'A0' (smaller but faster) or 'A2' (slower but better)")
    
    
def main():
    args = parse_args()
    
    # Read video
    video_frames = read_video(args.input_video)
    
    # Initialize tracker
    object_tracker = PlayerBallTracker()
    
    # Initialize action model
    movinet = ActionDetector()
    
    # get object tracks
    player_tracks, ball_tracks = object_tracker.get_object_tracks(video_frames)
    
    # Assign teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_teams(video_frames, player_tracks)
    
    # Draw output
    # Initialize drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    
    # Draw object tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks, player_assignment)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    
    # Save video
    save_video(output_video_frames, args.output_video)
    
    
if __name__ == "__main__":
    main()
    