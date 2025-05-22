from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd


class PlayerBallTracker:
    """
    A class that handles player and ball detection and tracking using YOLO and ByteTrack.

    This class combines YOLO object detection with ByteTrack tracking to maintain consistent
    player and ball identities across frames while processing detections in batches.
    """
    def __init__(self, model_path):
        """
        Initialize the PlayerTracker with YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def track_video(self, input_video_path: str, output_path: str):
        """
        Track objects with YOLO default track method
        
        Args:
            input_video_path (str): Path to the video to be tracked
            output_path (str): Path to where the tracked video is saved
        
        Returns:
            Nothing. Saves the video with tracks to output_path
        """
        self.model.track(source=input_video_path, save=True, project=output_path)
    
    def detect_frames(self, frames: list[np.ndarray], batch_size: int=20):
        """
        Detect players in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.
            batch_size (int): The amount of frames to be processed simultaneously

        Returns:
            list: YOLO detection results for each frame.
        """
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, batch_size: int=20):
        """
        Get player and ball tracking results for a sequence of frames.

        Args:
            frames (list): List of video frames to process.
            batch_size (int): The amount of frames to be processed simultaneously

        Returns:
            list 1: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
            list 2: List of dictionaries containing ball tracking information for each frame.
        """
        detections = self.detect_frames(frames, batch_size)

        player_tracks=[]
        
        # Player tracks
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            player_tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['person']:
                    player_tracks[frame_num][track_id] = {"bbox":bbox}
        
        # Ball detections
        ball_tracks = []
        ball_tracks.append({})
        chosen_bbox =None
        max_confidence = 0
        
        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            confidence = frame_detection[2]
            
            if cls_id == cls_names_inv['ball']:
                if max_confidence<confidence:
                    chosen_bbox = bbox
                    max_confidence = confidence

        if chosen_bbox is not None:
            ball_tracks[frame_num][1] = {"bbox":chosen_bbox}
        
        return player_tracks, ball_tracks
        
    def interpolate_ball_tracks(self, ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking results.

        Args:
            ball_positions (list): List of ball positions with potential gaps.

        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
