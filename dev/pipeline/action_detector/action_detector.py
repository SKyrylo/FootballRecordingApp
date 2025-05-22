import torch
from movinets import MoViNet


class ActionDetector():
    """
    A class that handles action classification using MoViNet.
    """
    def __init__(self, model_path, model_name, device):
        """
        Initialize the action recognition model.
        
        Args:
            model_path (str): Path to the MoViNet model weights.
            model_name (str): Name of the model, e.g. 'a0' or 'a2'.
            device (str): Device to execute the model on, e.g. 'cuda' or 'cpu'.
        """
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        self.model = MoViNet.get_model(model_name=model_name, 
                                       pretrained=False, 
                                       head_activation=None).load_state_dict(state_dict).to(device).eval()
        
    def track_video(self):
        pass
