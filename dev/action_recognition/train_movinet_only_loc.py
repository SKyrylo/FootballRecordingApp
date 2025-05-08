import torch
import time
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
# import transforms as T
from movinets import MoViNet
from movinets.config import _C
from movinets.models import *
import sys
import json
import os
import pickle
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np

# import torch
from torch.utils.data import Dataset, DataLoader

class MoViNetWithLocalization(MoViNet):
    
    def __init__(self, architecture, causal, pretrained):
        super(MoViNetWithLocalization, self).__init__(architecture, causal, pretrained)
        
#         self.backbone = backbone
#         self.features = self.extract_features(backbone, layer_name)
        
        # Classifier head
        self.classifier_new = nn.Sequential(
            ConvBlock3D(
                640, 2048, kernel_size=(1, 1, 1), tf_like=False, causal=False, conv_type="3d", stride=(1, 1, 1), bias=True
            ),
            Swish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Conv3d(2048, 9, (1, 1, 1))
        )

        # Localization head
        self.localizer_new = nn.Sequential(
            # Define the layers for the new output branch
            ConvBlock3D(
                640, 2048, kernel_size=(1, 1, 1), tf_like=False, causal=False, conv_type="3d", stride=(1, 1, 1), bias=True
            ),
            Swish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Conv3d(2048, 2, (1, 1, 1))
        )
    
    def extract_features(self, md, layer_name):
        layers = []
        for name, module in md.named_children():
            layers.append(module)
            if name == layer_name:
                break
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        x = self.avg(x)
        x_cls = self.classifier_new(x)
        x_loc = self.localizer_new(x)
        x_cls = x_cls.flatten(1)
        x_loc = x_loc.flatten(1)
        return x_cls, x_loc

class KickPunchDataset(Dataset):
    def __init__(self, data, labels, n_frames, transform=None):
        self.data = [(x, y) for x, y in zip(data, labels)]
#         self.labels = labels
        self.n_frames = n_frames
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def format_frames(self, frame):
        if self.transform:
            frame = self.transform(frame)
        return frame
    
    def load_video(self, video_path):
        # Read each video frame by frame
        result = []
#         print(video_path)
        # video_path_splits = video_path.split("/")
        # if video_path_splits[0] == "Prepared_videos":
        #     video_path_splits[0] = video_path_splits[0] + "-2"
        # video_path = os.path.join(*video_path_splits)
        src = cv2.VideoCapture(str(video_path))
        
        # video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # need_length = 1 + (n_frames - 1) * frame_step
        
        # if need_length > video_length:
        #   start = 0
        # else:
        #   max_start = video_length - need_length
        #   start = random.randint(0, max_start + 1)
        
        # src.set(cv2.CAP_PROP_POS_FRAMES, start)
        # ret is a boolean indicating whether read was successful, frame is the image itself
        # ret, frame = src.read()
        # result.append(format_frames(frame, output_size))
        
        for _ in range(self.n_frames):
            ret, frame = src.read()
            if ret:
                frame = self.format_frames(frame)#, output_size)
                result.append(frame)
            else:
                result.append(np.zeros_like(result[0]))
        src.release()
        result = np.array(result)#[..., [2, 1, 0]]
        
        return result
    
    def __getitem__(self, idx):
        res = self.load_video(self.data[idx][0])
        # label = torch.nn.functional.one_hot(self.labels[idx], 5)
        sample = {
            'data': res,
            'label_cls': self.data[idx][1][0],
            'label_loc': torch.Tensor(self.data[idx][1][1:])
        }
        return sample

# Training functions

def calc_acc(y_true, y_pred):
    n_correct = torch.sum(y_true == y_pred)
    return n_correct/y_true.size(0)

def calc_tiou(y_true, y_pred):
    inter_left = torch.max(torch.column_stack([y_true[:, 0], y_pred[:, 0]]), dim=1).values # max of begginings
    inter_right = torch.min(torch.column_stack([y_true[:, 1], y_pred[:, 1]]), dim=1).values # min of ends
    union_left = torch.min(torch.column_stack([y_true[:, 0], y_pred[:, 0]]), dim=1).values # min of begginings
    union_right = torch.max(torch.column_stack([y_true[:, 1], y_pred[:, 1]]), dim=1).values # max of ends
    intersection = torch.clamp(inter_right - inter_left, min=0)
    union = (union_right - union_left) - torch.clamp(-(inter_right - inter_left), min=0)
    tiou = intersection / (union + 1e-6)
    return torch.sum(tiou) / y_true.size(0)

def train_iter_stream(model, optimz, data_load, loss_val, epoch, loss_fn=None, class_weight=None, scheduler=None, n_clips = 2, n_clip_frames=8):
    """
    In causal mode with stream buffer a single video is fed to the network
    using subclips of lenght n_clip_frames. 
    n_clips*n_clip_frames should be equal to the total number of frames presents
    in the video.
    
    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    #clean the buffer of activations
    print("------------------------")
    print(f"EPOCH {epoch}")
    epoch_loss, epoch_acc = 0, 0
    samples = len(data_load.dataset)
    model.cuda()
    model.train()
    model.clean_activation_buffers()
    optimz.zero_grad()
    # pbar = tqdm(enumerate(data_load))
    for i, sample in enumerate(data_load):# pbar:
        res = sample["data"].permute(0, 2, 1, 3, 4)
        res = res.cuda()
        target = sample["label"].cuda()
        l_batch, acc_batch = 0, 0
        #backward pass for each clip
        for j in range(n_clips):
          if loss_fn is None:
            output = F.log_softmax(model(res[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]), dim=1)
            loss = F.nll_loss(output, target, weight=class_weight)
          else:
            output = F.softmax(model(res[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]), dim=1)
#             target = F.one_hot(target, num_classes=5)
#             target = target.float()
            loss = loss_fn(output, target)
          _, pred = torch.max(output, dim=1)
          if loss_fn is None:
            loss = F.nll_loss(output, target)/n_clips
          else:
            loss = loss_fn(output, target)/n_clips
          loss.backward()
          accuracy = pred.eq(target).sum() # calc_acc(target.cuda(), pred.cuda())
          acc_batch += accuracy.item()
        l_batch += loss.item()*n_clips
        acc_batch = acc_batch / n_clips
        optimz.step()
        optimz.zero_grad()
#         if not scheduler is None:
#             scheduler.step()
        #clean the buffer of activations
        model.clean_activation_buffers()
        epoch_loss += l_batch 
        epoch_acc += acc_batch
        # print(accuracy)
        # pbar.set_description(f"{i+1} of {samples//batch_size + 1} batches passed. Loss: {round(epoch_loss / (i + 1), 5)}, Accuracy: {round(epoch_acc / ((i + 1)*batch_size), 5)}")
        if i % 10 == 0:
            print(f"{i+1} of {samples//batch_size + 1} batches passed. Loss: {round(epoch_loss / (i + 1), 5)}, Accuracy: {round(epoch_acc / ((i + 1)*batch_size), 5)}")
            loss_val.append(l_batch)
    if not scheduler is None:
        scheduler.step()

def evaluate_stream(model, data_load, loss_val, loss_fn=None, n_clips = 2, n_clip_frames=8):
    print("------------------------")
    model.eval()
    model.cuda()
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    with torch.no_grad():
        # pbar = tqdm(enumerate(data_load))
        for i, sample in enumerate(data_load):# pbar:
            res = sample["data"].permute(0, 2, 1, 3, 4)
            res = res.cuda()
            target = sample["label"].cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
              if loss_fn is None:
                output = F.log_softmax(model(res[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]), dim=1)
                loss = F.nll_loss(output, target)
              else:
                output = F.softmax(model(res), dim=1)
#                 target = F.one_hot(target, num_classes=5)
#                 target = target.float()
                loss = loss_fn(output, target)
            _, pred = torch.max(output, dim=1)
            tloss += loss.item()
            csamp += pred.eq(target).sum()
            # pbar.set_description(f"{i+1} of {samples//batch_size + 1} batches passed. Loss: {round(tloss / (i + 1), 5)}, Accuracy: {round(100.0 * csamp.item() / samples, 5)}")
    aloss = tloss /  len(data_load)
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) + '  Accuracy:' + '{:5}'.format(csamp) + '/' + '{:5}'.format(samples) + ' (' + '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')
    return csamp.item() / samples

def train_iter(model, optimz, data_load, loss_val, epoch, class_weight=None, scheduler=None, loss_fn=None):
    print("------------------------")
    print(f"EPOCH {epoch}")
    epoch_loss_cls, epoch_loss_loc, epoch_acc, epoch_tiou = 0, 0, 0, 0
    samples = len(data_load.dataset)
    model.train()
    model.cuda()
    model.clean_activation_buffers()
    optimz.zero_grad()
    # pbar = tqdm(enumerate(data_load))
    for i, sample in enumerate(data_load): # pbar:
        res = sample["data"].permute(0, 2, 1, 3, 4)
        result = model(res.cuda())
        if loss_fn is None:
            out_loc = F.relu(result[1])
            loss_loc = F.mse_loss(out_loc, sample["label_loc"].cuda())
        else:
            out = F.softmax(model(res.cuda()), dim=1)
            loss = loss_fn(out, sample["label"].cuda())
        loss_loc.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        epoch_loss_loc += loss_loc.item()
        tiou_ = calc_tiou(sample["label_loc"].cuda(), out_loc.cuda())
        epoch_tiou += tiou_.item()
        # print(accuracy)
        # pbar.set_description(f"{i+1} of {samples//batch_size + 1} batches passed. Loss: {round(epoch_loss / (i + 1), 5)}, Accuracy: {round(epoch_acc / (i + 1), 5)}")
        if i % 50 == 0:
            print(f"{i+1} of {samples//batch_size + 1} batches passed. Loc. Loss {round(epoch_loss_loc / (i + 1), 5)}, tIoU {round(epoch_tiou / (i + 1), 5)}")
            loss_val.append(loss_loc.item())
    if not scheduler is None:
        scheduler.step()
 
def evaluate(model, data_load, loss_val, loss_fn=None):
    print("------------------------")
    model.eval()
    
    samples = len(data_load.dataset)
    csamp, tiou_e, tloss_cls, tloss_loc = 0, 0, 0, 0
    model.clean_activation_buffers()
    with torch.no_grad():
        # pbar = tqdm(enumerate(data_load))
        for i, sample in enumerate(data_load): # pbar:
            res = sample["data"].permute(0, 2, 1, 3, 4)
            result = model(res.cuda())
            if loss_fn is None:
                output_loc = F.relu(result[1])
                loss_loc = F.mse_loss(output_loc, sample["label_loc"].cuda())
            else:
                output = F.softmax(model(res.cuda()), dim=1)
                loss = loss_fn(output, sample["label"].cuda())
            
            tloss_loc += loss_loc.item()
            tiou_e += calc_tiou(sample["label_loc"].cuda(), output_loc).item()
            model.clean_activation_buffers()
            # pbar.set_description(f"{i+1} of {samples//batch_size + 1} batches passed. Loss: {round(tloss / (i + 1), 5)}, Accuracy: {100.0 * csamp.item() / (i + 1), 5}")
    aloss_loc = tloss_loc / samples
    loss_val.append(aloss_loc)
    print('\nAverage test loc. loss: ' + '{:.4f}'.format(aloss_loc) + '  tIoU: ' + '{:4.2f}'.format(100.0 * tiou_e / samples) + '%\n')
    return tiou_e / samples


if __name__ == "__main__":
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device: {torch.cuda.device(0)}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    # Load new splits

    with open("train_dataset_localization_26K.json", "r") as rfile:
        train_dict = json.load(rfile)

    with open("val_dataset_localization_26K.json", "r") as rfile:
        val_dict = json.load(rfile)

    with open("test_dataset_localization_26K.json", "r") as rfile:
        test_dict = json.load(rfile)
    
    # Recalculate weights for new splits
    
    # For train split
    y_new = [t[0] for t in train_dict.values()]
    y_set = set(y_new)

    class_counts = {k: 0 for k in list(y_set)}

    for label in y_new:
        class_counts[label] += 1

    print(class_counts)

    # For test split
    y_val_new = [v[0] for v in val_dict.values()]
    y_set = set(y_val_new)

    class_counts_val = {k: 0 for k in list(y_set)}

    for label in y_val_new:
        class_counts_val[label] += 1

    print(class_counts_val)

    # Calculate class weights
    class_weights = {}
    i = 0
    for k, v in class_counts.items():
        class_weights[k] = len(train_dict) / (len(class_counts)*v)
        i += 1
    print(class_weights)

    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x / 255.), # scale in [0, 1],
        T.Resize((224, 224)),
        T.RandomHorizontalFlip()
        # T.RandomCrop((172, 172))
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x / 255.), # scale in [0, 1],
        T.Resize((224, 224))
        # T.CenterCrop((172, 172))
    ])
        
    train_dataset = KickPunchDataset(list(train_dict.keys()), list(train_dict.values()), 15, transform=transform) # old
    test_dataset = KickPunchDataset(list(val_dict.keys()), list(val_dict.values()), 15, transform=transform_test) # old

    # Create a DataLoader
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Iterate over batches
    # for batch in train_dataloader:
    #    # Access batch['data'] and batch['label'] for each batch
    #    print(batch['data'].shape, batch['label'].shape)
    #    print(batch['label'])
    #    break

    # Load the streaming model

    model = MoViNetWithLocalization(_C.MODEL.MoViNetA2, causal = False, pretrained = True)

    # Freezing all layers except localizer

    non_trainable_list = [
        "classifier.0.conv_1.conv3d.weight",
        "classifier.0.conv_1.conv3d.bias",
        "classifier.3.conv_1.conv3d.weight",
        "classifier.3.conv_1.conv3d.bias",
        "classifier_new.0.conv_1.conv3d.weight",
        "classifier_new.0.conv_1.conv3d.bias",
        "classifier_new.3.weight",
        "classifier_new.3.bias"
    ]

    for name, param in model.named_parameters():
        if name in non_trainable_list:
            param.requires_grad = False
    
    for name, param in model.named_parameters():
        print(f'Layer: {name}, Requires Grad: {param.requires_grad}')
    
    # Custom classifier

    weight_tensor = torch.Tensor(list(class_weights.values()))
    print(weight_tensor)

    # Training loop
    # import pdb
    # pdb.set_trace()

    N_EPOCHS = 50

    # model = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = True )
    start_time = time.time()

    trloss_val, tsloss_val = [], []
    # model.classifier[3] = torch.nn.Conv3d(2048, 51, (1,1,1))
    optimz = optim.Adam(model.parameters(), lr=0.00005)#, weight_decay=0.05)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimz, T_max=10, eta_min=0.000005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimz, T_0=5, T_mult=1, eta_min=0.000005)
    train_loss = torch.nn.CrossEntropyLoss(weight=weight_tensor.cuda(), label_smoothing=0.1)
    test_loss = torch.nn.CrossEntropyLoss()
    best_accuracy, last_epoch_checkpoint = 0, 0
    es_counter, patiance = 0, 10
    for epoch in range(1, N_EPOCHS + 1):
        # print('Epoch:', epoch)
        train_iter(model, optimz, train_dataloader, trloss_val, epoch, class_weight=weight_tensor.cuda())#, scheduler=scheduler
        current_val_accuracy = evaluate(model, test_dataloader, tsloss_val)
        if current_val_accuracy > best_accuracy:
            torch.save(model, f"weights/movinet_a2_ft_loc_{epoch}_epoch_val_tiou_{str(round(current_val_accuracy, 4)).replace('.', '_')}.pth")
            if os.path.exists(f"weights/movinet_a2_ft_loc_{last_epoch_checkpoint}_epoch_val_tiou_{str(round(best_accuracy, 4)).replace('.', '_')}.pth"):
                os.remove(f"weights/movinet_a2_ft_loc_{last_epoch_checkpoint}_epoch_val_tiou_{str(round(best_accuracy, 4)).replace('.', '_')}.pth")
            best_accuracy = current_val_accuracy
            last_epoch_checkpoint = epoch
            print(f"Checkpoint saved on {epoch} epoch!")
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= patiance:
                print("Reached the maximum number of epochs without update on accuracy. Stopping the training!")
                break
    
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')