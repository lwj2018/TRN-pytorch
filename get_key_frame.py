# test the pre-trained model on a single video
# (working on it)
# Bolei Zhou and Alex Andonian

import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image

import torch.nn.parallel
import torch.optim
from models import TSN
from transforms import *
import datasets_video
from torch.nn import functional as F
import cv2 as cv

def get_frames(video_path, length):
    frames = []
    for i in range(length):
        frame = Image.open("{}/img_{:05d}.jpg".format(video_path,i+1)).convert('RGB')
        frames.append(frame)
    return frames

def get_input_frames(frames, indexes):
    input_frames = []
    for index in indexes:
        input_frames.append(frames[index])
    return input_frames

# options
parser = argparse.ArgumentParser(description="test TRN on a single video")
parser.add_argument('--video_path', type=str, default=None)
parser.add_argument('--length', type=int, default=None)
parser.add_argument('--test_segments', type=int, default=3)
parser.add_argument('--modality', type=str, default='RGB',
                    choices=['RGB', 'Flow', 'RGBDiff'], )
parser.add_argument('--dataset', type=str, default='pbd-v0',
                    choices=['something', 'jester', 'moments', 'pbd-v0'])
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
parser.add_argument('--weight', type=str, default='/media/storage/liweijie/c3d_models/trn/TRN_pbd-v0_RGB_BNInception_TRNmultiscale_segment3_best.pth.tar')
parser.add_argument('--num_class', type=int, default=5)
parser.add_argument('--arch', type=str, default="BNInception")

args = parser.parse_args()

directory = args.video_path.split('/')[-1]
label = int(directory.split('_')[0])
# Load model.
net = TSN(args.num_class,
          args.test_segments,
          args.modality,
          base_model=args.arch,
          consensus_type=args.consensus_type,
          img_feature_dim=args.img_feature_dim, print_spec=False)

weights = args.weight
checkpoint = torch.load(weights)
#print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
net.cuda().eval()

# Initialize frame transforms.

transform = torchvision.transforms.Compose([
    GroupOverSample(net.input_size, net.scale_size),
    Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    GroupNormalize(net.input_mean, net.input_std),
])

max_prob = 0
max_indexes = []
# Obtain video frames
frames = get_frames(args.video_path, args.length)
for i in range(args.length//args.test_segments):
    for j in range(args.length//args.test_segments,2*args.length//args.test_segments):
        for k in range(2*args.length//args.test_segments,args.length):
            indexes = [i,j,k]
            print("\33[35m now indexes: {} {} {}\033[0m".format(i,j,k))
            input_frames = get_input_frames(frames, indexes)
            data = transform(input_frames)
            input_var = torch.autograd.Variable(data.view(-1, 3, data.size(1), data.size(2)),
                                                volatile=True).unsqueeze(0).cuda()
            logits = net(input_var)
            scores = torch.mean(F.softmax(logits), dim=0).data
            probs, idx = scores.sort(0, True)
            if idx[0]==label:
                if probs[0]>max_prob:
                    max_prob = probs[0]
                    max_indexes = indexes

print("\033[36m max prob is : {}\033[0m".format(max_prob))
print("\033[36m max indexes is : {}\033[0m".format(max_indexes))

image1 = np.asarray(frames[max_indexes[0]])
image2 = np.asarray(frames[max_indexes[1]])
image3 = np.asarray(frames[max_indexes[2]])
height = image1.shape[0]
width = image1.shape[1]
image = np.zeros([height,width*3,3])
image[:,:width,:] = image1
image[:,width:2*width,:] = image2
image[:,width*2:width*3,:] = image3
image = Image.fromarray(np.uint8(image)).convert('RGB')
image.save('output/key_frame_{}.jpg'.format(directory))


