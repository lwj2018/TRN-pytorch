import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from dataset import TSNDataSet, VideoDataset
from models import TSN
from transforms import *
from ops import ConsensusModule
import datasets_video
import pdb
from torch.nn import functional as F
import cv2 as cv
import os


# options
parser = argparse.ArgumentParser(
    description="TRN testing on the full validation set")
parser.add_argument('dataset', type=str, choices=['something','jester','moments','charades','pbd','pbdnew','pbd-v0'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='TRN',
                    choices=['avg', 'TRN','TRNmultiscale'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--softmax', type=int, default=0)
parser.add_argument('--video_width', type=int, default=171)
parser.add_argument('--video_height', type=int, default=128)
parser.add_argument('--directory', type=str, default='gearbox')
parser.add_argument('--video_prefix', type=str, default='{:06d}.jpg')
parser.add_argument('--video_outpath', type=str, default='output/video')
parser.add_argument('--video_root', type=str, default='/home/liweijie/C3D/C3D-v1.0/examples/c3d_finetuning/input/test_frm')
parser.add_argument('--seq_length', type=int, default=20)
# video_lenth = origin_video_length 
parser.add_argument('--video_length', type=int, default=903)
parser.add_argument('--speed', type=int, default=1)

args = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality)
num_class = len(categories)

net = TSN(num_class, args.test_segments if args.crop_fusion_type in ['TRN','TRNmultiscale'] else 1, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          img_feature_dim=args.img_feature_dim,
          )

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = {}

# data_loader['5'] = torch.utils.data.DataLoader(
#         VideoDataset( directory=args.directory, num_segments=args.test_segments,
#                    root_path=args.video_root,
#                    new_length=1 if args.modality == "RGB" else 5,
#                    modality=args.modality,
#                    image_tmpl=args.video_prefix,
#                    test_mode=True,
#                    video_length=args.video_length,
#                    seq_length=5,
#                    speed=args.speed,
#                    transform=torchvision.transforms.Compose([
#                        cropping,
#                        Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
#                        ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
#                        GroupNormalize(net.input_mean, net.input_std),
#                    ])),
#         batch_size=1, shuffle=False,
#         num_workers=0, pin_memory=True)

data_loader['10'] = torch.utils.data.DataLoader(
        VideoDataset( directory=args.directory, num_segments=args.test_segments,
                   root_path=args.video_root,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl=args.video_prefix,
                   test_mode=True,
                   video_length=args.video_length,
                   seq_length=10,
                   speed=args.speed,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

data_loader['20'] = torch.utils.data.DataLoader(
        VideoDataset( directory=args.directory, num_segments=args.test_segments,
                   root_path=args.video_root,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl=args.video_prefix,
                   test_mode=True,
                   video_length=args.video_length,
                   seq_length=20,
                   speed=args.speed,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

data_loader['40'] = torch.utils.data.DataLoader(
        VideoDataset( directory=args.directory, num_segments=args.test_segments,
                   root_path=args.video_root,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl=args.video_prefix,
                   test_mode=True,
                   video_length=args.video_length,
                   seq_length=40,
                   speed=args.speed,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)


if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


#net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net = torch.nn.DataParallel(net.cuda())
net.eval()

data_gen = {}
total_num = {}
for k in data_loader.keys():
    data_gen[k] = enumerate(data_loader[k])
    total_num[k] = len(data_loader[k].dataset)

output = []

def eval_video(video_data):
    i, data = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    # TODO: why view?
    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        volatile=True)
    input_var = torch.autograd.Variable(data,
                                    volatile=True)

    rst = net(input_var)
    if args.softmax==1:
        # take the softmax to normalize the output to probability
        rst = F.softmax(rst)

    rst = rst.data.cpu().numpy().copy()

    if args.crop_fusion_type in ['TRN','TRNmultiscale']:
        rst = rst.reshape(-1, 1, num_class)
    else:
        rst = rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

    return i, rst

proc_start_time = time.time()

top1 = AverageMeter()
top5 = AverageMeter()
out_frm_folder = "{}/{}".format(args.video_outpath, args.directory)
if not os.path.exists(args.video_outpath):
    os.makedirs(args.video_outpath)
if not os.path.exists(out_frm_folder):
    os.makedirs(out_frm_folder)
out_video_name = "{}/{}_segment{}_window{}_speed{}.avi".format(args.video_outpath,args.directory, 
                                                            args.test_segments, args.seq_length, args.speed)
print("\033[32m Creating the video writer...\033[0m")
writer = cv.VideoWriter(out_video_name,cv.VideoWriter_fourcc('M','J','P','G'),15.0,(args.video_width,args.video_height))
# the np array for recording score
save_file_name = "{}/{}_segment{}_window{}_speed{}_scores.npy".format(args.video_outpath,args.directory,
                                                            args.test_segments,args.seq_length,args.speed)

# infor_list = []
# for k in data_gen.keys():
#     for i, data in data_gen[k]:
#         rst = eval_video((i, data))
#         scores = np.mean(rst[1], axis=0)
#         video_pred = float(np.squeeze(np.argmax(scores)))
#         max_score = float(np.squeeze(np.max(scores)))
#         cnt_time = time.time() - proc_start_time
#         infor = np.zeros([6], dtype=np.float32)
#         infor[0] = i                # start frame
#         infor[1] = i+int(k)         # end frame
#         infor[2] = video_pred       # frame label
#         infor[3] = max_score        # label score
#         infor[4] = 1                # not selected : 1, selected : 0
#         infor[5] = 1                # not abused : 1, abused : 0
#         infor_list.append(infor)
#         print('\033[34m [window {}] video {} done, total {}/{}, average {:.3f} sec/video, pred label: {}\033[0m'.format(int(k),
#                                                                         i, 
#                                                                         i+1,
#                                                                         total_num[k],
#                                                                         float(cnt_time) / (i+1),
#                                                                         video_pred))

# infor_list = np.array(infor_list)
# length = infor_list.shape[0]
# while(np.max(infor_list[:,4])>0):
#     max_index = np.argmax(infor_list[:,3]*infor_list[:,4])
#     max_start_frame = infor_list[max_index, 0]
#     max_end_frame = infor_list[max_index, 1]
#     infor_list[max_index, 4] = 0    
#     for i in range(length):
#         if infor_list[i,4]!=0:
#             now_start_frame = infor_list[i,0]
#             now_end_frame = infor_list[i,1]
#             iou = (min(now_end_frame,max_end_frame) - max(now_start_frame,max_start_frame)) / (max(now_end_frame,max_end_frame) - min(now_start_frame,max_start_frame))
#             if iou>0.5:
#                 infor_list[i,4] = 0
#                 infor_list[i,5] = 0
#             elif (max_start_frame>now_start_frame) and (max_end_frame<now_end_frame):
#                 infor_list[i,4] = 0
#                 infor_list[i,5] = 0
# np.save("output/infor_list.npy",infor_list)
infor_list = np.load("output/infor_list.npy")
show_map = np.zeros([200,args.video_length,3])
# blue : place, green : screw, red : turndown, purple : press, yellow : empty  
color_map = {0:[255,0,0],1:[0,255,0],2:[0,0,255],3:[255,0,255],4:[0,255,255]}
length = infor_list.shape[0]
for i in range(length):
    if infor_list[i,5]!=0 and infor_list[i,3]>10:
        print(infor_list[i,0],infor_list[i,1],infor_list[i,2],infor_list[i,3])
        for j in range(int(infor_list[i,0]),int(infor_list[i,1])):
            for ch in range(3):
                show_map[:,j,ch] = color_map[infor_list[i,2]][ch]

cv.imwrite("output/show_map.jpg",show_map)

