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
parser.add_argument('dataset', type=str, choices=['something','jester','moments',
                                        'charades','pbd','pbdnew','pbd-v0','pbd-v0.1'])
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
#parser.add_argument('--video_root', type=str, default='/home/liweijie/C3D/C3D-v1.0/examples/c3d_finetuning/input/test_frm')
parser.add_argument('--video_root', type=str, default='/media/storage/liweijie/datasets/test_videos')
parser.add_argument('--seq_length', type=int, default=20)
# video_lenth = origin_video_length 
parser.add_argument('--video_length', type=int, default=903)
parser.add_argument('--speed', type=int, default=1)
parser.add_argument('--threshold', type=int, default=20)

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

data_loader = torch.utils.data.DataLoader(
        VideoDataset( directory=args.directory, num_segments=args.test_segments,
                   root_path=args.video_root,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl=args.video_prefix,
                   test_mode=True,
                   video_length=args.video_length,
                   seq_length=args.seq_length,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


#net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net = torch.nn.DataParallel(net.cuda())
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
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
    print("\033[31m the shape of data is:{}\033[0m".format(data.size()))
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
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

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
save_arr = np.zeros([max_num,num_class])
save_file_name = "{}/{}_segment{}_window{}_speed{}_scores.npy".format(args.video_outpath,args.directory,
                                                            args.test_segments,args.seq_length,args.speed)

now_label = -1
possible_label = -1
now_lasting_count = 0
possible_lasting_count = 0
lasting_threshold = args.threshold
for i, data in data_gen:
    if i >= max_num:
        break
    rst = eval_video((i, data))
    scores = np.mean(rst[1], axis=0)
    video_pred = np.argmax(scores)
    for j in range(num_class):
        save_arr[i,j] = float(np.squeeze(scores)[j])
    video_label = categories[video_pred]
    cnt_time = time.time() - proc_start_time

    image_name = args.video_prefix.format((i+args.seq_length)*args.speed+1)
    image_name = "{}/{}/{}".format(args.video_root, args.directory, image_name)
    print("read from: {}".format(image_name))
    img_data = cv.imread(image_name)
    img_data = cv.resize(img_data,(args.video_width, args.video_height))

    if video_pred!=now_label:
        if video_pred == possible_label:
            possible_lasting_count+=1
        elif video_pred != possible_label:
            possible_lasting_count=0
        possible_label = video_pred
    elif video_pred==now_label:
        now_lasting_count += 1
        now_lasting_count += possible_lasting_count
        possible_lasting_count == 0
    if possible_lasting_count>=lasting_threshold and possible_label!=-1:
        now_label = possible_label
        now_lasting_count = possible_lasting_count
        possible_lasting_count = 0
        now_label_str = categories[now_label]
        if now_label_str == 'turndown':
            cv.putText(img_data,now_label_str,(50,20),cv.FONT_HERSHEY_PLAIN,1,(255,0,255))
        else:
            cv.putText(img_data,now_label_str,(100,20),cv.FONT_HERSHEY_PLAIN,1,(255,0,255))
    if now_lasting_count>=lasting_threshold and now_label!=-1:
        now_label_str = categories[now_label]
        if now_label_str == 'turndown':
            cv.putText(img_data,now_label_str,(50,20),cv.FONT_HERSHEY_PLAIN,1,(255,0,255))
        else:
            cv.putText(img_data,now_label_str,(100,20),cv.FONT_HERSHEY_PLAIN,1,(255,0,255))
    cv.imwrite("{}/{:0>6}.jpg".format(out_frm_folder, i+1), img_data)  
    writer.write(img_data)

    print('\033[34m video {} done, total {}/{}, average {:.3f} sec/video, now label: {}\033[0m'.format(i, 
                                                                    i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1),
                                                                    now_label))

np.save(save_file_name, save_arr)



