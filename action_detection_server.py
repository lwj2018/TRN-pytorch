import os
import re
import time
import cv2 as cv
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
import socket
from collections import deque
import threading

q = deque()     # the queue for images 
video_label = ""      # the predicted str label
imageCount = 0
imageTimeStamp = 0
labelTimeStamp = 0
imageMutex = threading.Lock()     # it is used for the match between imageTimeStamp and image
labelMutex = threading.Lock()     # it is used for the match between labelTimeStamp and label

def tcplink(sock, addr):
    '''
        协议：
            'predict', 'train' :  status | buf_size
            'record'           :  status | buf_size | if_end | label | labelCount
    '''
    print('Accept new connection from %s:%s...' % addr)
    out_folder = "{}/{}".format(args.video_root, args.directory)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    while True:
        # 接受信息，按照协议进行解析
        start = time.time()
        message = sock.recv(100)
        for i in range(100):
            char = message[i]
            if char == 0:
                break
        message = message[0:i]
        print("\033[36m [MESSAGE]:  {}".format(message))
        message = message.decode()
        messages = message.split('|')
        status = messages[0]
        if status == 'predict' or status == 'train':
            assert len(messages)==2
            buf_size = int(messages[1].rstrip('\0'))
        if status == 'record':
            assert len(messages)==5
            buf_size = int(messages[1])
            if_end = messages[2]
            label = int(messages[3])
            labelCount = int(messages[4].rstrip('\0'))
        #print("\033[36m the buf size is :{}\033[0m".format(buf_size))
        # 接收图片
        data = sock.recv(buf_size+1, socket.MSG_WAITALL)
        #print("\033[36m the length of data is: {}\033[0m".format(len(data)))
        # 恢复被转换为字节流的图片
        data = np.fromstring(data, dtype = np.uint8)
        img = cv.imdecode(data, cv.IMREAD_COLOR)
        # record 状态下记录图片
        if status == 'record':
            imageCount += 1
            out_folder = "{}/{:d}_{:d}".format(args.video_root, label, labelCount)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            cv.imwrite("{}/{:06d}.jpg".format(out_folder,imageCount), img)
        else:
            imageCount = 0  # 在其他状态下清空计数
        img = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB)).convert('RGB')
        end = time.time()
        print("\033[36m Receiving and processing image cost : {} ms\033[0m".format((end-start)*1000))
        # 维持固定长度的队列
        q.append(img)
        if len(q)>args.seq_length:
            q.popleft()
        # 发送动作标签
        sock.send((video_label+'\0').encode('utf-8'))

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
parser.add_argument('--directory', type=str, default='test1')
parser.add_argument('--video_prefix', type=str, default='{:06d}.jpg')
parser.add_argument('--video_outpath', type=str, default='output/video')
parser.add_argument('--video_root', type=str, default='/media/storage/liweijie/datasets/pbd-v0-origin')
parser.add_argument('--seq_length', type=int, default=20)
# video_lenth = origin_video_length 
parser.add_argument('--video_length', type=int, default=355)
parser.add_argument('--speed', type=int, default=1)
parser.add_argument('--threshold', type=int, default=30)
parser.add_argument('--host', type=str, default="10.12.218.183")
parser.add_argument('--port', type=int, default=3000)
parser.add_argument('--buf_size', type=int,default=10000)

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

transform = torchvision.transforms.Compose([
    GroupScale(net.scale_size),
    GroupCenterCrop(net.input_size),
    Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    GroupNormalize(net.input_mean, net.input_std),
])

net = torch.nn.DataParallel(net.cuda())
net.eval()

# creating the socket
# CONFIG
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# the port for listening
s.bind((args.host, args.port))
print("Waiting for connecting...")
# receive a new connection
s.listen(5)
sock, addr = s.accept()
t = threading.Thread(target=tcplink, args=(sock,addr))
t.start()
# config for post process
now_label = -1
possible_label = -1
now_lasting_count = 0
possible_lasting_count = 0
lasting_threshold = args.threshold
video_label = "no predict"   # string label will be send to client
if not os.path.exists("recvImg"):
    os.makedirs("recvImg")
while True:
    # key = cv.waitKey(1)
    # if key=='q' or key=='Q':
    #     break
    if len(q)>=args.seq_length:
        # process the q to get input tensor
        img_list = list(q)
        tick = args.seq_length/float(args.test_segments)
        img_list = [img_list[int(tick / 2.0 + tick * x)] for x in range(args.test_segments)]
        imageCount += 1
        input_data = transform(img_list)
        start = time.time()
        input_var = torch.autograd.Variable(input_data.view(-1, 3, input_data.size(1), input_data.size(2)),
                                   volatile=True).unsqueeze(0)
        # predict
        logits = net(input_var)
        cnt_time = time.time() - start
        h_x = torch.mean(logits, dim=0).data
        probs, idx = h_x.sort(0, True)
        video_pred = idx[0]     # the int label for now video
        video_label = categories[video_pred]  # the str label for video
        print("\033[35m predict using net cost {}ms\033[0m".format(cnt_time*1000))    # print the costed time
        # flush the video label using some rule
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
            video_label = categories[now_label] # when action has lasting for enough time video_label changed
        if now_lasting_count>=lasting_threshold and now_label!=-1:
            video_label = categories[now_label] # when action has lasting for enough time video_label changed
    print("\033[35m now label is : {}\033[0m".format(video_label))
s.close()


