import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data_functions import TrainData
from val_data_functions import ValData
from utils import to_psnr, print_log, validation_val, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random 
from torch.utils.tensorboard import SummaryWriter

from deform_deweather_net import DeformDeweatherNet

from utils import calc_psnr, calc_ssim

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-train_dir', help='Train data directory', default="/dataset/public/raindrop/train", type=str)
parser.add_argument('-test_dir', help='Test data directory', default="/dataset/public/raindrop/test", type=str)
parser.add_argument('-rain_subdir', help='Sub-directory for input data', default="data", type=str)
parser.add_argument('-gt_subdir', help='Sub-directory for gt data', default="gt", type=str)
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=4, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="ckpt", type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=1000, type=int)
parser.add_argument('-eval_interval', help='evaluation intervals', default=10, type=int)
parser.add_argument('-logdir', help='for tensorboard', default="deformable-deweathering", type=str)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
tensorboard_logdir = args.logdir
eval_interval = args.eval_interval
train_data_dir = args.train_dir
validate_data_dir = args.test_dir
rain_dir = args.rain_subdir
gt_dir = args.gt_subdir

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # If GPU is available, change to GPU
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}\ntrain_dir: {}\ntest_dir: {}'.format(learning_rate, crop_size, train_batch_size, val_batch_size, lambda_loss, train_data_dir, validate_data_dir))

log_dir = os.path.join("./logs", tensorboard_logdir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = DeformDeweatherNet()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

modelbest_path = "./{}/best".format(log_dir)
if os.path.exists(modelbest_path):
    resume_state = torch.load(modelbest_path)
    net.load_state_dict(resume_state["state_dict"])
    optimizer.load_state_dict(resume_state["optimizer"])
    epoch_start = resume_state["epoch"]
    print("----- {} trained loaded -----".format(modelbest_path))
elif os.path.exists("./{}/pretrain".format(log_dir)):
    resume_state = torch.load("./{}/pretrain".format(log_dir))
    net.load_state_dict(resume_state["state_dict"])
    print("----- pretrained model loaded -----")
else:
    print('--- no weight loaded ---')


loss_network = LossNetwork(vgg_model)
loss_network.eval()

lbl_train_data_loader = DataLoader(TrainData(train_data_dir, rain_dir, gt_dir, crop_size), batch_size=train_batch_size, shuffle=True, num_workers=8)

val_data_loader1 = DataLoader(ValData(validate_data_dir, rain_dir, gt_dir), batch_size=val_batch_size, shuffle=False, num_workers=8)

print("Number of training batches: {}".format(len(lbl_train_data_loader)))
print("Number of validation batches: {}".format(len(val_data_loader1)))

net.eval()

old_val_psnr1, old_val_ssim1 = validation_val(net, val_data_loader1, device, exp_name)
print('Rain Drop old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr1, old_val_ssim1))

net.train()

writer = SummaryWriter(log_dir)

count = epoch_start * len(lbl_train_data_loader)

for epoch in range(epoch_start,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, lr_decay=0.5, step=100)
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)
        optimizer.zero_grad()
        net.train()
        pred_image = net(input_image)
        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)
        loss = smooth_loss + lambda_loss*perceptual_loss 
        loss.backward()
        optimizer.step()
        psnr_list.extend(to_psnr(pred_image, gt))
        count += 1
        writer.add_scalar("Loss/train", loss.item(), count)
        if not (batch_id % 100):
            save_state = {
                            "epoch": epoch,
                            "state_dict": net.state_dict(),
                            "optimizer": optimizer.state_dict()
                        }
            torch.save(save_state, os.path.join(log_dir, "latest"))
            print('Epoch: {0}, Iteration: {1}, loss: {2}'.format(epoch, batch_id, loss.item()))

    train_psnr = sum(psnr_list) / len(psnr_list)

    save_state = {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
    torch.save(save_state, os.path.join(log_dir, "latest"))

    if epoch % eval_interval == 0:
        net.eval()
        psnr_list = []
        ssim_list = []
        val_lossval_list = []

        for batch_id, val_data in enumerate(val_data_loader1):

            with torch.no_grad():
                input_im, gt, imgid = val_data
                input_im = input_im.to(device)
                gt = gt.to(device)
                pred_image = net(input_im)
                pred_image = torch.clamp(pred_image, 0.0, 1.0)

                val_smooth_loss = F.smooth_l1_loss(pred_image, gt)
                val_perceptual_loss = loss_network(pred_image, gt)
                val_loss = val_smooth_loss + lambda_loss * val_perceptual_loss 

                val_lossval_list.append(val_loss.item())

            psnr_list.extend(calc_psnr(pred_image, gt))
            ssim_list.extend(calc_ssim(pred_image, gt))

        val_psnr1 = sum(psnr_list) / (len(psnr_list) + 1e-10)
        val_ssim1 = sum(ssim_list) / (len(ssim_list) + 1e-10)
        val_lossval = sum(val_lossval_list) / (len(val_lossval_list) + 1e-10)

        writer.add_scalar("Validation/PSNR", val_psnr1, count)
        writer.add_scalar("Validation/SSIM", val_ssim1, count)
        writer.add_scalar("Validation/loss", val_lossval, count)

        one_epoch_time = time.time() - start_time
        print("Rain Drop")
        print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr1, val_ssim1, exp_name)

        if val_psnr1 >= old_val_psnr1:
            save_state = {
                            "epoch": epoch,
                            "state_dict": net.state_dict(),
                            "optimizer": optimizer.state_dict()
                        }
            torch.save(save_state, os.path.join(log_dir, "best"))
            print('model saved')
            old_val_psnr1 = val_psnr1
