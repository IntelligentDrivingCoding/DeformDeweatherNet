import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
# from val_data_functions_test1 import ValData
# from val_data_functions_raindrop_testa import ValData
from val_data_functions_snow100kl import ValData
from utils import validation, validation_val, validation_val_all, validation_val_input
import os
import numpy as np
import random
# from transweather_model_customized import Transweather
from transweather_model_change import Transweather
# from transweather_model_original import Transweather

from torchinfo import summary

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

val_batch_size = 1

# log_dir = "transweather_change_k3_3DCN_raw_256_gradnorm"
log_dir = "transweather_change_k7_3DCN_raw_256_gradnorm"

# log_dir = "transweather_customized_raw_256_gradnorm"
# log_dir = "transweather_change_k3_raw_256_gradnorm"
# log_dir = "transweather_raw"

# log_dir = "transweather_change_singledeform_k3_raw_newtest"
# log_dir = "transweather_change_singledeform_k7_raw_newtest"
# log_dir = "transweather_change_k3_raw_newtest"

model = "best" # "best" or "latest"
if_clamp = True
# if_clamp = False
save_dir = os.path.join("./logs", log_dir, "tmp")

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = '/dataset/public/raindrop/test_for_paper/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Validation data loader --- #

val_filename = 'raindroptesta.txt' ## This text file should contain all the names of the images and must be placed in ./data/test/ directory

val_data_loader = DataLoader(ValData(), batch_size=val_batch_size, shuffle=False, num_workers=8)

# --- Define the network --- #

net = Transweather().cuda()

summary(net, input_size=(1, 3, 256, 256), device='cuda')

net = nn.DataParallel(net, device_ids=device_ids)


# --- Load the network weight --- #
best_model_path = os.path.join("./logs", log_dir, model)
if log_dir == "transweather_raw":
    net.load_state_dict(torch.load(best_model_path))
else:
    net.load_state_dict(torch.load(best_model_path)['state_dict'])

# --- Use the evaluation model in testing --- #
net.eval()
category = "raindroptest"

# if os.path.exists('./results/{}/{}/'.format(category,exp_name))==False:     
#     os.makedirs('./results/{}/{}/'.format(category,exp_name))   


print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim = validation_val(net, val_data_loader, device, log_dir,category, save_tag=False, if_clamp=if_clamp)
# val_psnr, val_ssim = validation_val(net, val_data_loader, device, save_dir, category, save_tag=True)
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))

# val_psnr, val_ssim, val_mse, val_fsim, val_niqe, val_brisque, another_brisque = validation_val_all(net, val_data_loader, device, save_dir, category, save_tag=True)
# # val_psnr, val_ssim, val_mse, val_fsim, val_niqe, val_brisque, another_brisque = validation_val_input(net, val_data_loader, device, save_dir, category, save_tag=False)
# end_time = time.time() - start_time
# print('val_psnr: {0:.2f}'.format(val_psnr))
# print('val_ssim: {0:.4f}'.format(val_ssim))
# print('val_mse: {0:.4f}'.format(val_mse))
# print('val_fsim: {0:.4f}'.format(val_fsim))
# print('val_niqe: {0:.4f}'.format(val_niqe))
# print('val_brisque: {0:.4f}'.format(val_brisque))
# print('another val_brisque: {0:.4f}'.format(another_brisque))
# # print('validation time is {0:.4f}'.format(end_time))
