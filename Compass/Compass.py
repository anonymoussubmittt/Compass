import torch
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse
import time
from tqdm import tqdm
import joblib

from utils.compass_utils import one_masking_statistic, double_masking_detection, double_masking_detection_nolemma1
from utils.setup import get_model, get_data_loader
from utils.defense import gen_mask_set, double_masking_precomputed, certify_precomputed

#
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default='checkpoints', type=str, help="directory of checkpoints")
parser.add_argument('--data_dir', default='./../../../../public', type=str, help="directory of data")
parser.add_argument('--dataset', default='imagenet', type=str,
                    choices=('imagenette', 'imagenet', 'cifar', 'cifar100', 'svhn', 'flower102'), help="dataset")
parser.add_argument("--model", default='vit_base_patch16_224_cutout2_128', type=str, help="model name")
parser.add_argument("--num_img", default=-1, type=int,
                    help="number of randomly selected images for this experiment (-1: using the all images)")
parser.add_argument("--mask_stride", default=-1, type=int, help="mask stride s (square patch; conflict with num_mask)")
parser.add_argument("--num_mask", default=6, type=int,
                    help="number of mask in one dimension (square patch; conflict with mask_stride)")
parser.add_argument("--patch_size", default=35, type=int, help="size of the adversarial patch (square patch)")
parser.add_argument("--pa", default=-1, type=int,
                    help="size of the adversarial patch (first axis; for rectangle patch)")
parser.add_argument("--pb", default=-1, type=int,
                    help="size of the adversarial patch (second axis; for rectangle patch)")
parser.add_argument("--dump_dir", default='dump', type=str, help='directory to dump two-mask predictions')
parser.add_argument("--override", action='store_true', help='override dumped file')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parser.parse_args()
DATASET = args.dataset
MODEL_DIR = os.path.join('.', args.model_dir)
DATA_DIR = os.path.join(args.data_dir, DATASET)
DUMP_DIR = os.path.join('.', args.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)

MODEL_NAME = args.model
NUM_IMG = args.num_img

# get model and data loader
model = get_model(MODEL_NAME, DATASET, MODEL_DIR)
val_loader, NUM_IMG, ds_config = get_data_loader(DATASET, DATA_DIR, model, batch_size=1, num_img=NUM_IMG, train=False)

device = 'cuda'
model = model.to(device)
model.eval()
cudnn.benchmark = True

# generate the mask set
mask_list, MASK_SIZE, MASK_STRIDE = gen_mask_set(args, ds_config)

# the computation of two-mask predictions is expensive; will dump (or resue the dumped) two-mask predictions.
SUFFIX = '_two_mask_{}_{}_m{}_s{}_{}.z'.format(DATASET, MODEL_NAME, MASK_SIZE, MASK_STRIDE, NUM_IMG)

clean_corr = 0
robust = 0
orig_corr = 0
NUM_IMG = 0
statistics_dict = {}



# cifar10 2.4%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_cifar_vit_base_patch16_224_cutout2_128_m(66, 66)_s(32, 32)_10000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))

# cifar100 2.4%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_cifar100_vit_base_patch16_224_cutout2_128_m(66, 66)_s(32, 32)_10000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_cifar100_vit_base_patch16_224_cutout2_128_10000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_cifar100_vit_base_patch16_224_cutout2_128_10000.z"))

# imagenette 2%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenette_vit_base_patch16_224_cutout2_128_m(64, 64)_s(33, 33)_3925.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenette_vit_base_patch16_224_cutout2_128_3925.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenette_vit_base_patch16_224_cutout2_128_3925.z"))

# imagenet 1%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(56, 56)_s(34, 34)_50000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))

# imagenet 2%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(64, 64)_s(33, 33)_50000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))

# imagenet 3%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,
#                                                "prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(69, 69)_s(31, 31)_50000.z"))
# orig_prediction_list = joblib.load(
#     os.path.join(DUMP_DIR, "orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR, "label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))

# imagenet 25%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(117, 117)_s(22, 22)_50000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))

# cifar10 2.4%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_cifar_vit_base_patch16_224_cutout2_128_m(66, 66)_s(32, 32)_10000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))

# gtsrb 2%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_gtsrb_vit_base_patch16_224_cutout2_128_m(64, 64)_s(33, 33)_12630.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_gtsrb_vit_base_patch16_224_cutout2_128_12630.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_gtsrb_vit_base_patch16_224_cutout2_128_12630.z"))

# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_gtsrb_vit_base_patch16_224_cutout2_69_m(64, 64)_s(33, 33)_12630.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_gtsrb_vit_base_patch16_224_cutout2_69_12630.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_gtsrb_vit_base_patch16_224_cutout2_69_12630.z"))

# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_gtsrb_vit_base_patch16_224_cutout2_64_m(64, 64)_s(33, 33)_12630.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_gtsrb_vit_base_patch16_224_cutout2_64_12630.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_gtsrb_vit_base_patch16_224_cutout2_64_12630.z"))


# imagenet
# imagenet 0.5%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(50, 50)_s(35, 35)_50000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))

# # imagenet 2%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(64, 64)_s(33, 33)_50000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))

# imagenet 4.6%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(77, 77)_s(30, 30)_50000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
#
# # imagenet 8.2%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(90, 90)_s(27, 27)_50000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))

# # imagenet 12.80%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(104, 104)_s(25, 25)_50000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))

# # imagenet 18.4%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(117, 117)_s(22, 22)_50000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))

# imagenet 25.00%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(130, 130)_s(19, 19)_50000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))



# CIFAR
# cifar 0.5%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_cifar_vit_base_patch16_224_cutout2_128_m(50, 50)_s(35, 35)_10000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))

# # cifar 2%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_cifar_vit_base_patch16_224_cutout2_128_m(64, 64)_s(33, 33)_10000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))

# cifar 4.6%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_cifar_vit_base_patch16_224_cutout2_128_m(77, 77)_s(30, 30)_10000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
#
# # cifar 8.2%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_cifar_vit_base_patch16_224_cutout2_128_m(90, 90)_s(27, 27)_10000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))

# # cifar 12.80%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_cifar_vit_base_patch16_224_cutout2_128_m(104, 104)_s(25, 25)_10000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))

# # cifar 18.4%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_cifar_vit_base_patch16_224_cutout2_128_m(117, 117)_s(22, 22)_10000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))

# cifar 25.00%
# prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_cifar_vit_base_patch16_224_cutout2_128_m(130, 130)_s(19, 19)_10000.z"))
# orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
# label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_cifar_vit_base_patch16_224_cutout2_128_10000.z"))
#



# statistic for two-mask detection
warning_cert = 0
warning_notcert = 0
cert_nowarning = 0
cert_warning = 0
correct_sample = 0
my_cert=0

for j in range(36):
    # if j<5:
    #     continue
    for i, (prediction_map, label, orig_pred) in enumerate(zip(prediction_map_list, label_list, orig_prediction_list)):
        NUM_IMG += 1
        prediction_map = prediction_map + prediction_map.T - np.diag(
            np.diag(prediction_map))  # generate a symmetric matrix from a triangle matrix
        prediction_label, index = double_masking_detection(prediction_map, bear=j)
        # -1 for cert and warn
        # -2 for not cert and warn
        # -3 for cert and not warn
        # -4 for cert and warn
        if prediction_label == label:
            correct_sample += 1
            if index == -1:
                warning_cert += 1
            elif index == -2:
                warning_notcert += 1
            elif index == -3:
                cert_nowarning += 1
            elif index == -4:
                cert_warning += 1

    my_cert=cert_nowarning+cert_warning+warning_cert
    their_cert=cert_nowarning


    print("\n")
    print("NUM_IMG"+str(NUM_IMG))
    print("bear is " + str(j))
    print("my_cert" + str(my_cert)+" "+str(my_cert/NUM_IMG))
    print("correct_sample"+str(correct_sample)+" "+str(correct_sample/NUM_IMG))

    print("\n")
    warning_cert = 0
    warning_notcert = 0
    cert_nowarning = 0
    cert_warning = 0
    correct_sample = 0
    my_cert =0
    NUM_IMG=0
#
# cor=0






# statistic for one-mask
# for i,(prediction_map,label,orig_pred) in enumerate(zip(prediction_map_list,label_list,orig_prediction_list)):
#     NUM_IMG+=1
#     prediction_map = prediction_map + prediction_map.T - np.diag(np.diag(prediction_map)) #generate a symmetric matrix from a triangle matrix
#     # if double_masking_precomputed(prediction_map) == label:
#     disagree_num,pred=one_masking_statistic(prediction_map)
#     # if pred == label and disagree_num==0:
#     if orig_pred == label:
#         cor+=1
#     print("cor",cor)

# if pred==label:
#     if statistics_dict.get(disagree_num) is None:
#         statistics_dict[disagree_num] = 1
#     else:
#         statistics_dict[disagree_num] += 1
#
# print("\n")
# for i in sorted (statistics_dict) :
#     print ((i, statistics_dict[i]), end =" ")


# statistic for two-mask recovery
# for i,(prediction_map,label,orig_pred) in enumerate(zip(prediction_map_list,label_list,orig_prediction_list)):
#     NUM_IMG+=1
#     prediction_map = prediction_map + prediction_map.T - np.diag(np.diag(prediction_map)) #generate a symmetric matrix from a triangle matrix
#     robust += certify_precomputed(prediction_map,label)
#     clean_corr += double_masking_precomputed(prediction_map) == label
#     orig_corr += orig_pred == label
#
#     print("------------------------------")
#     print("Certified robust accuracy:",robust)
#     print("Clean accuracy with defense:",clean_corr)
#     print("Clean accuracy without defense:",orig_corr)
