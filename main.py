from __future__ import print_function
import os
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

from trainer.fixbi_trainer import train_fixbi
from src.dataset import get_dataset
import src.utils as utils

import wandb


##################### argument ################################
#############           (1)                 ##################
##############################################################
parser = argparse.ArgumentParser(description="FIXBI EXPERIMENTS")
parser.add_argument('-db_path', help='gpu number', type=str, default='database')
parser.add_argument('-baseline_path', help='baseline path', type=str, default='AD_Baseline')
parser.add_argument('-save_path', help='save path', type=str, default='Logs/save_test')
parser.add_argument('-source', help='source', type=str, default='amazon')
parser.add_argument('-target', help='target', type=str, default='dslr')
parser.add_argument('-workers', default=4, type=int, help='dataloader workers')
parser.add_argument('-gpu', help='gpu number', type=str, default='0')
parser.add_argument('-epochs', default=100, type=int)
parser.add_argument('-batch_size', default=32, type=int)

parser.add_argument('-th', default=2.0, type=float, help='Threshold')
parser.add_argument('-bim_start', default=100, type=int, help='Bidirectional Matching')
parser.add_argument('-sp_start', default=100, type=int, help='Self-Penalization')
parser.add_argument('-cr_start', default=100, type=int, help='Consistency Regularization')
parser.add_argument('-lam_sd', default=0.7, type=float, help='Source Dominant Mixup ratio')
parser.add_argument('-lam_td', default=0.3, type=float, help='Target Dominant Mixup ratio')


def main(args):

    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Use GPU(s): {} for training".format(args.gpu))
    print(args)

##################### dataset ################################
#############           (2)                 ##################
##############################################################
# -> dataset.py
    num_classes, resnet_type = utils.get_data_info()
    # resnet_type = 50, num_classes = 31
    src_trainset, src_testset = get_dataset(args.source, path=args.db_path)
    # train, test 별로 image data folder
    tgt_trainset, tgt_testset = get_dataset(args.target, path=args.db_path)
    # train, test 별로 image data folder



################### dataloader ##############################
#############           (3)                 ##################
##############################################################
    src_train_loader = torchdata.DataLoader(src_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_train_loader = torchdata.DataLoader(tgt_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_test_loader = torchdata.DataLoader(tgt_testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)



###################### Model #################################
#############           (4)                 ##################
##############################################################
# -> utils.py
    lr, l2_decay, momentum, nesterov = utils.get_train_info()
    # 0.001, 5e-4, 0.9, False
    net_sd, head_sd, classifier_sd = utils.get_net_info(num_classes)
    # Pretrained_ResNet50, nn.Linear(256, num_classes), # linear -> batch -> relu
    net_td, head_td, classifier_td = utils.get_net_info(num_classes)
    # Pretrained_ResNet50, nn.Linear(256, num_classes), # linear -> batch -> relu



###################### param #################################
#############          optim                ##################
####################### Loss #################################
    learnable_params_sd = list(net_sd.parameters()) + list(head_sd.parameters()) + list(classifier_sd.parameters())
    # 각 모델별 parameter
    learnable_params_td = list(net_td.parameters()) + list(head_td.parameters()) + list(classifier_td.parameters())
    # 각 모델별 parmeter 
    optimizer_sd = optim.SGD(learnable_params_sd, lr=lr, momentum=momentum, weight_decay=l2_decay, nesterov=nesterov)
    optimizer_td = optim.SGD(learnable_params_td, lr=lr, momentum=momentum, weight_decay=l2_decay, nesterov=nesterov)
    # optim 설정
    sp_param_sd = nn.Parameter(torch.tensor(5.0).cuda(), requires_grad=True)
    sp_param_td = nn.Parameter(torch.tensor(5.0).cuda(), requires_grad=True)
    # softmax sharpening para
    optimizer_sd.add_param_group({"params": [sp_param_sd], "lr": lr})
    optimizer_td.add_param_group({"params": [sp_param_td], "lr": lr})

    ce = nn.CrossEntropyLoss().cuda()
    mse = nn.MSELoss().cuda()


###################### Baseline ##############################
###############           (6)                 ################
##############################################################
# -> utils.py
    net_sd, head_sd, classifier_sd = utils.load_net(args, net_sd, head_sd, classifier_sd)
    # 각 모델에 가중치 load
    net_td, head_td, classifier_td = utils.load_net(args, net_td, head_td, classifier_td)

    #net_td, head_td, classifier_td = copy.deepcopy(net_sd), copy.deepcopy(head_sd), copy.deepcopy(classifier_sd)
    # 내부 객체까지 복사
    loaders = [src_train_loader, tgt_train_loader]
    # 각 데이터 로더
    optimizers = [optimizer_sd, optimizer_td]
    models_sd = [net_sd, head_sd, classifier_sd]
    models_td = [net_td, head_td, classifier_td]
    sp_params = [sp_param_sd, sp_param_td]
    # ?
    losses = [ce, mse]
    # optim, loss 등


###################### Train #################################
###############         (7)               ####################
##############################################################
    for epoch in range(args.epochs):
        train_fixbi(args, loaders, optimizers, models_sd, models_td, sp_params, losses, epoch)


###################### valid #################################
###############         (8)               ####################
##############################################################
# -> utils.py
        utils.evaluate(nn.Sequential(*models_sd), tgt_test_loader)
        # sd 검증
        utils.evaluate(nn.Sequential(*models_td), tgt_test_loader)
        # td 검증
        utils.final_eval(nn.Sequential(*models_sd), nn.Sequential(*models_td), tgt_test_loader)
        
        utils.save_net(args, models_sd, 'sdm')
        utils.save_net(args, models_td, 'tdm')

        wandb.log({
            "Train Loss": round(losses, 4)
        })

if __name__ == "__main__":
    args = parser.parse_args()
    wandb.init(project="dann", name="office-31", entity="jg_lee", config=vars(args))
    main(args)
    wandb.run.finish()
