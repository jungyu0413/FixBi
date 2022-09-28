import time
import torch
import torch.nn as nn
import src.utils as utils

###################### train #################################
###############                            ###################
##############################################################
def train_fixbi(args, loaders, optimizers, models_sd, models_td, sp_params, losses, epoch):
    print("Epoch: [{}/{}]".format(epoch, args.epochs))
    start = time.time()

###################### dataloader ############################
###############            (1)               #################
##############################################################
    src_train_loader, tgt_train_loader = loaders[0], loaders[1]
    # source_data_loader 입력, target_data_loader 입력
    optimizer_sd, optimizer_td = optimizers[0], optimizers[1]
    sp_param_sd, sp_param_td = sp_params[0], sp_params[1]
    # softmax sharpening para?
    ce, mse = losses[0], losses[1]
    # optim, loss등 분배

    utils.set_model_mode('train', models=models_sd)
    utils.set_model_mode('train', models=models_td)
    # model 학습으로 설정

    models_sd = nn.Sequential(*models_sd)
    # model안의 encoder, head, classifer 순차적으로 입력
    models_td = nn.Sequential(*models_td)
    # model안의 encoder, head, classifer 순차적으로 입력

    for step, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        
###################### Image, Label ##########################
###############            (2)                ################
##############################################################
        src_imgs, src_labels = src_data
        tgt_imgs, tgt_labels = tgt_data
        src_imgs, src_labels = src_imgs.cuda(non_blocking=True), src_labels.cuda(non_blocking=True)
        tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)
        # 각각 input과 output 받기

###################### Model(input) ##########################
###############            (3)                ################
##############################################################
        x_sd, x_td = models_sd(tgt_imgs), models_td(tgt_imgs)
        # 각각의 model에 데이터를 입력하고 출력받기


######################    Baseline   #########################
###############           pseudo-label            ############
##############################################################
# -> utils.py
        pseudo_sd, top_prob_sd, threshold_sd = utils.get_target_preds(args, x_sd)
        # source data입력
        # softmax의 최대값에 해당하는 label, 확률값, threshold(mean-2.0*std)

######################### Mixup ##############################
###############            (a)                ################
##############################################################
# -> utils.py
        fixmix_sd_loss = utils.get_fixmix_loss(models_sd, src_imgs, tgt_imgs, src_labels, pseudo_sd, args.lam_sd)
        # mixed_x = 0.7 * x(src) + (1-0.7) * x(tar) 
        # mixed_x = models_sd(mixed_x)
        # loss = 0.7 * (mixed_x와 src_label의 ce) + (1-0.7) * (mixed_x와 target img의 model prediction_label 값의 ce)
        # (a) Loss : 람다 0.7
        
        pseudo_td, top_prob_td, threshold_td = utils.get_target_preds(args, x_td)
        # target data입력
         # softmax의 최대값에 해당하는 label, 확률값, threshold(mean-2.0*std)

        fixmix_td_loss = utils.get_fixmix_loss(models_td, src_imgs, tgt_imgs, src_labels, pseudo_td, args.lam_td)
        # 위와 동일한데 target dominant로 계산
        # (a) Loss : 람다 0.3

        total_loss = fixmix_sd_loss + fixmix_td_loss
        # (a) total Loss

        if step == 0:
            print('Fixed MixUp Loss (SDM): {:.4f}'.format(fixmix_sd_loss.item()))
            print('Fixed MixUp Loss (TDM): {:.4f}'.format(fixmix_td_loss.item()))
            # 각 (a)에 대한 Loss
            

################  Bidirectional Matching  ####################
###############            (b-1)                ##############
##############################################################
        # Bidirectional Matching
        if epoch > args.bim_start:
            # 100번 학습 이후, bim lodd 시작.
            bim_mask_sd = torch.ge(top_prob_sd, threshold_sd)
            # 원소간 비교하여 앞의 값이 뒤의 값보다 크면 True 즉, if top_prob_sd > threshold_sd
            # softmax의 confidence값이 threshold_sd보다 크거나 같으면 True
            bim_mask_sd = torch.nonzero(bim_mask_sd).squeeze()
            # nonzero의 index
            bim_mask_td = torch.ge(top_prob_td, threshold_td)
            # target에 대해서도 동일하게 진행
            bim_mask_td = torch.nonzero(bim_mask_td).squeeze()
            # target에 대해서도 동일하게 진행
            # (b)에서 confidence score를 구하는 과정
            if bim_mask_sd.dim() > 0 and bim_mask_td.dim() > 0:
            # dim이 0보다 크다 :
                if bim_mask_sd.numel() > 0 and bim_mask_td.numel() > 0:
                    # 원소의 갯수 : threshold보다 큰 원소가 있다.
                    bim_mask = min(bim_mask_sd.size(0), bim_mask_td.size(0))
                    # sd에서와 td에서의 선택된 데이터가 동일하게 되게 조정.
                    bim_sd_loss = ce(x_sd[bim_mask_td[:bim_mask]], pseudo_td[bim_mask_td[:bim_mask]].cuda().detach())
                    # threshold를 넘어선 값들 중 :target img의 model prediction과 tdm의 pseudo-label ce
                    bim_td_loss = ce(x_td[bim_mask_sd[:bim_mask]], pseudo_sd[bim_mask_sd[:bim_mask]].cuda().detach())
                    # threshold를 넘어선 값들 중 : ''와 sdm의 ce
                    # (b) bidrectional loss
                    total_loss += bim_sd_loss
                    total_loss += bim_td_loss
                    # (b) bidrectional loss
                    if step == 0:
                        print('Bidirectional Loss (SDM): {:.4f}'.format(bim_sd_loss.item()))
                        print('Bidirectional Loss (TDM): {:.4f}'.format(bim_td_loss.item()))


################### Self-penalization   ######################
###############            (b-2)                ##############
##############################################################
        if epoch <= args.sp_start:
            # 100번 학습하기 전
            sp_mask_sd = torch.lt(top_prob_sd, threshold_sd)
            # 뒤에값이 더 크면 True ↔ torch.ge와 반대
            sp_mask_sd = torch.nonzero(sp_mask_sd).squeeze()
            # threshold보다 작은 값 index
            sp_mask_td = torch.lt(top_prob_td, threshold_td)
            # target에 대해서도 동일하게 진행
            sp_mask_td = torch.nonzero(sp_mask_td).squeeze()
            # target에 대해서도 동일하게 진행
            if sp_mask_sd.dim() > 0 and sp_mask_td.dim() > 0:
                # dim이 0보다 크다 : threshold가 더 크다
                if sp_mask_sd.numel() > 0 and sp_mask_td.numel() > 0:
                    # threshold가 더 크다
                    sp_mask = min(sp_mask_sd.size(0), sp_mask_td.size(0))
                    sp_sd_loss = utils.get_sp_loss(x_sd[sp_mask_sd[:sp_mask]], pseudo_sd[sp_mask_sd[:sp_mask]], sp_param_sd)
                    # 스스로의 1-softmax(input)+smoothing_param의 확률분포와 ce
                    sp_td_loss = utils.get_sp_loss(x_td[sp_mask_td[:sp_mask]], pseudo_td[sp_mask_td[:sp_mask]], sp_param_td)
                    # 스스로의 1-softmax(input)+smoothing_param의 확률분포와 ce
                    # smoothing 변수를 준 값으로 빼서 loss를 조정하면 해당 값에 대해서 더 잘 학습.
                    total_loss += sp_sd_loss
                    total_loss += sp_td_loss
                    # (b) : self-penalization
                    if step == 0:
                        print('Penalization Loss (SDM): {:.4f}', sp_sd_loss.item())
                        print('Penalization Loss (TDM): {:.4f}', sp_td_loss.item())


###################  consistency Loss  #######################
###############            (c)                ################
##############################################################
        if epoch > args.cr_start:
            # 100번 학습이후
            mixed_cr = 0.5 * src_imgs + 0.5 * tgt_imgs
            # midean lambda를 사용하여
            out_sd, out_td = models_sd(mixed_cr), models_td(mixed_cr)
            # 각 모델의 output을 출력하고
            cr_loss = mse(out_sd, out_td)
            # mse 최소화
            total_loss += cr_loss
            # (c) : consistency Loss
            if step == 0:
                print('Consistency Loss: {:.4f}', cr_loss.item())

        optimizer_sd.zero_grad()
        optimizer_td.zero_grad()
        total_loss.backward()
        optimizer_sd.step()
        optimizer_td.step()

    print("Train time: {:.2f}".format(time.time() - start))
