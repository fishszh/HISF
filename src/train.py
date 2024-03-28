import os, sys
import time
from tqdm import tqdm
working_dir = ''
sys.path.insert(0, working_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from config import Config

import pathlib, glob
import numpy as np
import torch
import torch.nn as nn
import torchio as tio
# from torchkeras import KerasModel
from torch.utils.tensorboard import SummaryWriter

from dataset import HICropDataSet, HICubeDataSet
from loss import MixLoss, ELLoss
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.swin_unetr import SwinUNETR
from networks.unet_lk import UNetLK
from networks.uxnet_3d import UXNET
from metrics import Metrics
from utils import demo_predict_cube_mask



def main():
    cfg = Config()
    if not os.path.exists(f'./data/log/{cfg.nowtime}'):
        os.makedirs(f"./data/log/{cfg.nowtime}")

    with open(f"./data/log/{cfg.nowtime}/config.py", 'a+') as f:
        with open('./src/config.py') as cf:
            f.writelines(cf.readlines())
    # dataset
    if cfg.input_mode == 'Cube':
        ds_train = HICubeDataSet(cfg.batch_size, cfg.num_samples_per_cube, cfg.target_shape, True)
        ds_valid = HICubeDataSet(cfg.batch_size, cfg.num_samples_per_cube, cfg.target_shape, False)
        dl_train = HICubeDataSet.get_dataloader(
                        ds_train, 
                        cfg.batch_size, 
                        shuffle=True, 
                        num_workers=cfg.num_workers)
        dl_valid = HICubeDataSet.get_dataloader(
                        ds_valid,
                        cfg.batch_size,
                        shuffle     = True,
                        num_workers = cfg.num_workers)
    elif cfg.input_mode == 'Crop':
        ds_train = HICropDataSet(cfg.batch_size, cfg.num_samples_per_cube, cfg.crop_size, True)
        ds_valid = HICropDataSet(cfg.batch_size, cfg.num_samples_per_cube, cfg.crop_size, False)
        dl_train = HICropDataSet.get_dataloader(
                        ds_train, 
                        cfg.batch_size, 
                        shuffle=True, 
                        num_workers=cfg.num_workers)
        dl_valid = HICropDataSet.get_dataloader(
                        ds_valid,
                        cfg.batch_size,
                        shuffle     = True,
                        num_workers = cfg.num_workers)
    
    # Initialize model
    if cfg.model_name == "UNetLK":
        net = UNetLK(cfg.in_chans, cfg.num_classes, 32)
    elif cfg.model_name == "UXNet":
        net = UXNET(
            in_chans=1,
            out_chans=1,
            depths=[2,2,2,2],
            feat_size=[16,32,64,128],
            drop_path_rate=0,
            layer_scale_init_value=0.,
            hidden_size=768,
            spatial_dims=3
        )
    elif cfg.model_name == "unetr":
        net =  UNETR(
            in_channels=1, 
            out_channels=1, 
            img_size=cfg.crop_size
        )
    elif cfg.model_name == "swin_unetr":
        net = SwinUNETR(
            in_channels=1,
            img_size=cfg.crop_size,
            out_channels=1,
        )
    if cfg.model_weights: 
        print(f"loading model weights from {cfg.path_weight}/{cfg.model_weights}...")
        model_weights = torch.load(cfg.path_weight + "/" + cfg.model_weights + ".pkl")
        net.load_state_dict(model_weights)
    model   = nn.DataParallel(net).cuda()
    # model = Model(net)
    
    loss_func = MixLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.init_lr)

    # tensorboard
    writer = SummaryWriter(cfg.path_tb)
    if not os.path.exists(cfg.path_tb):
        os.makedirs(cfg.path_tb)

    print("Start Training")
    nowtime = time.strftime("%Y-%m-%d %H:%M:%S")
    print("=========="*3 + f"{nowtime}")
    metrics_name = ['loss', 'dice']
    best_dice = 0
    pbar = tqdm(range(cfg.epochs_bg+1, cfg.epochs_end+1), desc="Total:", ncols=80)
    for epoch in pbar: 
        # 1. training ----------------------------------------------------------
        npe = len(dl_train)
        skip_step = npe // 10
        cfg.decay_step = npe
        metrics = dict(zip(metrics_name, [0] * len(metrics_name)))
        
        for step_train, (features, labels) in enumerate(dl_train, 1): 
            # print("features:",features.isnan().sum(), "labels:", labels.isnan().sum())
            metrics_cur = train_step(
                                cfg, model, optimizer, loss_func, 
                                features, labels, epoch, step_train)
            for mn in metrics_name:
                metrics[mn] = metrics[mn] + metrics_cur[mn]
            
            pbar.set_description(f"Total:{epoch}, train:{step_train}|{npe}, dice:{metrics_cur['dice']:.3f} ", True)
            

        writer.add_scalar(f"train/loss", metrics['loss']/npe, epoch)
        writer.add_scalar(f"train/dice", metrics['dice']/npe, epoch)
        
        if (metrics['dice']/npe < 1e-3) * (step_train > 0):
            print('Something wrong with dice = 0 !!!!!!!!!!!!!!!!!!!!!!!!')
            sys.exit()
        metrics["loss"] = 0
        metrics["dice"] = 0
        
        # 2. valid -------------------------------------------------------------
        npe = len(dl_valid)
        metrics =dict(zip(metrics_name, [0] * len(metrics_name)))

        for step_valid, (features, labels) in enumerate(dl_valid, 1):
            metrics_cur = valid_step(
                                cfg, model, optimizer, loss_func, 
                                features, labels, epoch, step_valid)
            for mn in metrics_name:
                metrics[mn] = metrics[mn] + metrics_cur[mn]

            pbar.set_description(f"Total: {epoch}, valid: {step_valid}|{npe}, dice:{metrics_cur['dice']:.3f} ", True)
        
        writer.add_scalar(f"valid/loss", metrics['loss']/npe, epoch)
        writer.add_scalar(f"valid/dice", metrics['dice']/npe, epoch)
        
        # recall = (metrics['TP'] + 1e-8) / (metrics['TP'] + metrics['FN'] + 1e-8)
        # precision = (metrics['TP'] + 1e-8) / (metrics['TP'] + metrics['FP'] + 1e-8)
        # f1 = 2*recall*precision/(recall+precision+1e-8)
        # writer.add_scalar(f"valid/recall", recall, epoch)
        # writer.add_scalar(f"valid/precision", precision, epoch)
        # writer.add_scalar(f"valid/f1", f1, epoch)

        cur_dice = metrics['dice'] / npe

        # 3. save model --------------------------------------------------------
        if not os.path.exists(cfg.path_weight):
            os.makedirs(cfg.path_weight)
        if epoch >= 2 and cur_dice > best_dice:
            best_dice = cur_dice
            filename = os.path.join(cfg.path_weight, f"best_model_weights-{epoch:04}-{best_dice:.3f}.pkl")
            torch.save(model.module.state_dict(), filename)    
        if epoch >= 2 and epoch % 2 == 0:
            filename = os.path.join(cfg.path_weight, f"model_weights-{epoch:04}.pkl")
            torch.save(model.module.state_dict(), filename)

        
        # clear saved model weights
        model_weights_list = glob.glob(f"{cfg.path_weight}/model*")
        model_weights_list.sort()
        for mw in model_weights_list[:-20]:
            os.remove(mw)
        
        demo_list = glob.glob(f"{cfg.path_demo}/*png")
        demo_list.sort()
        for demo in demo_list[:-10]:
            os.remove(demo)

    writer.close()
    pbar.close()            


def train_step(cfg, model, optimizer, loss_func, features, 
                labels, epoch, step_train):
    model.train()
    optimizer.zero_grad()

    features = features.cuda()
    labels   = labels.cuda()

    # forward
    predictions = model(features)
    loss = loss_func(predictions, labels)
    # loss = AsymmetricLossOptimized()(predictions, labels) + SigmoidFocalLoss()(predictions, labels)

    # evaluate metrics
    train_metrics = {'loss' : loss.item()}
    train_metrics["dice"] = Metrics.dice(predictions, labels).item()
    # TP, FN, FP = Metrics.counts(predictions, labels)
    # train_metrics["TP"] = TP 
    # train_metrics["FN"] = FN 
    # train_metrics["FP"] = FP 

    # backward
    loss.backward()

    # update parameters
    optimizer.step()
    optimizer.param_groups[0]['lr'] = decayed_lr(cfg, epoch)

    # demo
    if not os.path.exists(cfg.path_demo):
        os.makedirs(cfg.path_demo)
    if step_train % 50 == 0 and epoch>=2:
        feature = features[0,0].detach().cpu()
        label = labels[0,0].detach().cpu()
        prediction = predictions[0,0].sigmoid().detach().cpu()
        sub = tio.Subject(
            image = tio.ScalarImage(tensor=feature[None]),
            label = tio.LabelMap(tensor=label[None]),
            pred = tio.LabelMap(tensor=(prediction>0.5)[None])
        )
        sub.image.save(f"{cfg.path_demo}/train_{step_train}-image.nii.gz")
        sub.label.save(f"{cfg.path_demo}/train_{step_train}-label.nii.gz")
        sub.pred.save(f"{cfg.path_demo}/train_{step_train}-pred-label.nii.gz")
        # demo_predict(feature, label, prediction, 'train', epoch, step_train, cfg.path_demo)
        demo_predict_cube_mask(feature.numpy(), label.numpy(), prediction.numpy(), f'train', epoch, step_train, cfg.path_demo)


    return train_metrics


@torch.no_grad()
def valid_step(cfg, model, optimizer, loss_func, 
               features, labels, epoch, step_valid):
    model.eval()
    
    features = features.cuda()
    labels   = labels.cuda()
    
    predictions = model(features)
    loss = loss_func(predictions, labels)

    valid_metrics = {'loss' : loss.item()}
    valid_metrics["dice"] = Metrics.dice(predictions, labels).item()
    # TP, FN, FP = Metrics.counts(predictions, labels)
    # valid_metrics["TP"] = TP 
    # valid_metrics["FP"] = FP
    # valid_metrics["FN"] = FN

    # demo
    if step_valid % 50 == 0 and epoch>=2:
        feature = features[0,0].detach().cpu()
        label = labels[0,0].detach().cpu()
        prediction = predictions[0,0].sigmoid().detach().cpu()
        sub = tio.Subject(
            image = tio.ScalarImage(tensor=feature[None]),
            label = tio.LabelMap(tensor=label[None]),
            pred = tio.LabelMap(tensor=(prediction>0.5)[None])
        )
        sub.image.save(f"{cfg.path_demo}/valid_{step_valid}-image.nii.gz")
        sub.label.save(f"{cfg.path_demo}/valid_{step_valid}-label.nii.gz")
        sub.pred.save(f"{cfg.path_demo}/valid_{step_valid}-pred-label.nii.gz")
        # demo_predict(feature, label, prediction, 'valid', epoch, step_valid, cfg.path_demo)
        demo_predict_cube_mask(feature.numpy(), label.numpy(), prediction.numpy(), f'valid', epoch, step_valid, cfg.path_demo)

    return valid_metrics


def decayed_lr(cfg, epoch):
    epoch_max = max(epoch, cfg.epochs_end)
    return (1-epoch/epoch_max) ** 0.8 * (cfg.init_lr - cfg.end_lr) + cfg.end_lr


if __name__ == "__main__":
    main()