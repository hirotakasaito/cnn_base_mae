import os
import json
from pathlib import Path
from collections import namedtuple, OrderedDict
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
from torch.utils.data import DataLoader,random_split
from tensorboardX import SummaryWriter
import torchvision
from tqdm import tqdm
import time
import random

from util.prepare_output_dir import prepare_output_dir
from util.dataset import BaseDataset
from util.visualize import get_concat_h_multi, get_concat_v
from model import *
from pytorch_ssim import SSIM

def arg_parse():
    parser = argparse.ArgumentParser(description='')

    #base param
    parser.add_argument("--epoch",type=int,default=1500)
    parser.add_argument("--test-interval",type=int,default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument('--cuda', type=int, default=0, help='cuda number if -1, you can use CPU')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='number of cpu')
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--multi-gpu",type=str,default = True)
    parser.add_argument("--log-image-interval",type=int,default = 2)
    parser.add_argument("--size",type=int,default = 64)

    #learning rate
    parser.add_argument("--encoder-lr",type=float,default=1e-4)
    parser.add_argument("--obs_lr",type=float,default=1e-4)
    parser.add_argument("--loss-type",type=str,default="mse")

    #dir setting
    parser.add_argument('--log-dir', type=str, default='logs/attention/')
    parser.add_argument('--save-dir', type=str, default='weights')
    parser.add_argument('--logs-name',type=str,default='local')
    parser.add_argument('--dataset-dir', type=str, default='/share/private/27th/hirotaka_saito/Images/test2/')
    parser.add_argument('--pretrained-dir', type=str, default=None)

    #size
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--encoder-hidden-size",type=int,default=64)
    parser.add_argument("--observation-size",type=int,default=256)
    parser.add_argument("--embedded-img",type=int,default=256)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parse()

    observation_size = [3, args.observation_size, args.observation_size]
    result_output_path = Path(prepare_output_dir(args, args.log_dir))
    print(result_output_path)
    save_dir = result_output_path / args.save_dir #/ is connecting path
    result_output_path = str(result_output_path)
    save_dir = str(save_dir)
    print(save_dir)
    os.makedirs(save_dir)

    writer = SummaryWriter(os.path.join(result_output_path, "summary"))#tensorboard is displayed

    if args.cuda < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:'+str(args.cuda))
        print("GPU使用")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = os.path.join(os.path.dirname(__file__), args.dataset_dir)

    dataset = BaseDataset(dataset_dir = dataset_dir)

    train_num = int(0.9*len(dataset))
    test_num = len(dataset) -train_num

    print("train_num: %d" % train_num)
    print("test_num : %d" % test_num)
    train_data, test_data = random_split(dataset, [train_num, test_num])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers,pin_memory=True)

    pi = torch.acos(torch.zeros(1)).item() * 2

    # encoder = ResNetImageEncoder(
    #         observation_size = observation_size,
    #         embedding_size = args.embedding_size,
    #         hidden_size = args.encoder_hidden_size,
    #         ).to(device)

    # encoder = MultiAttentionNetwork(
    #         observation_size = observation_size,
    #         embedding_size = args.embedding_size,
    #         hidden_size = args.encoder_hidden_size,
    #         ).to(device)

    encoder = MultiAttentionNetworkWithVgg16(
            observation_size = observation_size,
            embedding_size = args.embedding_size,
            hidden_size = args.encoder_hidden_size,
            ).to(device)

    # encoder = Vgg16ImageEncoder(
    #         observation_size = observation_size,
    #         embedding_size = args.embedding_size,
    #         hidden_size = args.encoder_hidden_size,
    #         ).to(device)

    obs_model = ImageDecoder(
            observation_size = observation_size,
            embedded_obs = args.embedded_img,
            embedding_size = args.encoder_hidden_size,
            ).to(device)

    trans = torchvision.transforms.ToPILImage()
    ssim = SSIM()

    optimizer = optim.AdamW(params=[
        {"params": encoder.parameters(), "lr": args.encoder_lr},
        {"params": obs_model.parameters(), "lr": args.obs_lr},
    ], eps=args.eps,weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.obs_lr, max_lr=args.obs_lr*10,cycle_momentum=False)

    if args.multi_gpu:
        encoder = torch.nn.DataParallel(encoder)
        obs_model = torch.nn.DataParallel(obs_model)

    if args.pretrained_dir is not None:
        encoder.load_state_dict(torch.load(
            os.path.join(args.pretrained_dir, 'weights', 'encoder.pkl')))
        obs_model.load_state_dict(torch.load(
            os.path.join(args.pretrained_dir, 'weights', 'obs_model.pkl')))

    best_loss = np.inf
    num_update = 0
    size = args.size
    torch.backends.cudnn.benchmark = True
    random_mask = np.array([[96,96],[96,48],[96,144],[48,48],[48,96],[48,144],[144,48],[144,96],[144,144]])
    # mask = img[:,96:160,96:160,:]*0 + 0.9

    with tqdm(range(args.epoch)) as pbar:
        for epoch in pbar:
            encoder.train()
            obs_model.train()

            scaler = torch.cuda.amp.GradScaler()

            train_loss = 0.0
            train_img_loss = 0.0
            num_update=0

            for idx, data in enumerate(train_loader):
                img = data
                img = img.permute(0,2,3,1).to(device,non_blocking=True)
                _img = img.clone()

                # mask = img[:,64:192,64:192,:]*0 + 0.9
                i = random.randint(0,8)
                mask = img[:,random_mask[i][0]:random_mask[i][0]+size,random_mask[i][1]:random_mask[i][1]+size,:]*0 + 0.9
                mask_img = img
                # mask_img[:,64:192,64:192,:]= mask
                mask_img[:,random_mask[i][0]:random_mask[i][0]+size,random_mask[i][1]:random_mask[i][1]+size,:]= mask
                # clipping_img = _img[:,64:192,64:192,:]
                clipping_img = _img[:,random_mask[i][0]:random_mask[i][0]+size,random_mask[i][1]:random_mask[i][1]+size,:]
                embedded_img, attn = encoder(mask_img)

                # recon_img = obs_model(embedded_img)

                # img_loss = mse_loss(recon_img, img,reduction='none').mean([0]).sum()
                # img_loss = binary_cross_entropy_with_logits(recon_img, img,reduction='none').mean([0]).sum()
                # loss = img_loss
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    recon_img = obs_model(embedded_img)
                    if args.loss_type == "mse":
                        # mse_loss_img = mse_loss(recon_img, clipping_img,reduction='none').mean([0]).sum()
                        mse_loss_img = mse_loss(recon_img, clipping_img,reduction='none').mean([0]).sum()
                        loss = mse_loss_img
                    else:
                        ssim_loss = 1 - ssim(recon_img, clipping_img)
                        loss = ssim_loss

                scaler.scale(loss).backward()
                # loss.backward()
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()

                train_loss += loss.detach()
                num_update += 1

                # del loss
                del img
                del _img
                del mask_img
            scheduler.step()

            writer.add_scalar('cnn_base_mae/train/loss', train_loss/num_update, epoch)
            # writer.add_scalar('rssm/train/img_loss', train_img_loss/num_update, epoch)
            pbar.set_postfix(OrderedDict(Loss=train_loss/num_update))

            # test

            if (epoch+1) % args.test_interval == 0:
                encoder.eval()
                obs_model.eval()

                with torch.no_grad():
                    test_loss = 0.0
                    test_img_loss = 0.0
                    num_update = 0
                    mask_img_list = []
                    img_list = []
                    concat_img_list = []
                    attn_list = []
                    for idx, data in enumerate(test_loader):
                        img = data
                        img = img.permute(0,2,3,1).to(device,non_blocking=True)
                        _img = img.clone()
                        # clipping_img = _img[:,64:192,64:192,:]
                        # clipping_img = _img[:,96:160,96:160,:]
                        i = random.randint(0,8)
                        clipping_img = _img[:,random_mask[i][0]:random_mask[i][0]+size,random_mask[i][1]:random_mask[i][1]+size,:]
                        # mask = img[:,64:192,64:192,:]*0 + 0.9
                        mask = img[:,random_mask[i][0]:random_mask[i][0]+size,random_mask[i][1]:random_mask[i][1]+size,:]*0 + 0.9
                        # mask = img[:,96:160,96:160,:]*0 + 0.9

                        mask_img = img
                        # mask_img[:,64:192,64:192,:]= mask
                        mask_img[:,random_mask[i][0]:random_mask[i][0]+size,random_mask[i][1]:random_mask[i][1]+size,:]= mask
                        # mask_img[:,96:160,96:160,:]= mask
                        embedded_img, attn = encoder(mask_img)

                        recon_img = obs_model(embedded_img)

                        # img_loss = mse_loss(recon_img, clipping_img,reduction='none').mean([0]).sum()
                        img_loss = mse_loss(recon_img, clipping_img,reduction='none').mean([0]).sum()
                        # img_loss = binary_cross_entropy_with_logits(recon_img, img)
                        loss = img_loss
                        test_loss += loss.detach()
                        num_update += 1
                        #for visualize img
                        _vimg = _img.detach().to('cpu')
                        _attn = attn.detach().to('cpu')
                        _concat_img = mask_img.detach().to('cpu')
                        _mask_img = mask_img.detach().to('cpu')
                        _recon_img = recon_img.to('cpu')

                        _vimg = trans(_vimg[0].permute(2,0,1))
                        _attn = trans(_attn[0,0])
                        _attn = _attn.resize((256,256))
                        # _concat_img[:,64:192,64:192,:] = _recon_img
                        _concat_img[:,random_mask[i][0]:random_mask[i][0]+size,random_mask[i][1]:random_mask[i][1]+size,:] = _recon_img
                        _concat_img = trans(_concat_img[0].permute(2,0,1))
                        _mask_img = trans(_mask_img[0].permute(2,0,1))

                        img_list.append(_vimg)
                        attn_list.append(_attn)
                        mask_img_list.append(_mask_img)
                        concat_img_list.append(_concat_img)

                        del loss
                        del img
                        del _img
                        del recon_img
                        del mask_img

                        if idx == 5:
                            # ob = trans(ob[[2,1,0],:,:])
                            # recon_ob = trans(recon_ob[[2,1,0],:,:])
                            attn_imgs = get_concat_h_multi(attn_list)
                            trust_imgs = get_concat_h_multi(img_list)
                            mask_imgs = get_concat_h_multi(mask_img_list)
                            concat_imgs = get_concat_h_multi(concat_img_list)
                            output_img = get_concat_v(trust_imgs,mask_imgs,concat_imgs,attn_imgs)
                            # output_trust_img_np = np.asarray(trust_imgs).transpose(2,0,1)
                            # output_mask_img_np = np.asarray(mask_imgs).transpose(2,0,1)
                            # output_concat_img_np = np.asarray(concat_imgs).transpose(2,0,1)
                            output_img_np = np.asarray(output_img).transpose(2,0,1)
                            # writer.add_image('rssm/test/trust_img', output_trust_img_np, epoch)
                            # writer.add_image('rssm/test/mask_img', output_mask_img_np, epoch)
                            # writer.add_image('rssm/test/concat_img', output_concat_img_np, epoch)
                            writer.add_image('cnn_base_mae/test/outpu_img', output_img_np, epoch)
                            img_list.clear()
                            mask_img_list.clear()
                            concat_img_list.clear()

                writer.add_scalar('cnn_base_mae/test/loss', test_loss/num_update, epoch)

                # save model
                if args.multi_gpu:
                    if test_loss/num_update <= best_loss:
                        best_loss = test_loss/num_update
                        torch.save(encoder.module.state_dict(), os.path.join(save_dir, 'encoder.pkl'))
                        torch.save(obs_model.module.state_dict(), os.path.join(save_dir, 'obs_model.pkl'))
                    torch.save(encoder.module.state_dict(), os.path.join(save_dir, 'encoder_final.pkl'))
                    torch.save(obs_model.module.state_dict(), os.path.join(save_dir, 'obs_model_final.pkl'))

                else:
                    if test_loss/num_update <= best_loss:
                        best_loss = test_loss/num_update
                        torch.save(encoder.state_dict(), os.path.join(save_dir, 'encoder.pkl'))
                        torch.save(obs_model.state_dict(), os.path.join(save_dir, 'obs_model.pkl'))

                    torch.save(encoder.state_dict(), os.path.join(save_dir, 'encoder_final.pkl'))
                    torch.save(obs_model.state_dict(), os.path.join(save_dir, 'obs_model_final.pkl'))
        writer.close()
