import os
import torch


class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        # python train_wavelet.py --warmup --train_ps 320 \
        # --batch_size 4 \
        # --nepoch 1000 \
        # --gpu '0' \
        # --env env_name \
        # --save_dir ./logs/ \
        # --train_dir path/to/train/data/ \
        # --val_dir path/to/val/data/ \
        # --lr_initial 0.0004

        # dataset dir
        istd_train_dir = '/home/nan/dataset/SISR_DATASET/ISTD_Dataset/train'
        istd_val_dir = '/home/nan/dataset/SISR_DATASET/ISTD_Dataset/test'
        aistd_train_dir = '/home/nan/dataset/SISR_DATASET/ISTD_adjusted/train'
        aistd_val_dir = '/home/nan/dataset/SISR_DATASET/ISTD_adjusted/test'
        srd_train_dir = '/home/nan/dataset/SISR_DATASET/SRD/train'
        srd_val_dir = '/home/nan/dataset/SISR_DATASET/SRD/test'

        # global settings
        parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        parser.add_argument('--nepoch', type=int, default=1200, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=8, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=8, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default='SRD')
        parser.add_argument('--pretrain_weights', type=str, default='',
                            help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0004, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')
        parser.add_argument('--arch', type=str, default='Wave_UNet22_wo_fdc2', help='archtechture')
        parser.add_argument('--mode', type=str, default='shadow', help='image restoration mode')

        # args for saving
        parser.add_argument('--save_dir', type=str, default='./log', help='save dir')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='_srd', help='env')
        parser.add_argument('--checkpoint', type=int, default=200, help='checkpoint')

        # args for Uformer
        parser.add_argument('--embed_dim', type=int, default=64, help='dim of emdeding features')

        # args for training
        parser.add_argument('--train_ps', type=int, default=384, help='patch size of training sample')
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--train_dir', type=str, default=srd_train_dir, help='dir of train data')
        parser.add_argument('--val_dir', type=str, default=srd_val_dir, help='dir of train data')
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')

        return parser
