import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from LitModel import LitModel
from pytorch_lightning import Trainer,seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
import time
import torch
import json
from data.data_util import get_dataset
from argparse import ArgumentParser,Namespace




parser = ArgumentParser()
parser.add_argument("--random_seed", default=42)
parser.add_argument("--dataset",default="cub",choices=['cub','awa2','celeba'])
parser.add_argument("--freezebb",default=False)
parser.add_argument("--gpu",type=int,default=0)
parser.add_argument("--backbone",default='resnet101_imagenet')
parser.add_argument("--device",default='cuda:0')
parser.add_argument("--dl_dir",type=str,default='./exp')
parser.add_argument("--exp_dir",type=str,default='.')
parser.add_argument("--n_classes",default=200)
parser.add_argument("--emb_size",default=2048)
parser.add_argument("--cpt_size",default=112)
parser.add_argument("--hid_size",default=512)
parser.add_argument("--img_size",default=299)
parser.add_argument("--lambda_xy",default=3,type=int)
parser.add_argument("--lambda_xc",default=1,type=int)
parser.add_argument("--lambda_cy",default=1,type=int)
parser.add_argument("--momentum",default=0.9,type=float)
parser.add_argument("--beta_1",default=0.9,type=float)
parser.add_argument("--beta_2",default=0.9,type=float)
parser.add_argument("--weight_decay",default=4e-5,type=float)
parser.add_argument("--pretrained",default=True)
parser.add_argument("--sweep",default=False)
parser.add_argument("--optim",default='sgd')
parser.add_argument("--cy_permute_prob",type=float,default=0.2)
parser.add_argument("--cy_perturb_prob",type=float,default=0.2)
parser.add_argument("--learning_rate",default=1e-2,type=float)
parser.add_argument("--epochs",default=300,type=int)
parser.add_argument("--batch_size",default=64,type=int)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

dataset=args.dataset

args = vars(args)
with open(f'./configs/{dataset}.json') as f:
    args.update(json.load(fp=f))
args=Namespace(**args)

if __name__ == '__main__':

    torch.use_deterministic_algorithms(True)
    seed_everything(args.random_seed, workers=True)
    uuid = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    exp_dir = os.path.join("./exp", uuid)
    args.exp_dir = exp_dir
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)
    # load data, return train-val split, traindataset, and test dataloaders
    train_loader,val_loader, test_loader=get_dataset(args)
    args.max_epochs=args.epochs
    # configure saving critiria
    monitor_="epoch_val_loss" 
    mode_="min"
    #define model and all procedure
    model = LitModel(exp_dir,args)
    #define a trainer
    trainer = Trainer.from_argparse_args(args,devices=[args.gpu],
    default_root_dir=exp_dir,
    num_sanity_val_steps=0,
    callbacks=[
                    ModelCheckpoint(dirpath=args.exp_dir, save_top_k=5, save_last=True, monitor=monitor_,mode=mode_)
                ]
    )
    # wandb_logger.watch(model,log_graph=False)
    trainer.fit(model,train_dataloaders=train_loader, val_dataloaders=val_loader)
    model.freeze()
    trainer.test(model=model,ckpt_path=None,dataloaders=test_loader)