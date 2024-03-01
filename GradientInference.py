import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
import numpy as np
from networks.EBM import ECBM
from argparse import ArgumentParser,Namespace
from data.data_util import get_dataset
import torch.nn as nn
import logging
import time
from utils import *
import json
from data.cub import CONCEPT_GROUP_MAP


class GradientInference():
    # define hyperparams here.
    def __init__(self,model,tolerate,args):
        super(GradientInference, self).__init__()
        self.model=model
        self.stop_criteria=EarlyStopping(patience=tolerate)
        self.metrics=MetricCalculator()
        if args.intervene_type!='None':
            self.metrics_intervene=MetricCalculator()
        self.n_classes=args.n_classes
        self.cpt_size=args.cpt_size
        self.lambda_cy=args.lambda_cy
        self.lambda_xy=args.lambda_xy
        self.lambda_xc=args.lambda_xc
        self.lr_c=args.lr_c
        self.lr_y=args.lr_y
        self.missingratio=args.missingratio
        print("Missing ratio",self.missingratio)
        self.model_state_dict=args.trained_weight
        self.intervene_type=args.intervene_type

    
    def run_optim(self,x,y,c,optim,use_cy,lambda_cy,lambda_xy,lambda_xc):
        with torch.enable_grad():
            running=True
            while(running):
                self.model.eval()
                optim.zero_grad()
                energy=self.model(x,(c,y),False,use_cy=use_cy)
                xy_en,cy_en,c_en,prob=energy
                cpt_loss=torch.zeros([]).to('cuda:0')
                for i in range(c_en.shape[1]):
                    c_en_per_con=c_en[:,i,:]
                    predL = c_en_per_con.mean()
                    cpt_loss+=predL
                loss=lambda_xy*xy_en.mean()+lambda_xc*cpt_loss+lambda_cy*cy_en.mean()
                # idx+=1
                self.stop_criteria(xy_en.mean(),self.model.state_dict())
                loss.backward(retain_graph=True)
                optim.step()
                y_prob,c_prob=prob
                y_prob_inf=y_prob.clone().detach()
                c_prob_inf=c_prob.clone().detach()
                _, met_cy = torch.max(y_prob_inf, 1)
                met_cy=met_cy.squeeze(-1)
                _, met_c = torch.max(c_prob_inf, 2)
                met_c=met_c.squeeze(-1)
                if self.stop_criteria.early_stop:
                    running=False
                    
                
        with torch.no_grad():
            # self.model.load_state_dict(self.stop_criteria.best_param)
            self.model.eval()
            energy=self.model(x,(c,y),False,use_cy=False)
        self.stop_criteria.reset()
        return energy


    def inference(self,x,y,c):

        self.model.load_dict(self.model_state_dict)
        y_prob=torch.zeros((x.size(0),self.n_classes,1)).cuda()
        c_prob=torch.zeros((x.size(0),self.cpt_size,2)).cuda()
        # print(c_prob[0])
        
        self.model.energy_model.y_prob=nn.Parameter(y_prob)
        self.model.energy_model.c_prob=nn.Parameter(c_prob)

        
        lambda_cy=float(self.lambda_cy)
        lambda_xy=float(self.lambda_xy)
        lambda_xc=float(self.lambda_xc)

        energy=None
        optim_list=[
            {'params': [self.model.energy_model.c_prob],'lr': self.lr_c},
            {'params': [self.model.energy_model.y_prob],'lr': self.lr_y},
        ]
        optim=torch.optim.Adam(optim_list)
        energy=self.run_optim(x,y,c,optim,use_cy=True,lambda_cy=lambda_cy,lambda_xy=lambda_xy,lambda_xc=lambda_xc)

        
        if self.intervene_type=='group':
            print("Start group intervention")
            len_group_concept=len(CONCEPT_GROUP_MAP)
            c_prob_intervene=self.model.energy_model.c_prob.clone().detach()
            # all_length=c.shape[-1]
            gt_in=int(len_group_concept*(1-self.missingratio))
            gt_in_idx=torch.tensor(select_concept_group(gt_in,len_group_concept)).cuda()
            intervene_c=torch.nn.functional.one_hot(c.long(),num_classes=2)
            intervine_c=(intervene_c-0.5)*10
            c_prob_intervene[:,gt_in_idx,:]=intervine_c[:,gt_in_idx,:]
            y_prob=torch.zeros((x.size(0),self.n_classes,1)).cuda()
            self.model.energy_model.y_prob=nn.Parameter(y_prob)
            self.model.energy_model.c_prob=nn.Parameter(c_prob_intervene)
            lambda_cy=float(3)
            lambda_xy=float(0)
            lambda_xc=float(0)
            optim_list=[
            {'params': [self.model.energy_model.c_prob],'lr': self.lr_c},
            {'params': [self.model.energy_model.y_prob],'lr': self.lr_y},
            ]
            optim=torch.optim.Adam(optim_list)
            # Step 2
            energy_after_intv=self.run_optim(x,y,c,optim,use_cy=True,lambda_cy=lambda_cy,lambda_xy=lambda_xy,lambda_xc=lambda_xc)
        elif self.intervene_type=='individual':
            c_prob_intervene=self.model.energy_model.c_prob.clone().detach()
            all_length=c.shape[-1]
            gt_in=all_length*(1-self.missingratio)
            gt_in_idx=torch.tensor(generate_random_numbers(gt_in,all_length)).cuda()
            intervene_c=torch.nn.functional.one_hot(c.long(),num_classes=2)
            intervine_c=(intervene_c-0.5)*10
            c_prob_intervene[:,gt_in_idx,:]=intervine_c[:,gt_in_idx,:]
            y_prob=torch.zeros((x.size(0),self.n_classes,1)).cuda()
            self.model.energy_model.y_prob=nn.Parameter(y_prob)
            self.model.energy_model.c_prob=nn.Parameter(c_prob_intervene)
            lambda_cy=float(3)
            lambda_xy=float(0)
            lambda_xc=float(0)
            optim_list=[
            {'params': [self.model.energy_model.c_prob],'lr': self.lr_c},
            {'params': [self.model.energy_model.y_prob],'lr': self.lr_y},
            ]
            optim=torch.optim.Adam(optim_list)
            # Step 2
            energy_after_intv=self.run_optim(x,y,c,optim,use_cy=True,lambda_cy=lambda_cy,lambda_xy=lambda_xy,lambda_xc=lambda_xc)
        else:
            energy_after_intv=None
        

        return (energy_after_intv,energy)
    

        

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--random_seed", default=42)
    parser.add_argument("--dataset",default="cub")
    parser.add_argument("--freezebb",default=False)
    parser.add_argument("--backbone",default='resnet101_imagenet')
    parser.add_argument("--device",default='cuda:0')
    parser.add_argument("--exp_dir",type=str,default='.')
    parser.add_argument("--n_classes",default=200)
    parser.add_argument("--emb_size",default=2048)
    parser.add_argument("--cpt_size",default=112)
    parser.add_argument("--hid_size",default=512)
    parser.add_argument("--img_size",default=299)
    parser.add_argument("--momentum",default=0.9,type=float)
    parser.add_argument("--beta_1",default=0.9,type=float)
    parser.add_argument("--beta_2",default=0.9,type=float)
    parser.add_argument("--weight_decay",default=4e-5,type=float)
    parser.add_argument("--pretrained",default=True)
    parser.add_argument("--cy_permute_prob",type=float,default=0.2)
    parser.add_argument("--cy_perturb_prob",type=float,default=0.2)
    parser.add_argument("--learning_rate",default=1e-2,type=float)
    parser.add_argument("--epochs",default=300,type=int)
    parser.add_argument("--lr_c",default=1e-1)
    parser.add_argument("--lr_y",default=1e-1)
    parser.add_argument("--lambda_xy",default=1)
    parser.add_argument("--lambda_xc",default=1)
    parser.add_argument("--lambda_cy",default=0.01)
    parser.add_argument("--tolerence",default=10)
    parser.add_argument("--batch_size",default=64)
    parser.add_argument("--intervene_type", type= str, choices = ['group', 'individual'], default= 'None')
    parser.add_argument("--missingratio",type=float,default=0.0)
    parser.add_argument("--trained_weight",default='.')
    args = parser.parse_args()
    #TODO: update config

    
    dataset=args.dataset
    args = vars(args)
    with open(f'./configs/{dataset}_inference.json') as f:
        args.update(json.load(fp=f))
    args=Namespace(**args)


    uuid = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    exp_dir=os.path.join(args.exp_dir,f'{uuid}-{args.dataset}-{args.backbone}')
    if not os.path.exists(exp_dir): os.mkdir(os.path.join(args.exp_dir,f'{uuid}-{args.dataset}-{args.backbone}'))

    logging.basicConfig(filename=exp_dir+'/log.txt',
                     format = '%(message)s',
                     level=logging.INFO)

    logging.info(args)

    network=ECBM(args,args.emb_size,args.cpt_size,args.hid_size,n_classes=args.n_classes)
    network=network.cuda()
    inferer=GradientInference(network,args.tolerence,args)
    train_loader,val_loader, test_loader=get_dataset(args)

    if args.intervene_type!='None':
        logging.info("==============Running Interventions============")
        logging.info("Missing ratio:{}".format(args.missingratio))

    for idx,i in enumerate(test_loader):
        if args.dataset=="celeba":
            x,cy=i
            y,c=cy
        else:
            x,y,c=i
        x,y,c=x.cuda(),y.cuda(),c.cuda()
        out=inferer.inference(x,y,c)
        energy_after_intv,energy_before=out
        
        if energy_after_intv is not None:
            print('='*5,"After Intervention",'='*5)
            xy_en,cy_en,c_en,prob=energy_after_intv
            # print(prob)
            y_acc,c_acc_overall,c_acc=inferer.metrics_intervene.update(prob,(c,y))
            print(y_acc,c_acc_overall,c_acc)
        
        if energy_before is not None:
            print('='*5,"Before Intervention",'='*5)
            xy_en,cy_en,c_en,prob=energy_before
            # print(prob)
            y_acc,c_acc_overall,c_acc=inferer.metrics.update(prob,(c,y))
            print(y_acc,c_acc_overall,c_acc)
            
    logging.info("=============Final result================")
    
    if energy_before is not None:
        logging.info("=============Gradient Inderence================")
        y_acc,c_acc_overall,c_acc=inferer.metrics.return_metrics()

        logging.info('concept_accuracy-{}'.format(c_acc))
        logging.info('concept_overall_accuracy-{}'.format(c_acc_overall))
        logging.info('y_accuracy-{}'.format(y_acc))
        y_pred,y_gt,c_pred,c_gt=inferer.metrics.get_data()
        np.save(os.path.join(exp_dir,'y_pred.npy'),y_pred.cpu().numpy())
        np.save(os.path.join(exp_dir,'y_gt.npy'),y_gt.cpu().numpy())
        np.save(os.path.join(exp_dir,'c_pred.npy'),c_pred.cpu().numpy())
        np.save(os.path.join(exp_dir,'c_gt.npy'),c_gt.cpu().numpy())

    if energy_after_intv is not None:
        logging.info("=============After Intervention================")
        y_acc,c_acc_overall,c_acc=inferer.metrics_intervene.return_metrics()

        logging.info('concept_accuracy-{}'.format(c_acc))
        logging.info('concept_overall_accuracy-{}'.format(c_acc_overall))
        logging.info('y_accuracy-{}'.format(y_acc))
