import pytorch_lightning as pl
import torch
from networks.EBM import ECBM
from loss import EBMLoss_label,EBMLoss_concept
from metrics import MetricComputer,AverageMeter
import sklearn
import logging
import os

class LitModel(pl.LightningModule):
    # define hyperparams here.
    def __init__(self,logdir,args):
        super().__init__()
        self.learning_rate=args.learning_rate
        self.epochs=args.epochs
        self.batch_size = args.batch_size
        self.optim=args.optim
        self.wd=args.weight_decay
        self.dataset=args.dataset

        if args.optim.lower()=='adam':
            self.beta_1=args.beta_1
            self.beta_2=args.beta_2    
        elif args.optim.lower()=='sgd':
            self.momentum=args.momentum

        self.logdir=logdir
        self.class_list=[i for i in range(args.n_classes)]
        self.concept_list=[i for i in range(args.cpt_size)]
        self.network=ECBM(args=args,emb_size=args.emb_size,cpt_size=args.cpt_size,hid_size=args.hid_size,n_classes=args.n_classes)
        self.loss_label = EBMLoss_label(self.class_list,device=args.device)
        self.loss_concept = EBMLoss_concept(self.concept_list,device=args.device)

        self.epoch_summary = {"Accuracy": AverageMeter()}
        self.computer = MetricComputer(["accuracy"])
        self.lambda_xy=int(args.lambda_xy)
        self.lambda_cy=int(args.lambda_cy)
        self.lambda_xc=int(args.lambda_xc)
        logging.basicConfig(filename=logdir+'/log.txt',
                    format = '%(message)s',
                    level=logging.INFO)
        logging.info(args)
    


    # define model inference
    def forward(self, x,y,c,t):
        return self.network(x,y,c,t)

    # define optimizers and schedulers
    def configure_optimizers(self):
        if self.optim.lower()=='sgd':
            optimizer=torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.parameters()),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.wd,
                )
        elif self.optim.lower()=='adam':
            optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                       lr=self.learning_rate,
                                       betas=(self.beta_1,self.beta_2),
                                       weight_decay=self.wd)
        elif self.optim.lower()=='adamw':
            optimizer=torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                       lr=self.learning_rate,
                                       weight_decay=self.wd)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "epoch_loss_epoch",
        }

    def training_step(self, batch, batch_idx):
        if self.dataset=='celeba':
            batch_X, cy=batch
            batch_Y,batch_C=cy
        else:
            batch_X, batch_Y,batch_C = batch

        out = self.network(batch_X,batch_C,is_train=True,use_cy=True)
        out_xy,out_cy,out_c=out

        cls_loss = self.loss_label(out_xy,batch_Y)
        cy_loss = self.loss_label(out_cy,batch_Y)
        cpt_loss = self.loss_concept(out_c,batch_C)
        
        cls_weight=self.lambda_xy
        cy_weight=self.lambda_cy
        cpt_weight=self.lambda_xc


        loss=cpt_weight*cpt_loss+cls_weight*cls_loss+cy_loss*cy_weight
        
        # evaluate y from xy
        _, met_xy = torch.min(out_xy, 1)
        metrics_xy = self.computer(met_xy, batch_Y)

        # # evaluate y from cy
        _, met_cy = torch.min(out_cy, 1)
        metrics_cy = self.computer(met_cy, batch_Y)

        # evaluate c
        _, met_c = torch.min(out_c, 2)
        metrics_c = self.computer(met_c, batch_C)
        c_overall_acc=sklearn.metrics.accuracy_score(batch_C.cpu().numpy(), met_c.cpu().numpy())

        self.log("epoch_loss",loss,on_epoch=True)
        self.log("epoch_loss_xy",cls_loss,on_epoch=True)
        self.log("epoch_loss_cy",cy_loss,on_epoch=True)
        self.log("epoch_loss_c",cpt_loss,on_epoch=True)
        self.log("accuracy_train_c_overall",c_overall_acc,on_epoch=True,on_step=False)
        self.log("accuracy_train_c",metrics_c['accuracy'],on_epoch=True,on_step=False)
        self.log("accuracy_train_y_xy",metrics_xy["accuracy"],on_epoch=True,on_step=False)
        self.log("accuracy_train_y_cy",metrics_cy["accuracy"],on_epoch=True,on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.dataset=='celeba':
            batch_X, cy=batch
            batch_Y,batch_C=cy
        else:
            batch_X, batch_Y,batch_C = batch
        out = self.network(batch_X,batch_C,is_train=True,use_cy=True)
        out_xy,out_cy,out_c=out
        cls_loss = self.loss_label(out_xy,batch_Y)
        cy_loss = self.loss_label(out_cy,batch_Y)
        cpt_loss = self.loss_concept(out_c,batch_C)
        loss=cpt_loss+cls_loss+cy_loss
        
        # evaluate y from xy
        _, met_xy = torch.min(out_xy, 1)
        metrics_xy = self.computer(met_xy, batch_Y)

        # # evaluate y from cy
        _, met_cy = torch.min(out_cy, 1)
        metrics_cy = self.computer(met_cy, batch_Y)

        # evaluate c
        _, met_c = torch.min(out_c, 2)
        c_overall_acc=sklearn.metrics.accuracy_score(batch_C.cpu().numpy(), met_c.cpu().numpy())
        metrics_c = self.computer(met_c, batch_C)

        self.log("epoch_val_loss",loss,on_epoch=True)
        self.log("epoch_val_loss_xy",cls_loss,on_epoch=True)
        self.log("epoch_val_loss_cy",cy_loss,on_epoch=True)
        self.log("epoch_val_loss_c",cpt_loss,on_epoch=True)
        self.log("accuracy_val_c_overall",c_overall_acc,on_epoch=True,on_step=False)
        self.log("accuracy_val_c",metrics_c['accuracy'],on_epoch=True,on_step=False)
        self.log("accuracy_val_y_xy",metrics_xy["accuracy"],on_epoch=True,on_step=False)
        self.log("accuracy_val_y_cy",metrics_cy["accuracy"],on_epoch=True,on_step=False)
        return loss
 
    def test_step(self, batch, batch_idx):
        if self.dataset=='celeba':
            batch_X, cy=batch
            batch_Y,batch_C=cy
        else:
            batch_X, batch_Y,batch_C = batch
        out = self.network(batch_X,batch_C,is_train=True,use_cy=True)
        out_xy,out_cy,out_c=out
        
        # evaluate y from xy
        _, met_xy = torch.min(out_xy, 1)
        metrics_xy = self.computer(met_xy, batch_Y)

        # # evaluate y from cy
        _, met_cy = torch.min(out_cy, 1)
        metrics_cy = self.computer(met_cy, batch_Y)

        # evaluate c
        _, met_c = torch.min(out_c, 2)
        metrics_c = self.computer(met_c, batch_C)
        metrics_c_overall=sklearn.metrics.accuracy_score(batch_C.cpu().numpy(), met_c.cpu().numpy())

        self.log("accuracy_test_c_overall",metrics_c_overall,on_epoch=True,on_step=False)
        self.log("accuracy_test_c",metrics_c['accuracy'],on_epoch=True,on_step=False)
        self.log("accuracy_test_y_xy",metrics_xy["accuracy"],on_epoch=True,on_step=False)
        self.log("accuracy_test_y_cy",metrics_cy["accuracy"],on_epoch=True,on_step=False)
        return [metrics_c_overall,metrics_c['accuracy'],metrics_xy['accuracy'],metrics_cy['accuracy']]
    
    def test_epoch_end(self,outputs):
        all_results=torch.tensor(outputs)
        mean_val=torch.mean(all_results,0)
        logging.info("accuracy_test_c_overall: {}".format(mean_val[0]))
        logging.info("accuracy_test_c: {}".format(mean_val[1]))
        logging.info("accuracy_test_y_xy: {}".format(mean_val[2]))
        logging.info("accuracy_test_y_cy: {}".format(mean_val[3]))