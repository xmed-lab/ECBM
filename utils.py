import random
import numpy as np
import torch
from metrics import MetricComputer
import sklearn
from data.cub import CONCEPT_GROUP_MAP
import math


def generate_random_numbers(n,n_concept):

    numbers = set()
    while len(numbers) < float(n):
        numbers.add(random.randint(0, n_concept-1))

    return list(numbers)


class EarlyStopping():
    def __init__(self,patience=7,verbose=False,delta=1):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_param=None
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_param = None
    def __call__(self,val_loss,model):
        # logging.info("val_loss={},best={}".format(val_loss,self.best_score))
        score = -val_loss
        # logging.info("score={}".format(score))
        # logging.info("counter={}".format(self.counter))
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score <= self.best_score+self.delta:
            self.counter+=1
            # logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    def save_checkpoint(self,model):
        self.best_param=model
        # print("saving the best")
        
class WeightScheduler():
    def __init__(self,min,max,iters):
        self.min=min
        self.max=max
        self.iters=iters
        self.now_iters=0
    def reset(self):
        self.now_iters = 0

    def __call__(self):
        if self.now_iters<self.iters:
            now_scale=((self.now_iters/self.iters)-0.5)*10
            now_weight=self.max/(1+math.exp(-now_scale))
            self.now_iters+=1
            return now_weight
        else:
            return self.max

class MetricCalculator():
    def __init__(self):
        self.total_results_y=torch.tensor([])
        self.total_targets_y=torch.tensor([])
        self.total_results_c=torch.tensor([])
        self.total_targets_c=torch.tensor([])
        self.computer=MetricComputer(metric_names=['accuracy'])
        
    
    def update(self,pred,gt):
        y_prob,c_prob=pred
        c,y=gt
        y_prob_inf=y_prob.clone().detach()
        c_prob_inf=c_prob.clone().detach()
        _, met_cy = torch.max(y_prob_inf, 1)
        met_cy=met_cy.squeeze(-1)
        _, met_c = torch.max(c_prob_inf, 2)
        met_c=met_c.squeeze(-1)
        self.total_results_c = torch.cat((self.total_results_c, met_c.cpu()), 0)
        self.total_targets_c = torch.cat((self.total_targets_c, c.cpu()), 0)
        self.total_results_y = torch.cat((self.total_results_y, met_cy.cpu()), 0)
        self.total_targets_y = torch.cat((self.total_targets_y, y.cpu()), 0)
        return self.return_metrics()
        

    def return_metrics(self):
        y_acc = self.computer(self.total_results_y, self.total_targets_y)
        c_acc_overall=sklearn.metrics.accuracy_score(self.total_targets_c.cpu().numpy(), self.total_results_c.cpu().numpy())
        c_acc=0
        total_c_acc=[]
        for i in range(self.total_results_c.shape[-1]):
            metrics_c = self.computer(self.total_results_c[:,i], self.total_targets_c[:,i])
            total_c_acc.append(metrics_c['accuracy'])
            c_acc+=metrics_c["accuracy"]
        total_c_acc=torch.tensor(total_c_acc)
        c_acc=c_acc/self.total_results_c.shape[-1]
        return y_acc["accuracy"],c_acc_overall,c_acc


    def reset(self):
        self.total_results_y=torch.tensor([])
        self.total_targets_y=torch.tensor([])
        self.total_results_c=torch.tensor([])
        self.total_targets_c=torch.tensor([])
    
    def get_data(self):
        return(self.total_results_y,self.total_targets_y,self.total_results_c,self.total_targets_c)


def select_concept_group(n,n_concept):
    count=0
    to_be_intervened=[]
    for i in CONCEPT_GROUP_MAP:
        to_be_intervened+=CONCEPT_GROUP_MAP[i]
        count+=1
        if count==n:
            return to_be_intervened