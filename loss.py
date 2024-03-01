import torch
import torch.nn as nn
import torch
import numpy as np



class EBMLoss_label(nn.Module):
    def __init__(self, class_list,device):
        super(EBMLoss_label, self).__init__()
        self.class_list=class_list
        self.device=device

    def forward(self, energy,gt):

        batch_size=energy.size(0)
        y_tem = torch.tensor([self.class_list.index(tem) for tem in gt]).long().to(self.device)
        y_tem = y_tem.view(batch_size, 1)
        energy_pos = energy.gather(dim=1, index=y_tem)
        partition_estimate = -1 * energy
        partition_estimate = torch.logsumexp(partition_estimate, dim=1, keepdim=True)
        predL = energy_pos + partition_estimate
       

        return predL.mean()
    
class EBMLoss_concept(nn.Module):
    def __init__(self, class_list,device):
        super(EBMLoss_concept, self).__init__()
        self.class_list=class_list
        self.device=device

    def forward(self, energy,gt):

        y_tem=gt.unsqueeze(-1).to(torch.int64)
        cpt_loss=torch.zeros([]).to(self.device)
        for i in range(energy.shape[1]):
            energy_pos = energy[:,i:i+1].gather(dim=2, index=y_tem[:,i:i+1])
            partition_estimate = -1 * energy[:,i:i+1]
            partition_estimate = torch.logsumexp(partition_estimate, dim=2, keepdim=True)
            predL = energy_pos + partition_estimate
            predL = predL.mean()
            cpt_loss+=predL

        return cpt_loss
