import numpy as np
import os
import json
import torch
from data.cub_stats import CONCEPT_SEMANTICS, SELECTED_CONCEPTS
from data.data_util import CUB_DATA_DIR



with open('./cub_mapping.json','r') as f:
    gt_mapping=json.load(f)

base_dir="/home/yqinar/CBM/EBCBM_pub/exp/2024-01-25_11-23-09-cub-resnet101_imagenet"

y_pred=np.load(os.path.join(base_dir,"y_pred.npy"),allow_pickle=True)
y_gt=np.load(os.path.join(base_dir,"y_gt.npy"),allow_pickle=True)
c_pred=np.load(os.path.join(base_dir,"c_pred.npy"),allow_pickle=True)
c_gt=np.load(os.path.join(base_dir,"c_gt.npy"),allow_pickle=True)

# y_pred= torch.tensor(y_pred).argmax(dim=-1).numpy()
# c_pred= (c_pred >= 0.5).astype(np.int32)


y_labelset=np.unique(y_pred)
# print(len(y_labelset))
p_ym=[]
for ym in y_labelset:
    count = np.count_nonzero(y_pred == ym)
    p_ym.append(count / len(y_pred))
# print(p_ym)

p_C=[]
c_pred_jointset=np.unique(c_pred,axis=0)

for c_pred_joint in c_pred_jointset:
    count = np.count_nonzero(np.all(c_pred==c_pred_joint,axis=1))
    p_C.append(count / len(c_pred_jointset))
# print(p_C)




p_ym_cond_C=[]
for ym in y_labelset:
    single_cond=[int(0) for i in range(len(c_pred_jointset))]
    indices = np.where(ym == y_pred)[0]
    cond_c=c_pred[indices]
    joint_c,count=np.unique(cond_c,axis=0,return_counts=True)
    for j,co in zip(joint_c,count):
        for idx,i in enumerate(c_pred_jointset):
            if np.all(i==j,axis=0):
                single_cond[idx]=co/len(cond_c)
    p_ym_cond_C.append(single_cond)


# print(p_ym_cond_C[0])

p_ck_joint_ym=[]
for ym in y_labelset:
    single_cond=[]
    indices = np.where(ym == y_pred)[0]
    cond_c=c_pred[indices]
    for idx_ck in range(len(c_pred[0])):
        count = np.count_nonzero(1 == cond_c[:,idx_ck])
        single_cond.append(count/len(cond_c))
    p_ck_joint_ym.append(single_cond)

joint_importance=p_ym_cond_C


# Marginal importance
# print(len(c_pred[0]))
marginal_importance=[]
for idx_y,y in enumerate(p_ym):
    single_y_importance=[]
    for idx_ck in range(len(c_pred[0])):
        importance=p_ck_joint_ym[idx_y][idx_ck]
        single_y_importance.append(importance)
    marginal_importance.append(single_y_importance)

with open(f'{CUB_DATA_DIR}/classes.txt','r') as f:
    classes_=f.readlines()
classes=[]
for i in classes_:
    renamed=i.split(" ")[-1][4:-2]
    classes.append(renamed)



count=0
most_import_joint_c=[]
for check_idx,y in enumerate(y_labelset):
    print('='*5,f"Importance Score for Label {int(y)}: {classes[int(y)]}",'='*5)
    print('='*5,f"Joint Importance Score for Label {int(y)}: {classes[int(y)]}",'='*5)
    print(joint_importance[check_idx])
    print(len(joint_importance[check_idx]))
    # print('='*5,f"Most important joint c index",'='*5)
    much_likely=np.argmax(joint_importance[check_idx])
    print('='*5,f"Most important joint concept set",'='*5)
    print(list(c_pred_jointset[much_likely].astype(int)))
    most_import_joint_c.append(list(c_pred_jointset[much_likely].astype(int)))
    print('='*5,f"Marginal Importance Score for Label {int(y)}: {classes[int(y)]}",'='*5)
    print(marginal_importance[check_idx])
    print('='*5,f"Concept groundtruth for Label {int(y)}: {classes[int(y)]}",'='*5)
    print(gt_mapping[int(y)])
    tf=(gt_mapping[int(y)]==list(c_pred_jointset[much_likely].astype(int)))
    if tf:
        count+=1
    print(tf)
    c_gt_select=gt_mapping[int(y)]
    idx=np.nonzero(c_gt_select)[0].astype(int)
    mapping=np.array(SELECTED_CONCEPTS)[idx]
    names=np.array(CONCEPT_SEMANTICS)[mapping]
    print(names)
    print("="*5,"pred top5",'='*5)
    single_checkidx=marginal_importance[check_idx]
    _,topidx=torch.topk(torch.tensor(marginal_importance[check_idx]),5)
    mapping=np.array(SELECTED_CONCEPTS)[topidx]
    names=np.array(CONCEPT_SEMANTICS)[mapping]
    scoresidx=topidx.tolist()
    for name,scoreidx in zip(names,scoresidx):
        print(f"{name}: {single_checkidx[scoreidx]}")
    print("+"*10)

