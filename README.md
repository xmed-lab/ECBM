# Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Conditional Interpretations (ECBMs)
This repo is the official implementation for our ICLR 2024 paper:

[**Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Conditional Interpretations**](https://openreview.net/forum?id=I1quoTXZzc)

Xinyue Xu, Yi Qin, Lu Mi, Hao Wang, Xiaomeng Li

*Twelfth International Conference on Learning Representations (ICLR), 2024.*

[[Paper]()] [[OpenReview](https://openreview.net/forum?id=I1quoTXZzc))] [[PPT]()]

## Overview of our ECBM

<p align="center">
<img src="fig/figure-v7-1.png" alt="" width="80%"/>
</p>
**Top:** During training, ECBM learns positive concept embeddings ${v}_k^{(+)}$ (in black), negative concept embeddings ${v}_k^{(-)}$ (in white), the class embeddings $\mathbf{u}_m$ (in black), and the three energy networks by minimizing the three energy functions, $E_{{\theta}}^{class}({x}, {y})$, $E_{{\theta}}^{concept}({x}, {c})$, and $E_{{\theta}}^{global}({c}, {y})$ using the total loss function. The concept $\mathbf{c}$ and class label $\mathbf{y}$ are treated as constants. 

**Bottom:** During inference, we (1) freeze all concept and class embeddings as well as all networks, and (2) update the predicted concept probabilities $\widehat{\mathbf{c}}$ and class probabilities $\widehat{\mathbf{y}}$ by minimizing the three energy functions using the total loss function.

The total loss function:

${L}_{total}^{all} = {E}_{({x},{c},{y}) \sim p_{{D}}({x},{c},{y})}[{L}_{total}({x}, {c},{y})]$.

## Installation

### Prerequisites

We run all experiments on single NVIDIA RTX3090 GPU. 

```python
pip install -r requirements.txt
```

### Dataset Preperation

+ [**CUB**](https://worksheets.codalab.org/bundles/0x518829de2aa440c79cd9d75ef6669f27)
+ [**CelebA**](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
+ [**AWA2**](https://cvml.ista.ac.at/AwA2/)

## Configuration

 Configurations are in [config.json](./config.json) file.
 - Hyperparameter search, set sweep = true.
 - Select dataset, set dataset='TARGET DATASET'.
 - If or not using pretrained weight, pretrained = ture.
 - Freeze backbone, set freezebb = true.
 - emb_size: the feature size after the feature encoder.
 - hid_size: projected feature size.
 - cpt_size: the number of concepts.

## Run Experiments

```python
python main.py
```

## Results

### Prediction

<p align="center">
<img src="fig/main_results.png" alt="" width="80%"/>
</p>

Accuracy on Different Datasets. We report the mean and standard deviation from five runs with different random seeds. For ProbCBM (marked with “*”), we report the best results from the ProbCBM paper (Kim et al., 2023) for CUB and AWA2 datasets.

### Concept Intervention

<p align="center">
<img src="fig/intervention_v2-1.png" alt="" width="80%"/>
</p>

Performance with different ratios of intervened concepts on three datasets (with error bars). The intervention ratio denotes the proportion of provided correct concepts. We use CEM with RandInt. CelebA and AWA2 do not have grouped concepts; thus we adopt individual intervention.

### Conditional Interpretations

<p align="center">
<img src="fig/concept-confidence-demo-fig-v2-1.png" alt="" width="80%"/>
</p>
Marginal concept importance for top $3$ concepts of $4$ different classes computed using Proposition 3.2. ECBM's estimation (Ours) is very close to the ground truth (Oracle).

## Reference

TODO



