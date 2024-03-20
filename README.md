# CosalPure

Official implementation of "COSALPURE: Learning Concept from Group Images for Robust Co-Saliency" in CVPR 2024. 
<!-- [arXiv](https://arxiv.org/pdf/2009.09258.pdf) -->

## Requirements

We recommend creating a new conda environment for this project. Conda can be installed through below instructions.


```bash

# clone our repo
git clone https://github.com/V1len/CosalPure
cd CosalPure

# create conda environment
conda create --name new_env --file environment.txt
```


## Datasets

### Used in our paper:

* ***Cosal2015*** (50 groups, 2015 images) "Detection of Co-salient Objects by Looking Deep and Wide, *IJCV(2016)*''

* ***iCoseg*** (38 groups, 643 images) ''iCoseg: Interactive Co-segmentation with Intelligent Scribble Guidance, *CVPR(2010)*''

* ***CoSOD3k*** (160 groups, 3316 images) ''Taking a Deeper Look at the Co-salient Object Detection, *CVPR(2020)*''

* ***CoCA*** (80 groups, 1295 images) ''Gradient-Induced Co-Saliency Detection, *ECCV(2020)*''


## Add degradation

For adversarial attack, please refer to the augment variant of **[*Jadena*](https://github.com/tsingqguo/jadena/)**.
Check ***attack.ipynb*** for details.

For common corruption, please refer to the degradation process of **[*ImageNet-C*](https://github.com/hendrycks/robustness/)**


## Method

```bash

# concept learning
python concept_learning.py

# concept-guided purification
python concept_guided_purification.py
```

## Evaluate

### Used in our paper:

**[*GICD*](https://github.com/backseason/PoolNet/)**"Gradient-induced co-saliency detection.*ECCV(2020)*"

**[*GCAGC*](https://github.com/backseason/PoolNet/)**"Adaptive graph convolutional network
with attention graph clustering for co-saliency detection.*CVPR(2020)*"

**[*PoolNet*](https://github.com/backseason/PoolNet/)**"A simple pooling-based design
for real-time salient object detection.*CVPR(2019)*"



Aforementioned CoSOD models should be downloaded to ***weights/***.

Then run ***evaluate.ipynb*** for evaluation.


## Citation

*To be updated.*


