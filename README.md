# [Improving Multi-label Recognition using Class Co-Occurrence Probabilities (ICPR 2024, CVPRW 2024)](https://arxiv.org/abs/2404.16193)
Authors: [Samyak Rawlekar*](https://samyakr99.github.io/), [Shubhang Bhatnagar*](https://shubhangb97.github.io/), [Narendra Ahuja](https://vision.ai.illinois.edu/narendra-ahuja/)

[Paper](https://arxiv.org/abs/2404.16193) | [Project Page](https://shubhangb97.github.io/MLR_gcn/) | [Poster](https://shubhangb97.github.io/MLR_gcn/img/MLR_GCN_poster.pdf) | Slides 

Our implementation is built on the official implementation of  [DualCoOp](https://github.com/sunxm2357/DualCoOp).


## Environment

We use Pytorch with python 3.9. 

Use `conda env create -f environment.yml` to create the conda environment.
In the conda environment, install `pycocotools` and `randaugment` with pip:
```
pip install pycocotools
pip install randaugment
```
And follow [the link](https://github.com/KaiyangZhou/Dassl.pytorch) to install `dassl`.


## Training 
### MLR 
Use the following code to learn a model for MLR with Partial Labels
```
python train.py  --config_file configs/models/rn101_ep50.yaml \
--datadir <your_dataset_path> --dataset_config_file configs/datasets/<dataset>.yaml \
--input_size 448 --lr <lr_value>   --loss_w <loss_weight> \
-pp <porition_of_avail_label> --csc --imbalanced --path_to_relation
```
Some Args:
- `imbalance` : 0/1 (Use of proposed RASL loss)
- path_to_relation: path to the relation file of the corresponding dataset
- `dataset_config_file`: currently the code supports `configs/datasets/coco.yaml` and `configs/datasets/voc2007.yaml`  
- `lr`: `0.001` for VOC2007 and `0.002` for MS-COCO.
- `pp`: from 0 to 1. It specifies the portion of labels are available during the training.
- `loss_w`: to balance the loss scale with different `pp`. We use larger `loss_w` for smaller `pp`.
- `csc`: specify if you want to use class-specific prompts. We suggest to use class-agnostic prompts when `pp` is very small.   
Please refer to `opts.py` for the full argument list.
For Example:
```
python train.py --config_file configs/models/rn101_ep50.yaml --datadir data/VOC2007/VOCdevkit/VOC2007/ \
--dataset_config_file configs/datasets/voc2007.yaml --input_size 448 --lr 0.001 --loss_w 0.2 -pp 1 --csc \
--imbalanced 0 --path_to_relation co_occurrence_matrix_voc2007.pth


```


## Evaluation / Inference
### MLR with Partial Labels
```
python val.py --config_file configs/models/rn101_ep50.yaml \
--datadir <your_dataset_path> --dataset_config_file configs/datasets/<dataset>>.yaml \
--input_size 224  --pretrained <ckpt_path> --csc --imbalanced --path_to_relation
```

Please cite our work if you find it is helpful to your research.
```
@article{rawlekar2024improving,
  title={Improving Multi-label Recognition using Class Co-Occurrence Probabilities},
  author={Rawlekar, Samyak and Bhatnagar, Shubhang and Srinivasulu, Vishnuvardhan Pogunulu and Ahuja, Narendra},
  journal={arXiv preprint arXiv:2404.16193},
  year={2024}
}
```







