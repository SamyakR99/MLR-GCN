train.py --config_file configs/models/rn101_ep50.yaml --datadir /home/samyakr2/multilabel/data/coco14 --dataset_config_file /home/samyakr2/SHOP/DualCoOp/configs/datasets/coco.yaml --input_size 448 --lr 0.002 --loss_w 0.2 -pp 1 --csc --max_epochs 55 --imbalanced 0



train.py --config_file configs/models/rn101_ep50.yaml --datadir /home/samyakr2/multilabel/data/VOC2007/VOCdevkit/VOC2007/ --dataset_config_file /home/samyakr2/SHOP/DualCoOp/configs/datasets/voc2007.yaml --input_size 448 --lr 0.001 --loss_w 0.2 -pp 1 --csc --imbalanced 0