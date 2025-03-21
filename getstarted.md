# Getting Started Guide

Our framework is primarily built upon [Mask2Former](https://github.com/facebookresearch/Mask2Former).  
For detailed installation instructions, please refer to the [Setup Guide](https://github.com/facebookresearch/MaskFormer/blob/main/INSTALL.md). 
---

## 📂 dataset preparation

```
tool_data/7class_label/
├── annotations/
│   ├── train/          
│   └── val/            
├── images/
│   ├── train/          
│   └── val/            
└── classes.txt     
```    


## 🚀 training  

1. **Select a backbone_model**  
   `CHECKPOINTS/vits14_dino.pth`

To train a model with "train_net.py", first setup the corresponding datasets following datasets/README.md, then run:

```
1-GPU training 
python train_net.py \
  --config-file configs/coco/semantic-segmentation/dinov2/dinov2_vit_small.yaml \
  --num-gpus 1 \
  MODEL.WEIGHTS "CHECKPOINTS/vits14_dino.pth"
```


## 🚀 Inference Demo with Pre-trained Models

```
python demo/demo1.py \
    --config-file configs/coco/semantic-segmentation/dinov2/dinov2_vit_small.yaml \
    --input data/* \
    --output output \
    --opts MODEL.WEIGHTS DINOOUTPUT_SMALL/model_0062999.pth
```