import os
import random
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


LABEL_COLORS_LIST = [
    [0, 0, 0],              # Background. 黑色
    [135, 206, 250],        # Tooth. 浅蓝色 (Light Blue)，类似于天空蓝
    [63, 0, 127],           # Pulp. 深紫色 (Deep Indigo)
    [255, 0, 0],            # Caries. 红色 (Red)
    [192, 192, 192],        # Fillings. 银灰色 (Silver Gray)
    [255, 255, 0],          # Root canal fillings. 黄色 (Yellow)
    [0, 0, 255],            # Periapical lesion. 蓝色 (Blue)
    [255, 20, 147]          # Crown. 深粉色 (Deep Pink)
]



def get_semantic_dataset(root, split):
    """
    加载语义分割数据集
    Args:
        root: 数据集根目录
        split: 'train' 或 'val'
    """

    image_dir = os.path.join(root, "images", split)
    seg_dir = os.path.join(root, "annotations", split)
    

    image_files = [f for f in PathManager.ls(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    dataset_dicts = []
    for img_file in sorted(image_files):
        record = {}
        

        image_path = os.path.join(image_dir, img_file)
        seg_filename = os.path.splitext(img_file)[0] + '.png'  #
        seg_path = os.path.join(seg_dir, seg_filename)
        
        assert PathManager.exists(seg_path), f"{seg_path} not found"
        
        record["file_name"] = image_path
        record["sem_seg_file_name"] = seg_path  # 
        
        dataset_dicts.append(record)
    
    return dataset_dicts

def register_semantic_dataset(name, root, split):
    classes_file = os.path.join(root, "classes.txt")
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    DatasetCatalog.register(
        name,
        lambda: get_semantic_dataset(root, split)
    )
    

    MetadataCatalog.get(name).set(
        stuff_classes=classes,  
        stuff_colors=LABEL_COLORS_LIST,  
        ignore_label=255,   
        evaluator_type="sem_seg"
    )

def verify_dataset(dataset_name):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    print(f"Dataset {dataset_name}:")
    print(f"Number of images: {len(dataset_dicts)}")
    print(f"Classes: {metadata.stuff_classes}")
    
    for d in random.sample(dataset_dicts, min(3, len(dataset_dicts))):
        print("\nImage:", d["file_name"])
        print("Segmentation:", d["sem_seg_file_name"])
        assert os.path.exists(d["file_name"]), f"Image not found: {d['file_name']}"
        assert os.path.exists(d["sem_seg_file_name"]), \
            f"Segmentation not found: {d['sem_seg_file_name']}"
