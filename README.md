# Unofficial-PyTorch-YOLOV7
I re-implement YOLOV7 (CVPR 2023) in a simplified version based on my coding style.  
This repository aims to reproduce the YOLOV7 performance on the MS COCO dataset (69.7% AP<sub>50</sub>) and only contains part of the content.
## Create issues from this repository
Since this account is seldom used, please send an email to wangren@snu.ac.kr.
## Environment
Ubuntu 20.04.3 LTS  
Python 3.8.10  
CUDA 11.1  
cuDNN 8  
4 NVIDIA GeForce RTX 3090 24GB GPUs
## Setup
Step 1: Get repository  
```
git clone https://github.com/SMNUResearch/Unofficial-PyTorch-YOLOV7.git
cd Unofficial-PyTorch-YOLOV7
```
Step 2: Install dependencies  
```
sh install.sh
```
## Dataset Preparation
Step 1: Create a data directory
```
mkdir data
cd data
mkdir COCO
cd COCO
```
Step 2: Download COCO.zip from [[Google Drive]](https://drive.google.com/file/d/1FcIbbWalDVymehQIaIBBE4dEVNoPKQQF/view?usp=sharing)  
Step 3: Place COCO.zip in this directory  
Step 4: Extract files from COCO.zip  
```
unzip COCO.zip
rm COCO.zip
```
Step 5: Download image files (2017 Train and 2017 Val) from the official [[MS COCO website]](https://cocodataset.org/#download)  
Step 6: Unzip these two files and rename the folders as 'train' and 'valid'  
Step 7: Place 'train' and 'valid' folders in this directory  
Step 8: Check the structure of data files
```
├── config
│   ├── YOLOV7_COCO.yaml
├── data
│   ├── COCO
│       ├── train                              # 118287 images
│       ├── valid                              # 5000 images
│       ├── COCO_ANNOTATION_TRAIN.txt          # annotations of 117266 images (remove 1021 unlabelled images): image_name xmin,ymin,xmax,ymax,class_id ...
│       ├── COCO_VALID_2017.json               # ground truth annotations: instances_val2017.json
│       ├── COCO_VALID_LIST.txt                # list of 4952 images (remove 48 unlabelled images): image_name
├── model                              
│   ├── yolov7.py
└── ...
```
## Quick Test
Step 1: Download the pretrained weight [YOLOV7_COCO.pt](https://drive.google.com/file/d/1IQmu_GTC9tdVsnNj2aRr3h1W8vo7WeLI/view?usp=sharing)  
Step 2: Place the weight file in this repository  
Step 3: Performance evaluation (image size is 640)
```
python3 test.py
```
Step 4: Check the results
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.509
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.695
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.554
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.652
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.384
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.637
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.687
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.536
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.734
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.831
```
## Training (single-machine)
Multi-GPU [[Training log]](https://drive.google.com/file/d/1vt3JsFbkHKaAavbQPKOVzd5UQdj-3B6x/view?usp=sharing)
```
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 train.py
```
Single-GPU
```
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 train.py
```
Default arguments: 400 epochs, image size is 640, batch size is 32, num_workers is 4
## Reference
[https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)  
[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
