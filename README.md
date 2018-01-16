## attention

This repository was a project form of [keras_yolo2](https://github.com/experiencor/basic-yolo-keras),which cannot be run directly. I have done some modify to make it easier to be used.

## 2 data prepare

download coco dataset:


+ train images: http://images.cocodataset.org/zips/train2014.zip
+ validation images: http://images.cocodataset.org/zips/val2014.zip
+ train and validation annotations: http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip them to your disk like below:

```
D:\DATA\COCO
├─annotations_trainval2014
│  └─annotations
├─images
│  ├─train2014
│  └─val2014
├─keras_yolo_weights
└─pascal_format
    ├─train
    └─val

```

all **annotation files** such as `instances_train2014.json`,`person_keypoints_train2014.json` are included in `D:\DATA\COCO\annotations_trainval2014\annotations`.

`keras_yolo_weights` ，`pascal_format\train` and `pascal_format\val` are directories  created for training .


## 3. convert coco dataset to pascal format

Run `coco2pascal.py` script to do convertion. Do not forget to modify `subset` and `dst` to `train` and `val` when processing training and validation dataset.

## 4 train network

After convertion ,your dataset should be ready in `pascal_format` ,and you can run `train_yolo.py` script to do training.

## 5 note

Decoding coco format json files  with Python may take a while.


