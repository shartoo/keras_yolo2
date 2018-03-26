## 1 attention

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

The output of training looks like:

```
2405/5120 [=============>................] - ETA: 35:14 - loss: 2.02052018-01-16 14:39:18.135780: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Dummy Line 	[0]
2018-01-16 14:39:18.135901: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss XY 	[0.0516835116]
2018-01-16 14:39:18.136024: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss WH 	[0.344218343]
2018-01-16 14:39:18.136151: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss Conf 	[0.00673754606]
2018-01-16 14:39:18.136287: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss Class 	[0.719579101]
2018-01-16 14:39:18.136412: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Total Loss 	[1.12221849]
2018-01-16 14:39:18.136537: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Current Recall 	[0]
2018-01-16 14:39:18.136658: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Average Recall 	[0.0180941429]

2406/5120 [=============>................] - ETA: 35:13 - loss: 2.02012018-01-16 14:39:19.023213: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Dummy Line 	[0]
2018-01-16 14:39:19.023467: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss XY 	[0.0519431084]
2018-01-16 14:39:19.023775: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss WH 	[0.518281937]
2018-01-16 14:39:19.023897: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss Conf 	[0.00582001451]
2018-01-16 14:39:19.024018: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss Class 	[1.12371182]
2018-01-16 14:39:19.024135: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Total Loss 	[1.69975686]
2018-01-16 14:39:19.024249: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Current Recall 	[0]
2018-01-16 14:39:19.024364: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Average Recall 	[0.0180866253]

2407/5120 [=============>................] - ETA: 35:13 - loss: 2.02002018-01-16 14:39:19.833287: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Dummy Line 	[0]
2018-01-16 14:39:19.833411: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss XY 	[0.0596466064]
2018-01-16 14:39:19.833530: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss WH 	[0.538067877]
2018-01-16 14:39:19.833647: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss Conf 	[0.00888421573]
2018-01-16 14:39:19.833766: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Loss Class 	[1.2905916]
2018-01-16 14:39:19.833883: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\kernels\logging_ops.cc:79] Total Loss 	[1.89719033]
```

## 5 Weight File

Here is my weight file [keras_yolo2_weight(BaiduYunDisk)](https://pan.baidu.com/s/1ZLlbGaPj--LXql2m_I1gLg) trained from my code .

## 5 test detection

Running `test_detection.py` to test detection.

## 6 note

Decoding coco format json files  with Python may take a while.

