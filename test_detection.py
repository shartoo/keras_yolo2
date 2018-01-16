import cv2
import os
import matplotlib.pyplot as plt
from keras_yolo.construct_network import yolo_network,TRUE_BOX_BUFFER,OBJ_THRESHOLD,NMS_THRESHOLD,ANCHORS,CLASS,LABELS
from keras_yolo.utils import decode_netout, draw_boxes
import numpy as np


model = yolo_network()
model.load_weights("d:/data/coco/keras_yolo_weights/weights_coco.h5")

dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
image = cv2.imread('images/giraffe.jpg')

plt.figure(figsize=(10,10))

input_image = cv2.resize(image, (416, 416))
input_image = input_image / 255.
input_image = input_image[:,:,::-1]
input_image = np.expand_dims(input_image, 0)

netout = model.predict([input_image, dummy_array])

boxes = decode_netout(netout[0],
                      obj_threshold=OBJ_THRESHOLD,
                      nms_threshold=NMS_THRESHOLD,
                      anchors=ANCHORS,
                      nb_class=CLASS)
image = draw_boxes(image, boxes, labels=LABELS)

plt.imshow(image[:,:,::-1]); plt.show()
