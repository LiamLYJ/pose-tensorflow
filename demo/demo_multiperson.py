import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread, imsave

from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
from multiperson.visualize import PersonDraw, visualize_detections

import matplotlib.pyplot as plt


cfg = load_config("demo/pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

draw_multi = PersonDraw()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
# file_name = "demo/image_multi.png"
# file_name = '/home/gpu_server/lyj/coco/train2014/COCO_train2014_000000524297.jpg'
file_name = '/home/gpu_server/lyj/coco/train2014/COCO_train2014_000000524401.jpg'
# file_name = '/home/gpu_server/lyj/coco/train2014/COCO_train2014_000000524320.jpg'
# file_name = "demo/3_2_0000348.png"
# file_name = "demo/im0001.jpg"
image = imread(file_name, mode='RGB')

image_batch = data_to_input(image)

# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, pairwise_diff = predict.extract_cnn_output(
    outputs_np, cfg, dataset.pairwise_stats)

# for name, item in outputs_np.items():
#     print ('name of itermsis ', name)
#     print ('shape of %s is :' % name, item.shape)
#
# print ('shape of scmape is:', scmap.shape)
# print ('shape of locref is:', locref.shape)
# print ('shape of pairwise_diff is:', pairwise_diff.shape)

detections = extract_detections(cfg, scmap, locref, pairwise_diff)
# print ('len of detections coord:', len(detections.coord))
# print ('len of detections coord_grid:', len(detections.coord_grid))
# print ('len of detections conf:', len(detections.conf))
# print ('len of detections pairwise:', len(detections.pairwise))
# print ('coord:', detections.coord[0])
# print ('coord_grid: ', detections.coord_grid[0])
# print ('conf: ', detections.conf[0])
# print ('pairwise: ', detections.pairwise[0])

# print ('coord shape:', detections.coord[0].shape)
# print ('coord_grid shape: ', detections.coord_grid[0].shape)
# print ('conf shape: ', detections.conf[0].shape)
# print ('pairwise shape: ', detections.pairwise[0].shape)

unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(
    sm, detections)
person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)
# print ('person_conf_multi:', person_conf_multi.shape)
# print ('********')
# print (person_conf_multi)
# raise
img = np.copy(image)

visim_multi = img.copy()

fig = plt.imshow(visim_multi)
draw_multi.draw(visim_multi, dataset, person_conf_multi)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

plt.show()
visualize.waitforbuttonpress()
