import os
import sys

import numpy as np
import math
sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread, imsave, imresize

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

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
# file_name = "demo/image_multi.png"
file_name = '/home/gpu_server/lyj/coco/train2014/COCO_train2014_000000524297.jpg'
# file_name = '/home/gpu_server/lyj/coco/train2014/COCO_train2014_000000524401.jpg'
# file_name = '/home/gpu_server/lyj/coco/train2014/COCO_train2014_000000524320.jpg'
# file_name = "demo/3_2_0000348.png"
# file_name = "demo/im0001.jpg"
image = imread(file_name, mode='RGB')

image_batch = data_to_input(image)

# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, pairwise_diff = predict.extract_cnn_output(
    outputs_np, cfg, dataset.pairwise_stats)

print ('shape of scmap: ', scmap.shape)
print ('shape of pairwise: ', pairwise_diff.shape)

def show_heatmaps(img, scmap, cmap="jet"):
    interp = "bilinear"
    all_joints = cfg.all_joints
    all_joints_names = cfg.all_joints_names
    subplot_width = 3
    subplot_height = math.ceil((len(all_joints) + 1) / subplot_width)
    f, axarr = plt.subplots(subplot_height, subplot_width)
    for pidx, part in enumerate(all_joints):
        plot_j = (pidx + 1) // subplot_width
        plot_i = (pidx + 1) % subplot_width
        scmap_part = np.sum(scmap[:, :, part], axis=2)
        scmap_part = imresize(scmap_part, 8.0, interp='bicubic')
        scmap_part = np.lib.pad(scmap_part, ((4, 0), (4, 0)), 'minimum')
        curr_plot = axarr[plot_j, plot_i]
        curr_plot.set_title(all_joints_names[pidx])
        curr_plot.axis('off')
        curr_plot.imshow(img, interpolation=interp)
        curr_plot.imshow(scmap_part, alpha=.5, cmap=cmap, interpolation=interp)

    curr_plot = axarr[0,0]
    curr_plot.set_title('initial image')
    curr_plot.axis('off')
    curr_plot.imshow(img)    
    plt.show()

show_heatmaps(image,scmap)
plt.waitforbuttonpress(timeout=1)

# pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
# arrows = predict.argmax_arrows_predict(scmap, locref, pairwise_diff, cfg.stride)
# visualize.show_arrows(cfg, image, pose, arrows)
# visualize.waitforbuttonpress()

# def show_pairwise_diff(img, scmap, pairwise_diff,id,cmap="jet"):
#     interp = "bilinear"
#     num_joints = cfg.num_joints
#     all_joints = []
#     all_joints_names = []
#     for i in range(num_joints):
#         if i != id:
#             all_joints.append(i)
#         all_joints_names.append(str(i))
#     stride = 8
#     half_stride = stride / 2
#     subplot_width = 3
#     subplot_height = math.ceil( (len(all_joints) ) / subplot_width )
#     f,axarr = plt.subplots(subplot_height, subplot_width)
#     anchor_map = scmap[:,:,id]
#     scmap_h,scmap_w = scmap.shape[0], scmap.shape[1]
#     h = scmap_h * stride
#     w = scmap_w * stride
#     target_maps = np.zeros([h,w,num_joints -1 ])
#     def get_pairwise_index(j_id, j_id_end, num_joints):
#         return (num_joints - 1) * j_id + j_id_end - int(j_id < j_id_end)
#
#     for index,id_end in enumerate(all_joints):
#         pair_id = get_pairwise_index(id, id_end, num_joints)
#         diff_map_x = pairwise_diff[:,:,pair_id,0]
#         diff_map_y = pairwise_diff[:,:,pair_id,1]
#         for j in range(scmap_h):
#             for i in range(scmap_w):
#                 value = anchor_map[j,i]
#                 delta_x = int(round(diff_map_x[j,i]))
#                 delta_y = int(round(diff_map_y[j,i]))
#                 target_maps[j*stride:(j+1)*stride + delta_y,i*stride:(i+1)*stride + delta_x, index] = value
#                 # target_maps[j*stride:(j+1)*stride ,i*stride:(i+1)*stride , index] = value
#     for pidx, part in enumerate(all_joints):
#         plot_j = (pidx + 1) // subplot_width
#         plot_i = (pidx + 1) % subplot_width
#         target_map = target_maps[:,:,pidx]
#         # target_map = np.lib.pad(target_map, ((4, 0), (4, 0)), 'minimum')
#         curr_plot = axarr[plot_j, plot_i]
#         curr_plot.set_title(all_joints_names[part])
#         curr_plot.axis('off')
#         curr_plot.imshow(img, interpolation=interp)
#         curr_plot.imshow(target_map, alpha=.5, cmap=cmap, interpolation=interp)
#     plt.show()
#
# show_pairwise_diff(image, scmap, pairwise_diff,12)
# plt.waitforbuttonpress(timeout=1)
