import os
import sys

import numpy as np
import time
sys.path.append(os.path.dirname(__file__) + "/../")

sys.path.append("./lib/coco/PythonAPI/")
from pycocotools.coco import COCO

from scipy.misc import imread, imsave
import scipy.io as sio
from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections

class DataItem:
    pass

cfg = load_config("demo/get_training_data.yaml")

dataset = create_dataset(cfg)

ann_file = "/home/gpu_server/lyj/coco/annotations/person_keypoints_train2014.json"
file_head = '/home/gpu_server/lyj/coco/train2014/'
coco = COCO(ann_file)
imgIds = coco.getImgIds()
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

data = []
data_35 = []
pair = []
counter = 0

for imgId in imgIds:
    item = DataItem()
    pair_item = DataItem()

    img = coco.loadImgs(imgId)[0]
    item.im_path = file_head + img["file_name"]
    item.im_size = [3, img["height"], img["width"]]
    item.coco_id = imgId
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd = False)
    anns = coco.loadAnns(annIds)

    has_people = len(anns) > 0
    if has_people:
        # get CNN output :
        # scmap, locref, pairwise_diff
        input = imread(item.im_path, mode = 'RGB')
        image_batch = data_to_input(input)
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, pairwise_diff = predict.extract_cnn_output(
            outputs_np, cfg, dataset.pairwise_stats)

        item.scmap = scmap
        item.locref = locref
        item.pairwise_diff = pairwise_diff
        pair_item.pairwise_diff = pairwise_diff
        # get GT for this image
        all_person_keypoints = []
        for ann in anns: # loop through each person
            person_keypoints = []
            if ann["num_keypoints"] != 0:
                for i in range(cfg.num_joints):
                    x_coord = ann["keypoints"][3 * i]
                    y_coord = ann["keypoints"][3 * i + 1]
                    visibility = ann["keypoints"][3 * i + 2]
                    if visibility != 0: # i.e. if labeled
                        person_keypoints.append([i, x_coord, y_coord])
                all_person_keypoints.append(np.array(person_keypoints))
        if not all_person_keypoints:
            continue
        item.joints = all_person_keypoints
    else:
        continue
    counter += 1
    if counter % 10 == 0 :
        print ('processing img (%d--%d)'%(counter,counter+10))
    if counter <= 500:
        pair.append(pair_item)
    if counter <= 35000:
        data_35.append(item)
    data.append(item)
    if counter >= 10 :
        dataset = np.asarray(data)
        sio.savemat('data_10.mat',{'dataset':dataset})
        raise
#save files
pair_set = np.asarray(pair)
dataset = np.asarray(data)
dataset_35 = np.asarray(data_35)
sio.savemat('gt_output.mat',{'dataset':dataset})
sio.savemat('gt_output_35.mat',{'dataset':dataset_35})
sio.savemat('pair.mat',{'pairwise':pair_set})
print ('finished')
