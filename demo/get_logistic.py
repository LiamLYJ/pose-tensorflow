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
every_num = 2
save_interval = 2
max_item = 3 * every_num * save_interval

dataset = create_dataset(cfg)

ann_file = "/home/gpu_server/lyj/coco/annotations/person_keypoints_train2014.json"
file_head = '/home/gpu_server/lyj/coco/train2014/'

pairwiseDir = '/home/gpu_server/lyj/pose-tensorflow/models/coco/pairwise/'
if not os.path.exists(pairwiseDir):
    os.mkdir(pairwiseDir)

coco = COCO(ann_file)
imgIds = coco.getImgIds()
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

num_joints = 3
# num_joints = cfg.num_joints

print ('starting matlab engine.............')
import matlab
import matlab.engine
matlab_eng = matlab.engine.start_matlab()
matlab_eng.addpath('/home/gpu_server/lyj/deepcut/external/liblinear-1.94/matlab')
matlab_eng.addpath('/home/gpu_server/lyj/pose-tensorflow/matlab')
print ('finished starting matlab engine ~!')

counter = 0
data = []

X_POS = []
X_NEG = []
for imgId in imgIds:
    item = DataItem()

    img = coco.loadImgs(imgId)[0]
    item.im_path = file_head + img["file_name"]
    # item.im_size = [3, img["height"], img["width"]]
    # item.coco_id = imgId
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

        # item.scmap = scmap
        item.locref = locref
        item.pairwise_diff = pairwise_diff

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
    data.append(item)
    if (counter % every_num == 0 ):
        gt_data = np.asarray(data)
        sio.savemat('data.mat',{'dataset':gt_data})
        data = []
        index = 0
        save_features = (counter % (save_interval * every_num) == 0  )

        print ('#################################')
        print ('start to get features (%d-%d)'%(counter-every_num,counter))
        print ('#################################')
        for i in range(1,num_joints):
            for j in range(i+1,num_joints+1):
                [X_pos,X_neg] = matlab_eng.get_feat(i,j,nargout=2)
                if ((counter == every_num) or ((counter -every_num) % (save_interval * every_num ) == 0) ):
                    X_POS.append(X_pos)
                    X_NEG.append(X_neg)
                else :
                    X_POS[index] = matlab_eng.cat(1,X_POS[index],X_pos)
                    X_NEG[index] = matlab_eng.cat(1,X_neg[index],X_neg)

                if (save_features):
                    print ('#################################')
                    print ('saving computed features... (%d-%d)'%(i,j))
                    print ('#################################')
                    if os.path.exists('%sfeat_spatial_%d_%d.mat'%(pairwiseDir,i,j)):
                        tmp = sio.loadmat('%sfeat_spatial_%d_%d.mat'%(pairwiseDir,i,j))
                        old_X_pos = tmp['X_pos']
                        old_X_neg = tmp['X_neg']
                        save_X_pos = np.concatenate((old_X_pos,np.asarray(X_POS[index])),axis = 0)
                        save_X_neg = np.concatenate((old_X_neg,np.asarray(X_NEG[index])),axis = 0)
                        sio.savemat('%sfeat_spatial_%d_%d.mat'%(pairwiseDir,i,j),{'X_pos':save_X_pos,'X_neg':save_X_neg})
                    else:
                        print ('this is first time saving...(%d-%d)'%(i,j))
                        save_X_pos = np.asarray(X_POS[index])
                        save_X_neg = np.asarray(X_NEG[index])
                        sio.savemat('%sfeat_spatial_%d_%d.mat'%(pairwiseDir,i,j),{'X_pos':save_X_pos,'X_neg':save_X_neg})
                    print ('saved done')
                index += 1
        # clear memory for saving X_POS and X_NEG
        if (save_features):
            X_POS = []
            X_NEG = []
    if counter >= max_item:
        print ('ok, we have trained with enough items (%d)'%max_item)
        break

print ('counter is :', counter)

print ('#################################')
print ('start to train logistic regression ............')
print ('#################################')

for i in range(1,num_joints):
    for j in range(i+1, num_joints+1):
        tmp = sio.loadmat('%sfeat_spatial_%d_%d.mat'%(pairwiseDir,i,j))
        train_X_pos = matlab.single(tmp['X_pos'].tolist())
        train_X_neg = matlab.single(tmp['X_neg'].tolist())
        matlab_eng.train_logistic_model(i,j,train_X_pos,train_X_neg,pairwiseDir,nargout = 0)
        print ('trained with %d-%d'%(i,j))

print ('finished all !!!!')
