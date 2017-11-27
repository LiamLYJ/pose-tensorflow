import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input


cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
# file_name = "demo/image.png"
file_name = "demo/3.jpg"
image = imread(file_name, mode='RGB')

image_batch = data_to_input(image)

# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})

# for name,_ in outputs_np.items():
#     print (name)
# raise

scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

# Extract maximum scoring location from the heatmap, assume 1 person
pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
# print ('poseshapeis',pose.shape)
# print ('scmap shape is', scmap.shape)
# print ('locref shape is',locref.shape)
# # print ('value of scmap is', scmap[:,:,0])
# print ('value of locref is 0 index',locref[:,:,0,0])
# print ('value of locref is 1 index',locref[:,:,0,1])
# # print (pose)
# raise
# Visualise
visualize.show_heatmaps(cfg, image, scmap, pose)
visualize.waitforbuttonpress()
