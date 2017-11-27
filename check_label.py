import sys
sys.path.append("./lib/coco/PythonAPI/")
from pycocotools.coco import COCO
import numpy as np
import cv2

annotations_file = '/home/gpu_server/lyj/pose-tensorflow/annotation.npz'
annotations = np.load(annotations_file)
imgId = annotations['id'][0]
print ('imgId:',imgId)
GT_kp = annotations['kp'][0]
GT_box = annotations['box'][0]

# bbox [x1,y1,x2,y2,1(person cat_id)]
# kp :19 = origial 17 + 2 poitns
# id : img_id in COCO data
# counter : end index for imgIds
tmp_file = "/home/gpu_server/lyj/coco/annotations/person_keypoints_train2014.json"
coco = COCO(tmp_file)

data = coco.loadImgs([imgId])[0]

file_head = '/home/gpu_server/lyj/train2014/'
img = cv2.imread(file_head + data['file_name'])
i = 0
while (i <len(GT_kp)):
    assert (len(GT_kp) == len(GT_box))
    bbox = GT_box[i]
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0))
    p = GT_kp[i]
    x = p[-3]
    y = p[-2]
    cv2.circle(img,(x,y), 50, (0,0,255))
    cv2.imshow('img',img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()
    i += 1
