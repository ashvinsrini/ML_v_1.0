import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import cv2
src_pth = '/Users/ashvinsrinivasan/Downloads/archive/trainingSet/trainingSet'

dst_pth_img = '/Users/ashvinsrinivasan/Desktop/MNIST/imgs'
dst_pth_output = '/Users/ashvinsrinivasan/Desktop/MNIST/output'
fldrs = os.listdir(src_pth)
fldrs = [fld for fld in fldrs if '.DS' not in fld]
for fld in tqdm(fldrs):
    fls = os.listdir(os.path.join(src_pth, fld))
    for i, f in enumerate(fls):
        img_pth = os.path.join(src_pth,fld, f)
        img = cv2.imread(img_pth)
        output = np.array([float(fld)])
        np.save(os.path.join(dst_pth_img,'{}_sample_{}.npy'.format(fld, i)), img)
        np.save(os.path.join(dst_pth_output,'{}_sample_{}.npy'.format(fld, i)), output)
        #break