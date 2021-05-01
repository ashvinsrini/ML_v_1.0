import pandas as pd
import numpy as np
import random

random.seed(10)
import os

img_src_pth = '/Users/ashvinsrinivasan/Desktop/MNIST/imgs'
out_src_pth = '/Users/ashvinsrinivasan/Desktop/MNIST/output'
save_pth = '/Users/ashvinsrinivasan/Desktop/MNIST'
fls = os.listdir(img_src_pth)
# print(fls[0:10])
random.shuffle(fls)
# print(fls[0:10])
output_pth = []
input_pth = []
for f in fls:
    input_pth.append(os.path.join(img_src_pth, f))
    output_pth.append(os.path.join(out_src_pth, f))

df = pd.concat([pd.Series(input_pth), pd.Series(output_pth)], axis=1)
#df.to_csv(os.path.join(save_pth, 'dataset.csv'), header=False)
ind_tr = 0.8*df.shape[0]
ind_v = 0.1*df.shape[0]
df.loc[0:ind_tr].to_csv(os.path.join(save_pth, 'train.csv'), header=False)
df.loc[ind_tr:ind_tr+ind_v].to_csv(os.path.join(save_pth, 'val.csv'), header=False)
df.loc[ind_tr+ind_v:ind_tr+ind_v+ind_v].to_csv(os.path.join(save_pth, 'test.csv'), header=False)