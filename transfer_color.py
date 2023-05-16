import numpy as np
import cv2
import os
import os

import numpy as np,glob,shutil,tqdm
from os.path import join,exists,basename,dirname,splitext

#Color Transfer between Images 	Erik Reinhard

def get_mean_and_std(x,mask=None):
    if mask is None:
        x_mean, x_std = cv2.meanStdDev(x)
    else:
        x_mean0, x_std0 = cv2.meanStdDev(x[:,:,0][mask[:,:,0]])
        x_mean1, x_std1 = cv2.meanStdDev(x[:,:,1][mask[:,:,1]])
        x_mean2, x_std2 = cv2.meanStdDev(x[:,:,2][mask[:,:,2]])
        x_mean, x_std = np.array([x_mean0,x_mean1,x_mean2]), np.array([x_std0,x_std1,x_std2])
    return x_mean, x_std


prefix='/extension/var-www/csc'
if not exists(prefix):
    prefix='/home/csc'
samplenum=1000
imgpaths=sorted(glob.glob(prefix+'/human/trainingSet/images/*.jpg'))[:samplenum]
imgpaths=sorted(glob.glob(prefix+'/human/segTrainingSet/NV1/images/*.png'))[:samplenum]
maskpaths=sorted(glob.glob(prefix+'/human/segTrainingSet/NV1/masksHuman/*.png'))[:samplenum]
backgroundpaths=sorted(glob.glob(prefix+'/human/background_filtered/all_filtered/*'))
# stylepaths=sorted(glob.glob('data/images/skinStyle/*'))[::-1]
styledir='data/images/skinStyle/images'
styledir='/home/csc/human/craftDataset/dalle/images2'

stylepaths=sorted(glob.glob(join(styledir,'*')))[::-1]
# stylepaths=sorted(glob.glob('/home/csc/skin-tone-transfer/data/skinColors/*'))[::-1]
print(stylepaths)
for idx in tqdm.tqdm(range(len(imgpaths))):
    for coloridx,colorpath in enumerate(stylepaths):
        imgpath=imgpaths[idx]
        maskpath=maskpaths[idx]
        backgroundpath=backgroundpaths[idx]
        s = cv2.resize(cv2.imread(imgpath),(512,512))
        smask=cv2.resize(cv2.imread(maskpath),(512,512))
        bg=cv2.resize(cv2.imread(backgroundpath),(512,512))
        oris=s.copy()


        t=cv2.resize(cv2.imread(colorpath),(512,512))
        tmask=np.all(t[:,:,:]<250,axis=2)
        tmask=np.stack([tmask,tmask,tmask],axis=2)
        s = cv2.cvtColor(s,cv2.COLOR_BGR2LAB)
        t = cv2.cvtColor(t,cv2.COLOR_BGR2LAB)



        s_mean, s_std = get_mean_and_std(s)
        # t_mean, t_std = get_mean_and_std(t,tmask)
        # cv2.imshow('t',tmask.astype(np.uint8)*255)
        # cv2.waitKey(1)
        t_mean, t_std = get_mean_and_std(t,tmask)

        height, width, channel = s.shape
        s_mean, s_std, t_mean, t_std = s_mean.reshape(3), s_std.reshape(3), \
            t_mean.reshape(3), t_std.reshape(3)
        # for i in range(0,height):
        #     for j in range(0,width):
        #         for k in range(0,channel):
        #             x = s[i,j,k]
        #             x = ((x-s_mean[k])*max(t_std[k]/s_std[k],1))+t_mean[k]
        #             # round or +0.5
        #             # print(x)
        #             x = round(x)
        #             # boundary check
        #             x = 0 if x<0 else x
        #             x = 255 if x>255 else x
        #             s[i,j,k] = x
        s_mean, s_std, t_mean, t_std = s_mean.reshape(1,1,3), s_std.reshape(1,1,3), \
            t_mean.reshape(1,1,3), t_std.reshape(1,1,3)
        s = (s-s_mean)*np.maximum(t_std/s_std,np.ones((1,1,3)))+t_mean
        s = np.clip(s,0,255).astype(np.uint8)
        s = cv2.cvtColor(s,cv2.COLOR_LAB2BGR)
        t = cv2.cvtColor(t,cv2.COLOR_LAB2BGR)
        dirout=prefix+'/human/craftDataset/traningSet/styleTransfer/style_'+str(coloridx)+'/images'
        os.makedirs(dirout,exist_ok=True)
        savepath=os.path.join(dirout,os.path.basename(imgpath))

        s[np.logical_not(smask)]=bg[np.logical_not(smask)]

        # s = cv2.normalize(s, None, 0, 255,
        #                            cv2.NORM_MINMAX, cv2.CV_8UC1)

        cv2.imwrite(savepath,s)
        cv2.namedWindow('style',cv2.WINDOW_NORMAL)
        cv2.imshow('style',s)
        cv2.namedWindow('target',cv2.WINDOW_NORMAL)
        cv2.imshow('target',t)

        cv2.waitKey(1)
