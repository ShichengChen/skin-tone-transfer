import cv2
import numpy as np

img=cv2.imread('/home/csc/skin-tone-transfer/data/skinColors/skintone.jpeg')
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
for color in [[33,1,2],[144,110,100],[162,116,101],[193,144,129],[237,207,197],
              [109,95,104],[156,123,104],[186,145,117],[201,160,132],[230,198,173]]:
    img=np.ones((512,512,3))*color[::-1]
    cv2.imshow('img',img)
    cv2.imwrite('/home/csc/skin-tone-transfer/data/images/skinStyle/skintone_{}.png'.format(color),img)
while(cv2.waitKey(0)):pass
#RGB
#b5 33 1 2 [33,1,2]
#b4 144 110 100 [144,110,100]
#b3 162 116 101 [162,116,101]
#b2 193 144 129 [193,144,129]
#b1 237 207 197 [237,207,197]

#L5 109 95 104 [109,95,104]
#L4 156 123 104 [156,123,104]
#L3 186 145 117 [186,145,117]
#L2 201 160 132 [201,160,132]
#L1 230 198 173 [230,198,173]

# [[33,1,2],[144,110,100],[162,116,101],[193,144,129],[237,207,197],
# [109,95,104],[156,123,104],[186,145,117],[201,160,132],[230,198,173]]