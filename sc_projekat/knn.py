import numpy as np
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

for x in range(0,10):
    print x

    naziv = 'vid' + `x` + '/vid' + `x` + ' 026.jpg';
    print naziv
    img = cv2.imread(naziv)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = 20
    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 100
    lines = cv2.HoughLinesP(im_bw,1,np.pi/180,100,minLineLength,maxLineGap)
    #print lines


    blank_image = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),2)
    imgg = rgb2gray(blank_image)
    img_bw = imgg < 0.01
    img_bw = 1 - img_bw
    #plt.imshow(img_bw, 'gray')
    #plt.show()
    cv2.imwrite('houghlines'+ `x` +'.jpg',blank_image)
