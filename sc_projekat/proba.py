import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.measure import regionprops
from sklearn.datasets import fetch_mldata
from skimage.morphology import dilation
from skimage.morphology import square, diamond, disk
import cv2
import cPickle, gzip, numpy
import matplotlib.pyplot as plt

def distance(x,y):
    return np.sum(np.logical_xor(x,y))

def transform(sl):
    northern = north(sl)
    western = west(sl)
    dimensions = (28,28)
    nova_slika = np.zeros(dimensions)
    nova_slika[0:28 - northern, 0:28 - western] = sl[northern:28, western:28]
    #nova_slika = nova_slika.reshape(28*28)
    return nova_slika

def north(sl):
    dim = sl.shape
    for r in range(0,dim[0]):
        for c in range(0, dim[1]):
            if(sl[r,c]==1):
                return r
    return 0

def south(sl):
    dim = sl.shape
    for r in range(0,dim[0]):
        for c in range(0, dim[1]):
            if(sl[dim[0]-1-r,c]==1):
                return dim[0]-1-r
    return 0

def west(sl):
    dim = sl.shape
    for c in range(0,dim[1]):
        for r in range(0, dim[0]):
            if(sl[r,c]==1):
                return c
    return 0

def east(sl):
    dim = sl.shape
    for c in range(0,dim[1]):
        for r in range(0, dim[0]):
            if(sl[r,dim[1]-1-c]==1):
                return dim[1]-1-c
    return 0

def no_green(img):
    for col in range(0,640):
        for row in range(0,480):
            if img[row,col,1] > 1.3*img[row,col,0] or img[row, col, 1] > 1.3 * img[row, col, 2]:
                    img[row,col,0] = 0
                    img[row, col, 1] = 0
                    img[row, col, 2] = 0

def no_dots(img_bw):
    for col in range(0,640):
        for row in range(0,480):
            dim = img_bw.shape
            left = max(col - 2,0)
            right = min(col + 2,dim[0])
            bottom = min(row + 2,dim[0])
            top = max(row - 2,0)
            okolina = img_bw[left:right,top:bottom]
            if np.count_nonzero(okolina == 1) < 2:
                img_bw[left:right, top:bottom] = np.zeros_like(img_bw[left:right, top:bottom])


class Tracker:
    bbox = []
    code = 0
    intersected = False
    image = []
    def __init__(self, bbox,intersected, code,image):
        self.bbox = bbox
        self.intersected = intersected
        self.code = code
        self.image = image

    def update(self, centroid, bbox, rect):
        self.bbox = bbox

def dodaj_u_listu(element):
    global counterr
    global all_elements
    for i in xrange(len(all_elements)):
        cur_el_bb = all_elements[i].bbox
        w_razlika = abs(cur_el_bb[0] - element.bbox[0])
        h_razlika = abs(cur_el_bb[1] - element.bbox[1])
        razlika = w_razlika + h_razlika
        if razlika < 30:
            element.code = all_elements[i].code
            element.image = all_elements[i].image
            if element.intersected == False and all_elements[i].intersected == True:
                element.intersected = True
            all_elements[i] = element
            return
    element.code = counterr
    counterr = counterr + 1
    all_elements.append(element)




all_elements = []
counterr = 0

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

mnist_images = train_set[0]
mnist_labels = train_set[1]

for brojac in range(0,50000):
    slk = mnist_images[brojac]
    slk = slk.reshape(28,28)
    slk = slk < 0.5
    slk = 1 - slk
    slk = transform(slk)
    mnist_images[brojac] = slk.reshape(28*28)


text_file = open("out.txt", "w")

for x in range(0,10):
    all_elements = []
    hough = imread('houghlines'+ `x` +'.jpg')
    hough_gray = rgb2gray(hough)
    hju_bw = hough_gray < 0.01
    hju_bw = 1 - hju_bw

    print x
    for brojac in range(1,121):
        if brojac < 10:
            brstr = '00' + `brojac`
        if brojac > 9 and brojac < 100:
            brstr = '0' + `brojac`
        if brojac>99 and brojac<1000:
            brstr = `brojac`
        ci_name = 'vid'+ `x` +'/vid'+ `x` +' ' + brstr + '.jpg'
        img = imread(ci_name)
        imgg = rgb2gray(img)
        img_bw = imgg < 0.75
        img_bw = 1 - img_bw

        str_elem = disk(3)
        sldil = dilation(img_bw, selem=str_elem)

        limg = label(sldil)
        regions = regionprops(limg)

        for region in regions:
            b = region.bbox
            h = b[2] - b[0]
            w = b[3] - b[1]
            if w > 13 or h > 13:
                #slcc = img_bw[b[0]:b[2], b[1]:b[3]]
                #brtimg = slcc[west(slcc):east(slcc), north(slcc):south(slcc)]
                #my_bbox = [west(slcc),east(slcc), north(slcc),south(slcc)]
                slika = hough[b[0]:b[2], b[1]:b[3]]
                slikica = img_bw[b[0]:b[2], b[1]:b[3]]
                maska = np.zeros((28,28))
                s_dim = slikica.shape
                s_width = min(28,s_dim[0])
                s_height = min(28, s_dim[1])
                maska[0:s_width,0:s_height] = slikica[0:s_width,0:s_height]
                maska = transform(maska)
                presao = len(np.unique(slika)) > 5
                new_elem = Tracker(b,presao,0,maska)
                dodaj_u_listu(new_elem)

    zbir = 0
    for el in all_elements:
        if el.intersected:
            minimalna = 1000;
            minindex = 0;
            tindex = 0;
            for trenutna in mnist_images:
                dist =  distance(el.image,trenutna.reshape(28,28))
                if dist<minimalna:
                    minimalna = dist
                    minindex = tindex
                tindex = tindex + 1
            zbir = zbir + mnist_labels[minindex]
    text_file.write('video-' +`x`+ '.avi\t' +`zbir`+  "\n")


    print zbir
