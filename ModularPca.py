import numpy as np
import cv2
import os
from numpy import linalg as LA
import math

def read_images(path):
    images = []
    for root , directories, files in os.walk(path):
        names = files
    names.remove(names[len(names)-1])
    for image in names :
        for i in range(4):
            images.append(Slice(cv2.imread(path+"\\"+image, cv2.CV_LOAD_IMAGE_GRAYSCALE), 4)[i])
    return images

def normalize (X, low , high , dtype = None ):
    #this function scales pixels of an eigenface into range of (0, 255)
    X = np. asarray (X)
    minX , maxX = np. min (X), np. max (X)
    # normalize to [0...1].
    X = X - float ( minX )
    X = X / float (( maxX - minX ))
    # scale to [ low ... high ].
    X = X * (high - low )
    X = X + low
    if dtype is None :
        return np. asarray (X)
    return np. asarray (X, dtype = dtype )

def image_as_row(x) :
    #this function reshapes a 2d image to a 1d vector
    vi = np.concatenate([i for i in x], axis=0)
    return vi

def row2image(evs, x, y) :
    b = np.array([evs[y*i:y*(i+1)] for i in range(x)])
    return b

def covariance(m):
    dot = np.dot(m, m.transpose())
    return dot

def eigenvector(m, diff) :
    w, v = LA.eig(m)
    mv = np.dot(v, diff)
    return mv

def im_map(image, mean, mv):
    #this function maps an image into eigen space
    new = cv2.imread(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    slices = Slice(new, 4)
    s = []
    for i in slices :
        s.append(np.subtract(image_as_row(i), mean))
    u = []
    u2 = []
    
    for j in range(len(s)) :
        face = np.dot(s[0], mv[0].transpose())
        u.append(face)
        for i in range(mv.shape[0]-1) :
            face = np.dot(s[j], mv[i+1].transpose())
            u.append(face)
        u2.append(u)
        u = []
    return u2

def Euclidean_distance(v1, v2) :
    #فاصله ي اقليدسي!
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np. sqrt (np. sum (np. power ((v1-v2) ,2)))

def find_similar(image, mapped) :
    #find similar images based on euclidean distance
    closest = Euclidean_distance(image, mapped[0])
    index = 0
    for i in range(len(mapped)-1) :
        if Euclidean_distance(image, mapped[i+1]) < closest :
            closest = Euclidean_distance(image, mapped[i+1])
            index = i+1
    return index

def Slice(image, n) :
    d = math.sqrt(n)
    d = int(d)
    x = np.array(image).shape[0]/d
    y = np.array(image).shape[1]/d
    l = []
    for i in range(0, d):
        for j in range(0, d):
            l.append(image[i*x:(i+1)*x, j*y:(j+1)*y])
    return l

def main():
    #path of training images
    path = "images"
    
    #read training images
    images = read_images(path)

    #reshape images from 2d matrix to 1d vector
    vector = []
    for image in images :
        vector.append(image_as_row(image))
    vector = np.array(vector)
    
    #compute average of images
    mean = vector.mean(0)

    #compute difference of each image from average image
    diff = np.subtract(vector, mean)

    #compute covariance of difference's matrix and extract eigenvectors
    cov = covariance(diff)
    mv = eigenvector(cov, diff)

    #save eigenfaces in current directory
    y = images[0].shape[0]
    x = images[0].shape[1]

    for i in range(len(mv)):
        cv2.imwrite("eigenfaces\\eigenface"+str(i+1)+".jpg", row2image(normalize(mv[i], 0, 255), y, x))
        
    #mapping training images into eigen space
    for root , directories, files in os.walk(path):
        names = files
    mapped = [im_map(path+"\\"+names[i], mean, mv) for i in range(len(names)-2)]

    #test code above with a new image
    new_image = im_map('test.pgm', mean, mv)

    #compute distance of each slice of an image and sum them to find nearest images
    dd = 0    
    for i in range(len(new_image)) :
        dd += Euclidean_distance(mapped[0][i], new_image[i])
    d = 0
    index = 0
    for i in range(len(mapped)) :
        for j in range(len(new_image)) :
            d += Euclidean_distance(mapped[i][j], new_image[j])
        if abs(d) < dd :
            dd = d
            index = i
        d = 0
            
    print "New Picture Is Similar To "+names[index]

main()

