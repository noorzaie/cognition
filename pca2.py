import numpy as np
import cv2
import os
from numpy import linalg as LA
import Image

def read_images(path):
    images = []
    for root , directories, files in os.walk(path):
        names = files
    for image in names :
        images.append(cv2.imread(path+"\\"+image, cv2.CV_LOAD_IMAGE_GRAYSCALE))
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
    #new = cv2.imread(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #new = image_as_row(image)
    new = np.subtract(image, mean)
    u = []
    face = np.dot(new, mv[0].transpose())
    u.append(face)
    for i in range(mv.shape[0]-1) :
        face = np.dot(new, mv[i+1].transpose())
        u.append(face)
    return u

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

def main():
    #path of training images
    
    #read training images
    images = []
    for i in range(25) :
        path = "images\\s"+str(i+1)
        imagesi = read_images(path)
        for j in imagesi :
            images.append(j)
        
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
    mapped = []
    for i in range(25) :
        path = "images\\s"+str(i+1)
        imagesi = read_images(path)
        vectori = []
        for image in imagesi :
            vectori.append(image_as_row(image))
        vectori = np.array(vectori)
        meani = vectori.mean(0)
        mapped.append(meani)

    mapped2 = [im_map(mapped[i], mean, mv) for i in range(len(mapped)-1)]
        
    for root , directories, files in os.walk(path):
        names = files

    #test code above with a new image
    new = cv2.imread('test.pgm', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    new = image_as_row(new)
    new_image = im_map(new, mean, mv)
    index = find_similar(new_image, mapped2)
    print "New Picture Is Similar To s"+str(index+1)

main()

