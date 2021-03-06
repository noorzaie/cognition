import numpy as np
import cv2
import os
from numpy import linalg as LA
import Image
import collections

def read_images(path):
    images = []
    for root , directories, files in os.walk(path):
        names = files
    for image in names :
        images.append(cv2.imread(path+"\\"+image, cv2.CV_LOAD_IMAGE_GRAYSCALE))
    images.remove(images[len(images)-1])
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
    idx = np.argsort(-w)
    w=w[idx]
    mv = mv[idx, :]
    return mv

def im_map(image, mean, mv):
    #this function maps an image into eigen space
    new = cv2.imread(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    new = image_as_row(new)
    new = np.subtract(new, mean)
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

def similars(image, mapped, n):
    sims = {}
    for i in range(len(mapped)) :
        sims[Euclidean_distance(image, mapped[i])] = i
    sortedsims = collections.OrderedDict(sorted(sims.items()))
    return sortedsims.items()[0:n]

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
    index = find_similar(new_image, mapped)
    print "First layer with 25 images...\nNew Picture Is Similar To "+names[index]+"\n"

    sims = similars(new_image, mapped, 10)

    images = []
    for i in range(len(sims)):
        images.append(cv2.imread(path+"\\"+names[sims[i][1]], cv2.CV_LOAD_IMAGE_GRAYSCALE))

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
        cv2.imwrite("eigenface"+str(i+1)+".jpg", row2image(normalize(mv[i], 0, 255), y, x))
    
    #mapping training images into eigen space
    for root , directories, files in os.walk(path):
        names = files
    mapped = []
    mapped = [im_map(path+"\\"+names[sims[i][1]], mean, mv) for i in range(len(sims))]

    #test code above with a new image
    new_image = im_map('test.pgm', mean, mv)
    index = find_similar(new_image, mapped)
    print "Second layer with 10 images...\nNew Picture Is Similar To "+names[sims[index][1]]

main()

