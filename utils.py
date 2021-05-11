from os import walk
import numpy as np
import csv
from PIL import Image
import torch.nn as nn

def getXSamples(numSamples):
    all_files = []
    for (dirpath, dirnames, filenames) in walk("train_images"):
        all_files.extend(filenames)
    selected_files = np.random.choice(all_files,numSamples)

    reader = csv.DictReader(open('train.csv')) # read all labels from train.csv
    labels = {}

    for row in reader:                          # convert to dictionary key = filename, value = label
        labels[row['image']] = row['labels']
    X,Y = filenameToFeatureMatrix(selected_files,labels)  # convert liist of filenames as strings to Feature Matrix
    return X,Y
def filenameToFeatureMatrix(filenames,all_labels):
    '''Description: Takes a list of filenames form train_images, and converts them to Feature Matrix X
        X has the Shape N x C x H x W, where:

        N is the number of datapoints
        C is the number of channels
        H is the height of the intermediate feature map in pixels
        W is the height of the intermediate feature map in pixels

        Returns:
            X: Feature Matrix as described above.
            Y: Corresponding labels to X. In the exact same order.'''
    Y = []
    X = []
    for file in filenames:
        Y.append(all_labels[file])
        im = Image.open("train_images/"+file)
        im = im.resize((2672, 4000)) # Most frequent size of sample Image. All images must be the same size for CNN
        a = np.array(im)

        # seperate Image into 3 channels by color just like with CIFAR10
        r = a[:,:,0]
        g = a[:,:,1]
        b = a[:,:,2]


        X.append(np.asarray([r,g,b]))
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image