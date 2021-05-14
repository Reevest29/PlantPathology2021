from os import walk
from os import path
import os
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm
ROOT_DIR = '''C:\\Users\\tjtom\PycharmProjects\PlantPathology2021'''

import torch.nn as nn

def getXSamples(numSamples=-1,verbose=False):
    all_files = []
    train_images = os.path.join(ROOT_DIR,"train_images")
    train_csv = os.path.join(ROOT_DIR,"train.csv")
    print(train_images)
    for (dirpath, dirnames, filenames) in walk(train_images):
        all_files.extend(filenames)
    
    if numSamples == -1:
        selected_files = all_files
    else:
        selected_files = np.random.choice(all_files,numSamples,replace=False)

    reader = csv.DictReader(open(train_csv)) # read all labels from train.csv
    labels = {}

    for row in reader:                          # convert to dictionary key = filename, value = label
        labels[row['image']] = row['labels']
    X,Y = filenameToFeatureMatrix(selected_files,labels,verbose=False)  # convert liist of filenames as strings to Feature Matrix
    return X,Y
def filenameToFeatureMatrix(filenames,all_labels,verbose = False):
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
    for i in tqdm(range(len(filenames)),disable=verbose):
        try:
            file = filenames[i]
            Y.append(all_labels[file])
            im = Image.open(os.path.join(ROOT_DIR,"train_images",file))
            im = im.resize((4000, 2672)) # Most frequent size of sample Image. All images must be the same size for CNN
            a = np.array(im)
            im.close()

            # seperate Image into 3 channels by color just like with CIFAR10
            r = a[:,:,0]
            g = a[:,:,1]
            b = a[:,:,2]

            X.append(np.asarray([r,g,b]))
        except (MemoryError):
            print("Memory Error Found:")
            print("/t Filename:",file)
            print("/t Label:",all_labels[file])
            print("/t ImageObject:",im)

    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y