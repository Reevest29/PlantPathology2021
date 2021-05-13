from os import walk
import numpy as np
import csv
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable

def getXSamples(numSamples=-1):
    all_files = []
    for (dirpath, dirnames, filenames) in walk("train_images"):
        all_files.extend(filenames)
    
    if numSamples == -1:
        selected_files = all_files
    else:
        selected_files = np.random.choice(all_files,numSamples,replace=False)

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
    
def train(model, loader, loss_fn, optimizer, num_epochs = 1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader):
            print(t)
            x_var = Variable(x.type(gpu_dtype))
            y_var = Variable(y.type(gpu_dtype).long())

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        with torch.no_grad():
            x_var = Variable(x.type(gpu_dtype))

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    
def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()