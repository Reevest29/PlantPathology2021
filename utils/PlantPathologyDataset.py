import torch
from torch.utils.data import Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
from skimage import io
from skimage.transform import rescale,resize
import csv
class PlantPathologyDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, csv_file, root_dir, mappings_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mappings = {}
        reader = csv.DictReader(open(mappings_dir)) # read all labels from train.csv

        for row in reader:
            self.mappings[row['Class']] = int(row['Value'])


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])


        image = io.imread(img_name)
        image_resized = resize(image, (4000, 2672))
        image_rescaled = rescale(image_resized, .10, multichannel=True)
        image_rescaled = image_rescaled.reshape(3,image_rescaled.shape[0],image_rescaled.shape[1])

        label = str(self.labels.iloc[idx, 1])
        num_label = self.mappings[label] # get numerical label
        sample = (image_rescaled,num_label)
        return sample