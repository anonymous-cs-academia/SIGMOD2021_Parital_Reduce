from PIL import Image
import torch
import pickle
import numpy as np
#  dict_keys(['labels', 'data', 'filenames'])
class MyTrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        super(MyTrainDataset,self).__init__()
        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        self.targets = []
        with open(path, 'rb') as f:
            if "test" in path:
              input = pickle.load(f, encoding='latin1')
            else:
              input = pickle.load(f)
            self.data = input['data']
            self.targets = input['labels']

        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        '''
        return the length of dataset
        '''
        return len(self.data)
