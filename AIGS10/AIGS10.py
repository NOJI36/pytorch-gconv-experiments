from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset


class AIGS10(Dataset):
    classes = ('aeroplane','bird','boat','car','cat','chair','dog','horse','sheep','train')

    train_list = [
        '/traindata/aeroplane_train.npy', '/traindata/bird_train.npy', '/traindata/boat_train.npy', '/traindata/car_train.npy', '/traindata/cat_train.npy',
        '/traindata/chair_train.npy', '/traindata/dog_train.npy', '/traindata/horse_train.npy', '/traindata/sheep_train.npy', '/traindata/train_train.npy'
    ]

    test_list = [
        '/testdata/aeroplane_test.npy', '/testdata/bird_test.npy', '/testdata/boat_test.npy',
        '/testdata/car_test.npy', '/testdata/cat_test.npy',
        '/testdata/chair_test.npy', '/testdata/dog_test.npy', '/testdata/horse_test.npy',
        '/testdata/sheep_test.npy', '/testdata/train_test.npy'
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        #super(AIGS10, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set tor test set

        if self.train:
            paths = self.train_list
        else:
            paths = self.test_list

        self.data = []
        self.targets = []

        for idx, path in enumerate(paths):
            tmp_data = np.load(root + '/AIGS10' + path)
            tmp_data = tmp_data.transpose((0, 3, 1, 2))
            self.data.extend(tmp_data)
            self.targets.extend([idx]*len(tmp_data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)