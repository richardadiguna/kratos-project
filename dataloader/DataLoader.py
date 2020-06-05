import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, config):
        self.im_size = config.image_shape
        self.batch_size = config.batch_size
        self.dataset_path = config.dataset_path

        if not os.path.isfile(config.dataset_path):
            self.create_data_txt(config.dataset_dir)

        _, self.file_extension = os.path.splitext(config.dataset_path)
        self.get_dataset(config.dataset_path)

        self.total_epochs = 0
        self.current_index = 0
        self.iteration = 0

    def randomize(self, a, b):
        permutation = np.random.permutation(a.shape[0])
        shuffled_a = a[permutation]
        shuffled_b = b[permutation]
        return shuffled_a, shuffled_b

    def create_data_txt(self, dataset_dir):
        fnames = os.listdir(dataset_dir)

        def isdir(x):
            if x == 'PASSPORT' or x == 'OTHER':
                return False
            else:
                if os.path.isdir(os.path.join(dataset_dir, x)):
                    return True

        dirnames = [name for name in fnames if isdir(name)]

        with open(self.dataset_path, 'w') as f:
            lines = []
            label = 0
            for idx, dir_ in enumerate(dirnames):
                # Change this conditional statement
                # if you have different class distribution
                if dir_ == 'BACKGROUND':
                    label = 0
                elif dir_ == 'E-KTP':
                    label = 1
                elif dir_ == 'KTP':
                    label = 2
                elif dir_ == 'DUKCAPIL':
                    label = 3

                images_path = os.listdir(os.path.join(dataset_dir, dir_))
                desc = 'Writing kratos-dataset.txt for input dataset'
                for path in tqdm(images_path, desc=desc):
                    if os.path.isfile(os.path.join(dataset_dir, dir_, path)):
                        fpath = os.path.join(dataset_dir, dir_, path)
                        line = ''.join(fpath + ' ' + str(label) + ' ' + '\n')
                        lines.append(line)
            random.shuffle(lines)
            for line in lines:
                f.write(line)

    def read_file_names(self, image_list_file):
        filenames = []
        with open(image_list_file, 'r') as f:
            for line in tqdm(f, desc=image_list_file):
                filenames.append(line)
        return filenames

    def read_images_from_disk(self, filenames):
        images = []
        labels = []

        for i in range(0, len(filenames)):
            img = cv2.imread(filenames[i])

            if img is None:
                print(filenames[i])
                pass

            if img.shape != (128, 128):
                img = cv2.resize(img, (128, 128))

            images.append(img)

        return images

    def get_dataset(self, dataset_path):
        if self.file_extension == '.npz':
            data = np.load(dataset_path)

            X = data['features']
            y = data['labels']

            X, y = self.randomize(X, y)

            self.X_train, self.X_val, self.y_train, self.y_val = \
                train_test_split(X, y, test_size=0.20)

        elif self.file_extension == '.txt':

            txt_data = self.read_file_names(dataset_path)

            X = [fname.split(' ')[0] for fname in txt_data]
            y = [fname.split(' ')[1] for fname in txt_data]

            self.X_train, self.X_val, self.y_train, self.y_val = \
                train_test_split(X, y, test_size=0.20)

        self.num_train = len(self.X_train)
        self.num_val = len(self.X_val)

    def slice_data(self, X, y, file_extension, indices):

        if file_extension == '.npz':
            image_batch = X[indices]
            label_batch = y[indices]

        elif file_extension == '.txt':
            image_batch = self.read_images_from_disk(
                [X[index] for index in indices])
            label_batch = [y[index] for index in indices]

        return image_batch, label_batch

    def get_batch(self, trainable=True, random=True):
        X = self.X_train if trainable else self.X_val
        y = self.y_train if trainable else self.y_val

        num_files = self.num_train if trainable else self.num_val
        end_index = self.current_index + self.batch_size

        if random:
            self.indices = np.random.choice(
                num_files,
                self.batch_size)
        else:
            self.indices = np.arange(
                self.current_index,
                end_index)

            if end_index > self.num_files:
                self.indices[self.indices >= num_files] = np.arange(
                    0, np.sum(self.indices >= self.num_files))

        image_batch, label_batch = \
            self.slice_data(
                X,
                y,
                self.file_extension,
                self.indices)

        image_batch = np.reshape(
            np.squeeze(
                np.stack(
                    [image_batch])),
            newshape=(self.batch_size, self.im_size, self.im_size, 3))

        label_batch = np.stack(label_batch)

        self.current_index = end_index

        if self.current_index > num_files:
            self.current_index = self.current_index - num_files

        return image_batch, label_batch
