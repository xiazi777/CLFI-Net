import numpy as np
import imageio as reader
import os
import scipy.io as sio
from PIL import Image
from torchvision import transforms
INPUT_SIZE = (448, 448)

class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        self.preload = False

        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        self.train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        self.test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
        
        if self.preload:
            if self.is_train:
                self.train_img = [reader.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                                self.train_file_list[:data_len]]
                self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            else:
                self.test_img = [reader.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                                self.test_file_list[:data_len]]
                self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            if self.preload:
                img, target = self.train_img[index], self.train_label[index]
            else:
                img_path = os.path.join(self.root, 'images', self.train_file_list[index])
                img, target = reader.imread(img_path), self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((550, 550))(img)
            img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        else:
            if self.preload:
                img, target = self.test_img[index], self.test_label[index]
            else:
                img_path = os.path.join(self.root, 'images', self.test_file_list[index])
                img, target = reader.imread(img_path), self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((550, 550))(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class ALGAE():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        self.preload = False

        img_txt_file = open(
            os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        self.train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        self.test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

        if self.preload:
            if self.is_train:
                self.train_img = [reader.imread(os.path.join(self.root, 'image', train_file)) for train_file in
                                  self.train_file_list[:data_len]]
                self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            else:
                self.test_img = [reader.imread(os.path.join(self.root, 'image', test_file)) for test_file in
                                 self.test_file_list[:data_len]]
                self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            if self.preload:
                img, target = self.train_img[index], self.train_label[index]
            else:
                img_path = os.path.join(self.root, 'image', self.train_file_list[index])
                img, target = reader.imread(img_path), self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((550, 550))(img)
            img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        else:
            if self.preload:
                img, target = self.test_img[index], self.test_label[index]
            else:
                img_path = os.path.join(self.root, 'image', self.test_file_list[index])
                img, target = reader.imread(img_path), self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((550, 550))(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class CAR():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train

        data = sio.loadmat(os.path.join(root, 'cars_annos.mat'))
        annotations = data['annotations'].squeeze(0)
        class_names = data['class_names']

        self.train_file_list = [ann[0][0] for ann in annotations if ann[-1]==0]
        self.train_label = [ann[-2][0][0].astype('int64')-1 for ann in annotations if ann[-1]==0]

        self.test_file_list = [ann[0][0] for ann in annotations if ann[-1]==1]
        self.test_label = [ann[-2][0][0].astype('int64')-1 for ann in annotations if ann[-1]==1]
        

    def __getitem__(self, index):
        if self.is_train:
            img_path = os.path.join(self.root, self.train_file_list[index])
            img, target = reader.imread(img_path), self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((550, 550))(img)
            img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        else:
            img_path = os.path.join(self.root, self.test_file_list[index])
            img, target = reader.imread(img_path), self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((550, 550))(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class AIR():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train

        variants = []
        with open(os.path.join(root, 'data/variants.txt'), 'r') as f:
            for line in f:
                variants.append(line.strip())

        if self.is_train:
            self.train_file_list = []
            self.train_label = []
            with open(os.path.join(root, 'data/images_variant_trainval.txt'), 'r') as f:
                for line in f:
                    line = line.strip()
                    seven_digits = line[:7]
                    variant = line[8:]
                    self.train_file_list.append(os.path.join(root, 'data/images', seven_digits+'.jpg'))
                    self.train_label.append(variants.index(variant))
        else:
            self.test_file_list = []
            self.test_label = []
            with open(os.path.join(root, 'data/images_variant_test.txt'), 'r') as f:
                for line in f:
                    line = line.strip()
                    seven_digits = line[:7]
                    variant = line[8:]
                    self.test_file_list.append(os.path.join(root, 'data/images', seven_digits+'.jpg'))
                    self.test_label.append(variants.index(variant))
        
    def __getitem__(self, index):
        if self.is_train:
            img_path = self.train_file_list[index]
            img, target = reader.imread(img_path)[:-20, ...], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((550, 550))(img)
            img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        else:
            img_path = self.test_file_list[index]
            img, target = reader.imread(img_path)[:-20, ...], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((550, 550))(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':
    dataset = CAR(root='D:/z/Aproject/dataset/stanford_cars')
    print(dataset.train_label)
