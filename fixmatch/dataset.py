import numpy as np
import random
from torchvision import transforms, datasets
import torchvision
from PIL import Image
from augmentation.random_augment import RandAugment

random.seed(41)

def calculate_mean_std(dataset):
    pixel_sum = np.zeros(3)
    pixel_squared_sum = np.zeros(3)
    num_images = len(dataset)

    size = np.array(dataset[0][0]).shape[0]
    for idx in range(len(dataset)):
        pixels = np.array(dataset[idx][0]) / 255.0
        pixel_sum += np.sum(pixels, axis=(0, 1))
        pixel_squared_sum += np.sum(pixels**2, axis=(0, 1))

    mean = pixel_sum / (num_images * size * size)
    variance = (pixel_squared_sum / (num_images * size * size)) - mean**2
    std = np.sqrt(variance)
    #round to 4 decimal places
    mean = np.round(mean, 4)
    std = np.round(std, 4)
    return mean, std


def split_label_unlabel(labels, num_label, num_classes):
    labels_per_class = num_label // num_classes
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(np.array(labels) == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:labels_per_class])
        unlabeled_idx.extend(idx[labels_per_class:])
    return labeled_idx, unlabeled_idx

def one_hot(label, num_classes):
    return np.eye(num_classes)[label]

def get_cifar10(num_label):
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
    labeled_idx, unlabeled_idx = split_label_unlabel(trainset.targets, num_label, 10)
    
    train_labeled_dataset = CIFAR10SSL(root="data", indexs=labeled_idx, train=True)

    train_unlabeled_dataset = CIFAR10SSL(root="data", indexs=unlabeled_idx, train=True, is_labeled=False)
    
    test_dataset = CIFAR10SSL(root='data', train=False)
    
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_cifar100(num_label):
    
    trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True)
    labeled_idx, unlabeled_idx = split_label_unlabel(trainset.targets, num_label, 100)
    
    train_labeled_dataset = CIFAR100SSL(root="data", indexs=labeled_idx)

    train_unlabeled_dataset = CIFAR100SSL( root="data", indexs=unlabeled_idx, is_labeled=False)
    
    test_dataset = CIFAR100SSL(root='data', train=False)
    
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_svhn(num_label):    
    trainset = torchvision.datasets.SHVN(root='data', split="train", download=True)
    labeled_idx, unlabeled_idx = split_label_unlabel(trainset.labels, num_label, 10)
    
    train_labeled_dataset = SVHNSSL( root="data", indexs=labeled_idx)

    train_unlabeled_dataset = SVHNSSL( root="data", indexs=unlabeled_idx, is_labeled=False)
    
    test_dataset = SVHNSSL(root='data', split="test")
    
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_stl10(fold=0):
    train_labeled_dataset = STL10SSL(root='data', folds=fold)
    
    train_unlabeled_dataset = STL10SSL(root='data', split="unlabeled")
    
    test_dataset = STL10SSL(root='data', split="test")
    
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs=None, train=True, download=True, is_labeled=True):
        super().__init__(root, train=train, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.is_labeled = is_labeled
        
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470 , 0.2435, 0.2616]
        
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if not self.train:
            return self.normalize(img), one_hot(target, 10)
        
        weak = self.weak(img)
        strong = self.strong(img)
        
        if self.is_labeled:
            return self.normalize(weak), one_hot(target, 10)
        
        return self.normalize(weak), self.normalize(strong)


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs=None, train=True, download=True, is_labeled=True):
        super().__init__(root, train=train, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if not self.train:
            return self.normalize(img), one_hot(target, 100)

        weak = self.weak(img)
        strong = self.strong(img)

        if self.is_labeled:
            return self.normalize(weak), one_hot(target, 100)
        
        return self.normalize(weak), self.normalize(strong)
    
class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs=None, split="train", download=True):
        super().__init__(root, split=split, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
            
        mean = [0.4409, 0.4279, 0.3868]
        std = [0.2683, 0.261 , 0.2687]
        
        self.weak = transforms.Compose([
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.split == "test":
            return self.normalize(img), one_hot(target, 10)
        
        weak = self.weak(img)
        strong = self.strong(img)

        if self.is_labeled:
            return self.normalize(weak), one_hot(target, 10)
        
        return self.normalize(weak), self.normalize(strong)
    
class STL10SSL(datasets.STL10):
    def __init__(self, root, split="train", download=True, folds=None):
        if folds is not None:
            super().__init__(root, split=split, download=download, folds=folds)
        super().__init__(root, split=split, download=download)
            
        mean = [0.4409, 0.4279, 0.3868]
        std = [0.2683, 0.261 , 0.2687]
        
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        weak = self.weak(img)
        strong = self.strong(img)
        
        if self.split == "test":
            return self.normalize(img), one_hot(target, 10)

        if self.is_labeled:
            return self.normalize(weak), one_hot(target, 10)
        
        return self.normalize(weak), self.normalize(strong)