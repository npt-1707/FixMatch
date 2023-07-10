import numpy as np, random, torch
from torchvision import transforms, datasets
import torchvision
from PIL import Image
from augmentation.random_augment import RandAugment


def split_label_unlabel_valid(labels, num_label, num_classes):
    labels_per_class = num_label // num_classes
    labeled_idx = []
    unlabeled_idx = []
    valid_idx = []
    for i in range(num_classes):
        idx = np.where(np.array(labels) == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:labels_per_class])
        valid_idx.extend(idx[labels_per_class:100 + labels_per_class])
        unlabeled_idx.extend(idx[labels_per_class + 100:])
    return labeled_idx, unlabeled_idx, valid_idx


def one_hot(label, num_classes):
    return torch.eye(num_classes)[label]


def get_cifar10(args):
    trainset = torchvision.datasets.CIFAR10(root=args.root,
                                            train=True,
                                            download=True)
    labeled_idx, unlabeled_idx, valid_idx = split_label_unlabel_valid(
        trainset.targets, args.num_labels, 10)

    train_labeled_dataset = CIFAR10SSL(root=args.root, indexs=labeled_idx)

    train_unlabeled_dataset = CIFAR10SSL(root=args.root,
                                         indexs=unlabeled_idx,
                                         is_labeled=False)

    valid_dataset = CIFAR10SSL(root=args.root, indexs=valid_idx, is_valid=True)

    test_dataset = CIFAR10SSL(root=args.root, train=False)

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


def get_cifar100(args):

    trainset = torchvision.datasets.CIFAR100(root=args.root,
                                             train=True,
                                             download=True)
    labeled_idx, unlabeled_idx, valid_idx = split_label_unlabel_valid(
        trainset.targets, args.num_labels, 100)

    train_labeled_dataset = CIFAR100SSL(root=args.root, indexs=labeled_idx)

    train_unlabeled_dataset = CIFAR100SSL(root=args.root,
                                          indexs=unlabeled_idx,
                                          is_labeled=False)

    valid_dataset = CIFAR100SSL(root=args.root,
                                indexs=valid_idx,
                                is_valid=True)

    test_dataset = CIFAR100SSL(root=args.root, train=False)

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


def get_svhn(args):
    trainset = torchvision.datasets.SVHN(root=args.root,
                                         split="train",
                                         download=True)
    labeled_idx, unlabeled_idx, valid_idx = split_label_unlabel_valid(
        trainset.labels, args.num_labels, 10)

    train_labeled_dataset = SVHNSSL(root=args.root, indexs=labeled_idx)

    train_unlabeled_dataset = SVHNSSL(root=args.root,
                                      indexs=unlabeled_idx,
                                      is_labeled=False)

    valid_dataset = SVHNSSL(root=args.root, indexs=valid_idx, is_valid=True)

    test_dataset = SVHNSSL(root=args.root, split="test")

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


def get_stl10(args):
    train_labeled_dataset = STL10SSL(root=args.root, folds=args.fold)

    train_unlabeled_dataset = STL10SSL(root=args.root, split="unlabeled")

    valid_dataset = STL10SSL(root=args.root, folds=(args.fold + 1) % 10)

    test_dataset = STL10SSL(root=args.root, split="test")

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


class CIFAR10SSL(datasets.CIFAR10):

    def __init__(self,
                 root,
                 indexs=None,
                 train=True,
                 download=True,
                 is_labeled=True,
                 is_valid=False):
        super().__init__(root, train=train, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.is_labeled = is_labeled
        self.is_valid = is_valid
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]

        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect')
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)
        ])
        self.normalize = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if not self.train or self.is_valid:
            return self.normalize(img), one_hot(target, 10)

        weak = self.weak(img)
        strong = self.strong(img)

        if self.is_labeled:
            return self.normalize(weak), one_hot(target, 10)

        return self.normalize(weak), self.normalize(strong)


class CIFAR100SSL(datasets.CIFAR100):

    def __init__(self,
                 root,
                 indexs=None,
                 train=True,
                 download=True,
                 is_labeled=True,
                 is_valid=False):
        super().__init__(root, train=train, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.is_labeled = is_labeled
        self.is_valid = is_valid

        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]

        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect')
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)
        ])
        self.normalize = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if not self.train or self.is_valid:
            return self.normalize(img), one_hot(target, 100)

        weak = self.weak(img)
        strong = self.strong(img)

        if self.is_labeled:
            return self.normalize(weak), one_hot(target, 100)

        return self.normalize(weak), self.normalize(strong)


class SVHNSSL(datasets.SVHN):

    def __init__(self,
                 root,
                 indexs=None,
                 split="train",
                 download=True,
                 is_labeled=True,
                 is_valid=False):
        super().__init__(root, split=split, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        self.is_labeled = is_labeled
        self.is_valid = is_valid
        mean = [0.4409, 0.4279, 0.3868]
        std = [0.2683, 0.261, 0.2687]

        self.weak = transforms.Compose([
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect')
        ])
        self.strong = transforms.Compose([
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)
        ])
        self.normalize = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)

        if self.split == "test" or self.is_valid:
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
        std = [0.2683, 0.261, 0.2687]

        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect')
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)
        ])
        self.normalize = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)

        weak = self.weak(img)
        strong = self.strong(img)

        if self.split == "test":
            return self.normalize(img), one_hot(target, 10)

        if self.split == "train":
            return self.normalize(weak), one_hot(target, 10)

        return self.normalize(weak), self.normalize(strong)