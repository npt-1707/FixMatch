import torchvision
from torch.utils.data import ConcatDataset

def load_cifar10():
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True)

    return trainset, testset


def load_svhn():
    trainset = torchvision.datasets.SVHN(root='./data',
                                         split='train',
                                         download=True)
    testset = torchvision.datasets.SVHN(root='./data',
                                        split='test',
                                        download=True)
    # extraset = torchvision.datasets.SVHN(root='./data',
    #                                      split='extra',
    #                                      download=True)
    return trainset, testset


def load_cifar100():
    trainset = torchvision.datasets.CIFAR100(root='./data',
                                             train=True,
                                             download=True)

    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True)

    return trainset, testset

def load_stl10():
    trainset = torchvision.datasets.STL10(root='./data',
                                          split='train',
                                          download=True)
    
    unlabelset = torchvision.datasets.STL10(root='./data',
                                            split='unlabeled',
                                            download=True)

    testset = torchvision.datasets.STL10(root='./data',
                                         split='test',
                                         download=True)

    return trainset, unlabelset, testset
        