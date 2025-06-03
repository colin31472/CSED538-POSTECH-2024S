import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np 


def get_train_loader(batch_size,valid_size=0.2):
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    np.random.shuffle(indices)
    train_idx = indices[split:]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,sampler=SubsetRandomSampler(train_idx),num_workers=0, drop_last=True)
    return train_loader


def get_valid_loader(batch_size,valid_size=0.2):
    valid_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    num_train = len(valid_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    np.random.shuffle(indices)
    valid_idx = indices[:split] 
    
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,sampler=SubsetRandomSampler(valid_idx),num_workers=0, drop_last=True)
    return valid_loader


def get_test_loader(batch_size):
    test_dataset = datasets.CIFAR10(root='./data',train=False,download=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=0, drop_last=True)
    return test_loader

def get_loader4scheduler(batch_size):
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))
    
    np.random.shuffle(indices)
    train_idx = indices[split:]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,sampler=SubsetRandomSampler(train_idx),num_workers=0, drop_last=True)
    return train_loader