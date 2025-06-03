import argparse
import os 
import numpy as np
import torch
from absl import logging
import torchvision.models as models
from trainers.image_trainer import image_trainer
from data_loader import get_train_loader, get_valid_loader, get_test_loader
from models.ResNet import ResNet18

def get_image_model(model_name, pretrained):
    # TODO: making caching from trained
    if("ResNet18" == model_name):
        model = ResNet18()
    else:
        model = None

    if(model == None):
        raise ValueError(f"{model_name} is not a valid model name")
    if(torch.cuda.is_available()):
        model = model.cuda()

    return model


def main(config):
    logging.info(config)
    # Setting seeds for reproducibility
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    if(torch.cuda.is_available()):
        device = torch.device("cuda:0")
        logging.info(f"use device {device}")

    logging.info(f"loading image model {config.model_name}")
    model = get_image_model(config.model_name, config.from_pretrained)
    
    ################################################################
    logging.info("*"*30)
    logging.info(f"learning rate info:{config.initial_lr}, {config.lr_scheduler_name}, {config.new_technique_args}")
    logging.info("*"*30)
    ################################################################

    model.train()
    model, train_losses, valid_losses, train_acc, val_acc, lrs = train(config, model=model)
    checkpoint = torch.load(config.save_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    result_dict = {}
    result_dict['top1_test_acc'], result_dict['top5_test_acc']= evalauate(args=config, model=model)
    result_dict['train_losses'] = train_losses
    result_dict['valid_losses'] = valid_losses
    result_dict['train_acc'] = train_acc
    result_dict['val_acc'] = val_acc
    result_dict['lrs'] = lrs

    # if config.model_save_dir != None:
    #     suffix_list = ['model', 'seed', 'lr_rate', 'lr_scheduler']
    #     save_path = model_save(config, suffix_list)

    return result_dict



def train(args, model):
    if(args.from_pretrained): return model  # dependency with dataset? TODO: clarify what dataset the pretrained model is for
    train_loader = get_train_loader(args.batch_size)
    valid_loader = get_valid_loader(args.batch_size)
    Trainer = image_trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            training_args=args
    )

    model, train_losses, validation_losses, train_acc, valid_acc, lrs = Trainer.train()
    return model, train_losses, validation_losses, train_acc, valid_acc, lrs

def evalauate(args, model):
    top1_acc = None
    top5_acc = None
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    class_correct5 = list(0. for i in range(10))
    model.eval()
    test_loader = get_test_loader(args.batch_size)
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        output=model(images)
        _,pred=torch.max(output,1)
        _,pred5 = torch.topk(output, 5)
        pred5 = pred5.t()
        temp = pred5.eq(labels.view(1, -1).expand_as(pred5))
        correct5 = temp.float().sum(0)
        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
        for idx in range(args.batch_size):
            label = labels[idx]
            class_correct[label] += correct[idx].item()
            class_correct5[label] += correct5[idx].item()
            class_total[label] += 1
    correct = 0
    total = 0
    correct5 = 0
    for i in range(10):
        correct += class_correct[i]
        correct5 += class_correct5[i]
        total += class_total[i]
    top1_acc = correct / total
    top5_acc = correct5 / total
    return top1_acc, top5_acc