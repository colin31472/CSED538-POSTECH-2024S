import os
import torch
from torch import nn
import numpy as np


#TODO: argument 받아서 해당 argument에 맞게 훈련시킬 수 있게하기


class image_trainer():
    def __init__(self, model, train_loader, valid_loader, training_args):
        self.model = model
        self.train_loader = train_loader
        self.training_args = training_args
        self.val_loader = valid_loader

    def train(self):
        epochs = self.training_args.num_epochs
        batch_size = self.training_args.batch_size
        if(self.training_args.optimizer == "SGD"):
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.training_args.initial_lr) 
        elif(self.training_args.optimizer == "AdamW"):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_args.initial_lr)
        scheduler = self.training_args.lr_scheduler(optimizer, factors=self.training_args.new_technique_args['factors'], model=self.model, batch_size=self.training_args.new_technique_args['batch_size'], initial_lr=self.training_args.new_technique_args['initial_lr'])
        loader = self.train_loader
        criterion = nn.CrossEntropyLoss()

        loss_keeper={'train':[],'valid':[]}
        acc_keeper={'train':[],'valid':[]}
        lr_keeper = {'lr': []}
        
        train_class_correct = list(0. for i in range(10))
        valid_class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        val_class_total = list(0. for i in range(10))
        pre = 100
        for epoch in range(epochs):
            train_loss=0.0
            valid_loss=0.0
            train_correct=0.0
            valid_correct=0.0
            trained = 0
            print(f"Learning Rate :", scheduler.get_last_lr()[0].item(), "\n\n")
            for images, labels in loader:
                if(torch.cuda.is_available()):
                    images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, pred = torch.max(output, 1)
                train_correct=np.squeeze(pred.eq(labels.data.view_as(pred)))
                for idx in range(batch_size):
                    label = labels[idx]
                    train_class_correct[label] += train_correct[idx].item()
                    class_total[label] += 1
            self.model.eval()  ## with로 바꾸기?
            with torch.no_grad():
                for images, labels in self.val_loader:
                    if torch.cuda.is_available():
                        images,labels=images.cuda(),labels.cuda()
                    output=self.model(images)
                    loss=criterion(output,labels)
                    valid_loss+=loss.item()
                    _, pred = torch.max(output, 1)
                    valid_correct=np.squeeze(pred.eq(labels.data.view_as(pred)))
                    for idx in range(batch_size):
                        label = labels[idx]
                        valid_class_correct[label] += valid_correct[idx].item()
                        val_class_total[label] += 1
            train_loss = train_loss/len(loader)
            valid_loss = valid_loss/len(self.val_loader)

            train_acc=float(100. * np.sum(train_class_correct) / np.sum(class_total))
            valid_acc=float(100. * np.sum(valid_class_correct) / np.sum(val_class_total))

            # saving loss values
            loss_keeper['train'].append(train_loss)
            loss_keeper['valid'].append(valid_loss)

            # saving acc values
            acc_keeper['train'].append(train_acc)
            acc_keeper['valid'].append(valid_acc)

            # save checkpoint
            if pre > valid_loss:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                torch.save(checkpoint, f"{self.training_args.save_name}")
                print("checkpoint saved")
                pre = valid_loss

            print(f"Epoch : {epoch+1}")
            print(f"Training Loss : {train_loss}\tValidation Loss : {valid_loss}")
            print(f"Training Accuracy : {train_acc}\tValidation Accuracy : {valid_acc}")
            print(f"Learning Rate :", scheduler.get_last_lr()[0].item(), "\n\n")
            lr_keeper['lr'].append(scheduler.get_last_lr()[0].item())
            scheduler.step()


        return self.model, loss_keeper['train'], loss_keeper['valid'], acc_keeper['train'], acc_keeper['valid'], lr_keeper['lr']

