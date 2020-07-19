import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CyclicLR

from torchhk import RecordManager

from torchattacks import FGSM
from torchattacks import PGD

##################################################################
############################# Train ##############################
##################################################################

class BaseTrainer():
    def __init__(self, model, train_sets, test_sets):
        self.model = model
        self.record_base_keys = ['Epoch',
                                 'Clean(Tr)', 'FGSM(Tr)', 'PGD(Tr)', 'GN(Tr)',
                                 'Clean(Te)', 'FGSM(Te)', 'PGD(Te)', 'GN(Te)',
                                 'lr']
        self.record_keys = ["Loss", "Acc"]
        self.train_sets = train_sets
        self.test_sets = test_sets
        
    def train(self, train_loader, epochs=200,
              optimizer=None, scheduler=None,
              scheduler_type=None, save_path=None,
              save_type="Epoch", record_type="Epoch",
              device='cuda'):
        
        self.epochs = epochs
        self.device = torch.device(device)
        self.model = self.model.to(self.device).train()
        self.max_lter = len(train_loader)
        
        if save_path is not None :
            self._check_path(save_path)
            self._create_path(save_path)
       
        self._add_record_keys(self.record_keys)
        if record_type == "Iter" :
            self._add_record_keys(["Iter"])
        self._init_record()
        self._init_optim(optimizer, scheduler, scheduler_type)
        self._init_attacks()

        print("Train Information:")
        print("-Epochs:",self.epochs)
        print("-Optimizer:",self.optimizer)
        print("-Scheduler:",self.scheduler)
        print("-Save Path:",save_path)
        print("-Save Type: Per",save_type)
        print("-Record Type: Per",record_type)
        print("-Device:",self.device)
        
        for epoch in range(self.epochs):
            epoch_record = []
            self.epoch = epoch
            
            for i, (batch_images, batch_labels) in enumerate(train_loader):
                self.iter = i
                
                iter_record = self._do_iter(batch_images, batch_labels)
                self.rm.progress()

                is_last_batch = (i+1==self.max_lter)
                
                if self._check_type(self.scheduler_type, is_last_batch):
                    self.scheduler.step()

                if record_type == "Epoch":
                    epoch_record.append(iter_record)
                    if is_last_batch:
                        epoch_record = torch.tensor(epoch_record).float()
                        self._update_record([epoch+1,
                                            *[r.item() for r in epoch_record.mean(dim=0)]])
                        epoch_record = []
                else :
                    self._update_record([epoch+1, i+1, *iter_record])

                if save_type == "Epoch":
                    if is_last_batch:
                        self._save_model(save_path, epoch+1, None)
                else :
                    self._save_model(save_path, epoch+1, i+1)

                self.model = self.model.to(self.device).train()
                    
        self.rm.summary()
    
    def _do_iter(self, images, labels):
        X = images.to(self.device)
        Y = labels.to(self.device)

        pre = self.model(X)
        cost = nn.CrossEntropyLoss()(pre, Y)

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        _, pre = torch.max(pre.data, 1)
        total = pre.size(0)
        correct = (pre == Y).sum()
        cost = cost.item()
        
        return cost, 100*float(correct)/total
        
    def _add_record_keys(self, keys):
        for i, key in enumerate(keys) :
            self.record_base_keys.insert(1+i, key)
        
    def _init_record(self):
        self.rm = RecordManager(self.record_base_keys)
    
    def _update_record(self, records):
        train_images, train_labels = self.train_sets
        test_images, test_labels = self.test_sets
        
        fgsm_train = self.sample_fgsm(train_images, train_labels)
        pgd_train = self.sample_pgd(train_images, train_labels)
        gaussian_train = train_images + 0.1*torch.randn_like(train_images)
        gaussian_train = torch.clamp(gaussian_train, min=0, max=1)

        fgsm_test = self.sample_fgsm(test_images, test_labels)
        pgd_test = self.sample_pgd(test_images, test_labels)
        gaussian_test = test_images + 0.1*torch.randn_like(test_images)
        gaussian_test = torch.clamp(gaussian_test, min=0, max=1)
        
        self.rm.add([*records,
                     get_acc(self.model, [(train_images, train_labels)]),
                     get_acc(self.model, [(fgsm_train, train_labels)]),
                     get_acc(self.model, [(pgd_train, train_labels)]),
                     get_acc(self.model, [(gaussian_train, train_labels)]),
                     get_acc(self.model, [(test_images, test_labels)]),
                     get_acc(self.model, [(fgsm_test, test_labels)]),
                     get_acc(self.model, [(pgd_test, test_labels)]),
                     get_acc(self.model, [(gaussian_test, test_labels)]),
                     self.optimizer.param_groups[0]['lr']])
        
    def _init_optim(self, optimizer, scheduler, scheduler_type):
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        
        if optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1,
                                       momentum=0.9, weight_decay=5e-4)

        if (scheduler is None) or (scheduler=="Stepwise"):
            self.scheduler = MultiStepLR(self.optimizer,
                                         milestones=[60, 120, 160],
                                         gamma=0.2)
            self.scheduler_type = 'Epoch'

        if scheduler is 'Cyclic' :
            lr_steps = self.epochs * self.max_lter
            self.scheduler = CyclicLR(self.optimizer, base_lr=0,
                                      max_lr=0.2, step_size_up=lr_steps / 2, 
                                      step_size_down=lr_steps / 2)
            self.scheduler_type = 'Iter'
        
        if self.scheduler_type is None :
            raise ValueError("The type of scheduler must be specified as 'Epoch' or 'Iter'.")
            
    def _init_attacks(self):        
        self.sample_fgsm = FGSM(self.model, eps=8/255)
        self.sample_pgd = PGD(self.model, alpha=2/255, eps=8/255, iters=7)
        
    def _check_type(self, type, is_last_batch):
        if type=="Epoch" and is_last_batch:
            return True
        elif type=="Iter":
            return True
        return False
    
    def _check_path(self, path):
        if os.path.exists(path) :
            raise ValueError('[%s] is already exists.'%(path))
    
    def _create_path(self, path):
        os.makedirs(path)
            
    def _save_model(self, save_path, epoch, i):
        if save_path is not None:
            if i is None :
                torch.save(self.model.cpu().state_dict(),
                           save_path+"/"+str(epoch).zfill(3)+".pth")
            else :
                torch.save(self.model.cpu().state_dict(),
                           save_path+"/"+str(epoch).zfill(3)\
                           +"_"+str(i).zfill(3)+".pth")
        
    def save_all(self, save_path):
        self._check_path(save_path+".pth")
        self._check_path(save_path+".csv")
        torch.save(self.model.cpu().state_dict(), save_path+".pth")
        self.rm.to_csv(save_path+".csv")    

##################################################################
############################## Test ##############################
##################################################################


def get_acc(model, test_loader, device='cuda'):
    # Set Cuda or Cpu
    device = torch.device(device)
    model.to(device)
    
    # Set Model to Evaluation Mode
    model.eval()
    
    # Initialize
    correct = 0
    total = 0

    # For all Test Data
    for batch_images, batch_labels in test_loader:

        # Get Batches
        X = batch_images.to(device)
        Y = batch_labels.to(device)
        
        # Forward
        pre = model(X)

        # Calculate Accuracy
        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()

    return (100 * float(correct) / total)
