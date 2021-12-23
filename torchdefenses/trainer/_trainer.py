import os
from collections.abc import Iterable

import torch
from torch.optim import *
from torch.optim.lr_scheduler import *

from ._rm import RecordManager    
from .. import RobModel


r"""
Base class for all trainers.

"""

class Trainer():
    def __init__(self, name, model):
        assert isinstance(model, RobModel)
        self.name = name
        self.model = model
        self.device = next(model.parameters()).device
        
    def fit(self, train_loader, max_epoch=200, start_epoch=0,
            optimizer=None, scheduler=None, scheduler_type=None,
            save_type="Epoch", save_path=None, save_overwrite=False, 
            record_type="Epoch"):
        
        # Set Epoch and Iterations
        self.max_epoch = max_epoch
        self.train_loader = train_loader
        self.max_iter = len(train_loader)
            
        # Set Optimizer and Schduler
        self._init_optim(optimizer, scheduler, scheduler_type)
        
        # Check Save and Record Values
        self._check_valid_options(save_type)
        self._check_valid_options(record_type)
        
        # Check Save Path
        if save_path is not None:
            if save_path[-1] == "/":
                save_path = save_path[:-1]
            # Save Initial Model
            self._check_path(save_path, overwrite=save_overwrite)
            self._save_model(save_path, 0)
        else:
            raise ValueError("save_path should be given for save_type != None.")
            
        # Print Training Information
        if record_type is not None:
            self._init_record(record_type)
            print("["+self.name+"]")
            print("Training Information.")
            print("-Epochs:",self.max_epoch)
            print("-Optimizer:",self.optimizer)
            print("-Scheduler:",self.scheduler)
            print("-Save Path:",save_path)
            print("-Save Type:",str(save_type))
            print("-Record Type:",str(record_type))
            print("-Device:",self.device)
        
        # Training Start
        for epoch in range(self.max_epoch):
            self.epoch = epoch+1
            
            # If start_epoch is given, update schduler steps.
            if self.epoch < start_epoch:
                if self.scheduler_type == "Epoch":
                    self._update_scheduler()
                elif self.scheduler_type == "Iter":
                    for _ in range(max_iter):
                        self._update_scheduler()
                else:
                    pass
                continue
            
            for i, train_data in enumerate(train_loader):                
                self.iter = i+1
                
                # Set Train Mode
                self.model = self.model.to(self.device)
                self.model.train()
                
                # Do Iteration and get records.
                iter_record = self._do_iter(train_data)
                if not isinstance(iter_record, Iterable):
                    iter_record = [iter_record]
                
                # Check Last Batch
                is_last_batch = (self.iter==self.max_iter)
                self.model.eval()
                    
                # Update Records
                if record_type is not None:
                    self.rm.progress()
                    if record_type == "Epoch" and is_last_batch:
                        self._update_record([self.epoch, *iter_record])
                    elif record_type == "Iter":
                        self._update_record([self.epoch, self.iter, *iter_record])
                    
                    if save_path is not None:
                        self.rm.to_csv(save_path+".csv", verbose=False)
                else:
                    pass
                    
                # Save Model
                if save_type == "Epoch" and is_last_batch:
                    self._save_model(save_path, self.epoch)
                elif save_type == "Iter":
                    self._save_model(save_path, self.epoch, self.iter)
                else:
                    pass
                
                # Scheduler Step
                if self.scheduler_type=="Epoch" and is_last_batch:
                    self._update_scheduler()
                elif self.scheduler_type=="Iter":
                    self._update_scheduler()
                else:
                    pass
                
        # Print Summary
        if record_type is not None:
            self.rm.summary()
            self.rm.to_csv(save_path+".csv", verbose=False)
        
    def save_all(self, save_path, overwrite=False):
        self._check_path(save_path+".pth", overwrite=overwrite, file=True)
        self._check_path(save_path+".csv", overwrite=overwrite, file=True)
        print("Saving Model")
        torch.save(self.model.cpu().state_dict(), save_path+".pth")
        print("...Saved as pth to %s !"%(save_path+".pth"))
        print("Saving Records")
        self.rm.to_csv(save_path+".csv")
        self.model.to(self.device)
    
    ################################
    # CAN OVERRIDE BELOW FUNCTIONS #
    ################################
    
    # Do Iter
    def _do_iter(self, images, labels):
        raise NotImplementedError
        
    # Scheduler Update
    def _update_scheduler(self):
        self.scheduler.step()
    
    # Update Records
    def _update_record(self, records):
        self.rm.add([*records, self.optimizer.param_groups[0]['lr']])
        
    ####################################
    # DO NOT OVERRIDE BELOW FUNCTIONS #
    ###################################
            
    # Initialization RecordManager
    def _init_record(self, record_type):
        keys = ["Epoch"]
        if record_type == "Iter":
            keys = ["Epoch", "Iter"]
            
        for key in self.record_keys:
            keys.append(key)
            
        keys.append("lr")
        self.rm = RecordManager(keys)
        
    # Set Optimizer and Scheduler
    def _init_optim(self, optimizer, scheduler, scheduler_type):
        # Set Optimizer
        if not isinstance(optimizer, str):
            self.optimizer = optimizer     
        else:
            exec("self.optimizer = " + optimizer.split("(")[0] + "(self.model.parameters()," + optimizer.split("(")[1])

        # Set Scheduler
        if not isinstance(scheduler, str):
            self.scheduler = scheduler
            if self.scheduler is None:
                self.scheduler_type = None
            else:
                if scheduler_type is None:
                    raise ValueError("The type of scheduler must be specified as 'Epoch' or 'Iter'.")
                self.scheduler_type = scheduler_type
        else:
            if "Step(" in scheduler:
                # Step(milestones=[2, 4], gamma=0.1)
                exec("self.scheduler = MultiStepLR(self.optimizer, " + scheduler.split("(")[1])
                self.scheduler_type = 'Epoch'

            elif 'Cyclic(' in scheduler:
                # Cyclic(base_lr=0, max_lr=0.3)
                lr_steps = self.max_epoch * self.max_iter
                exec("self.scheduler = CyclicLR(self.optimizer, " + scheduler.split("(")[1].split(")")[0] + \
                     ", step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)")
                self.scheduler_type = 'Iter'

            elif 'Cosine' == scheduler:
                # Cosine
                self.scheduler = CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=0)
                self.scheduler_type = 'Epoch'
                
            else:
                exec("self.scheduler = " + scheduler.split("(")[0] + "(self.optimizer, " + scheduler.split("(")[1])
                self.scheduler_type = scheduler_type
            
    def _save_model(self, save_path, epoch, i=0):
        torch.save(self.model.cpu().state_dict(),
                   save_path+"/"+str(epoch).zfill(len(str(self.max_epoch)))\
                   +"_"+str(i).zfill(len(str(self.max_iter)))+".pth")
        
    # Check and Create Path
    def _check_path(self, path, overwrite=False, file=False):
        if os.path.exists(path):
            if overwrite:
                print("Warning: Save files will be overwritten!")
            else:
                raise ValueError('[%s] is already exists.'%(path))
        else:
            if not file:
                os.makedirs(path)
                
    # Check Valid Options
    def _check_valid_options(self, key):
        if key in ["Epoch", "Iter", None]:
            pass
        else:
            raise ValueError(key, " is not valid. [Hint:'Epoch', 'Iter', None]")
