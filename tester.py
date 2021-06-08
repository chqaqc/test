import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import os

class Tester():
    def __init__(self, model, model_type, loss_fn, optimizer, lr_schedule, log_batchs, is_use_cuda, test_data_loader, \
                metric=None, is_debug=False, logger=None, writer=None):
        self.model = model
        self.model_type = model_type
        self.loss_fn  = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.log_batchs = log_batchs
        self.is_use_cuda = is_use_cuda
        #self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.metric = metric
        #self.start_epoch = start_epoch
        #self.num_epochs = num_epochs
        self.is_debug = is_debug

        #self.cur_epoch = start_epoch
        self.best_acc = 0.
        self.best_loss = sys.float_info.max
        self.logger = logger
        self.writer = writer

    def fit(self):
        self._valid()

    def _dump_infos(self):
        self.logger.append('---------------------Current Parameters---------------------')
        self.logger.append('is use GPU: ' + ('True' if self.is_use_cuda else 'False'))
        self.logger.append('lr: %f' % (self.lr_schedule.get_lr()[0]))
        self.logger.append('model_type: %s' % (self.model_type))
        self.logger.append('current epoch: %d' % (self.cur_epoch))
        self.logger.append('best accuracy: %f' % (self.best_acc))
        self.logger.append('best loss: %f' % (self.best_loss))
        self.logger.append('------------------------------------------------------------')

    def _valid(self):
        self.model.eval()
        self.model.load_state_dict(torch.load('/mnt/omr/chenhq/CBAM_PyTorch/CBAM2/checkpoint/resnet50pretraind/Models_epoch_55.ckpt',map_location="cpu")['state_dict'])
        losses = []
        acc_rate = 0.
        if self.metric is not None:
            self.metric[0].reset()

        with torch.no_grad():              # Notice
            for i, (inputs, labels) in enumerate(self.test_data_loader):
                if self.is_use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    labels = labels.squeeze()
                else:
                    labels = labels.squeeze()

                outputs = self.model(inputs)            # Notice 
                loss = self.loss_fn[0](outputs, labels)

                if self.metric is not None:
                    prob     = F.softmax(outputs, dim=1).data.cpu()
                    self.metric[0].add(prob, labels.data.cpu())
                losses.append(loss.item())
            
        local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        #self.logger.append(losses)
        batch_mean_loss = np.mean(losses)
        print_str = '[%s]\tValidation: \t Class Loss: %.4f\t'     \
                    % (local_time_str, batch_mean_loss)
        if self.metric is not None:
            top1_acc_score = self.metric[0].value()[0]
            top5_acc_score = self.metric[0].value()[1]
            print_str += '@Top-1 Score: %.4f\t' % (top1_acc_score)
            print_str += '@Top-5 Score: %.4f\t' % (top5_acc_score)
        self.logger.append(print_str)
        if top1_acc_score >= self.best_acc:
            self.best_acc = top1_acc_score
            self.best_loss = batch_mean_loss