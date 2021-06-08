import os
from collections import OrderedDict
import argparse
import torch
import torchvision
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
#from data_loader.ImageNet_datasets import ImageNetData
import resnet_cbam as resnet50
import resnet_cbam
from trainer import Trainer
from utils.logger import Logger
from PIL import Image
from torchnet.meter import ClassErrorMeter
#from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
def load_state_dict(model_dir, is_multi_gpu):
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def main(args):

    if 0 == len(args.resume):
        logger = Logger(r'D:\\work\\code\\focal_loss\\logs'+args.model+'.log')
    else:
        logger = Logger(r'D:\\work\\code\\focal_loss\\logs'+args.model+'.log', True)

    logger.append(vars(args))

    if args.display:
        writer = SummaryWriter()
    else:
        writer = None

    gpus = args.gpu.split(',')
    #直接读
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_datasets =torchvision.datasets.ImageFolder(root=r'D:\work\dataset\train',transform=train_transform)
    train_dataloaders =DataLoader(train_datasets,batch_size=16, shuffle=True,num_workers=8)#Batch Size定义：一次训练所选取的样本数。 Batch Size的大小影响模型的优化程度和速度。
    val_datasets =torchvision.datasets.ImageFolder(root=r'D:\work\dataset\val',transform=valid_transform)
    val_dataloaders =DataLoader(val_datasets,batch_size=16, shuffle=False,num_workers=8)
    #直接读

    if args.debug:
        x, y =next(iter(train_dataloaders))
        logger.append([x, y])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True

    if  'resnet34' == args.model.split('_')[0]:
        my_model = models.resnet50(pretrained=False)
    elif 'resnet18-cbam' == args.model.split('_')[0]:
        my_model = resnet_cbam.resnet18_cbam(pretrained=False)
    elif 'resnet50' == args.model.split('_')[0]:
        my_model = resnet_cbam.resnet50(pretrained=False)
    elif 'resnet101' == args.model.split('_')[0]:
        my_model = models.resnet101(pretrained=False)
    else:
        raise ModuleNotFoundError

    #my_model.apply(fc_init)
    if is_use_cuda and 1 == len(gpus):
        my_model = my_model.cuda()
    elif is_use_cuda and 1 < len(gpus):
        my_model = nn.DataParallel(my_model.cuda())

    loss_fn = [nn.CrossEntropyLoss()]
    #loss_fn = FocalLoss()
    optimizer = optim.SGD(my_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) 
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)           #

    metric = [ClassErrorMeter([1,5], True)]
    start_epoch = 0
    num_epochs  = 300

    my_trainer = Trainer(my_model, args.model, loss_fn, optimizer, lr_schedule, 500, is_use_cuda, train_dataloaders, \
                        val_dataloaders, metric, start_epoch, num_epochs, args.debug, logger, writer)
    my_trainer.fit()
    logger.append('Optimize Done!')
  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='trainer debug flag')
    parser.add_argument('-g', '--gpu', default='1', type=str,
                        help='GPU ID Select')                    
    parser.add_argument('-m', '--model', default='resnet50',
                         type=str, help='model type')
    parser.add_argument('--batch_size', default=8,
                         type=int, help='model train batch size')
    parser.add_argument('--display', action='store_true', dest='display',
                        help='Use TensorboardX to Display')
    args = parser.parse_args()

    main(args)
