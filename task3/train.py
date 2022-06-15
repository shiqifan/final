import argparse
import os
import sys
from sklearn.metrics import accuracy_score , confusion_matrix
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.datasets as dataset


import torch
import torch.nn as nn

from models import Vit

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=32, help="size of image height")
parser.add_argument("--img_width", type=int, default=32, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
opt = parser.parse_args()
print(opt)

os.makedirs("saved_models/", exist_ok=True)
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

loss_fun = nn.CrossEntropyLoss()

Net = Vit(in_channels=3,patch_size=16,img_size=32,d_model=256,n_head=8,dim_feed_foward=2048,num_layers=8,n_classes=100)

if cuda:
   Net =  Net.cuda()
   
   
optimizer = torch.optim.Adam(Net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



train_data = dataset.CIFAR100 (root = "datasets/cifar100",
train = True,
transform = transforms_,
download = True)

# 测试数据集
test_data = dataset.CIFAR100 (root = "datasets/cifar100",
train = False,
transform = transforms_,
download = True)

# 
log_dir    = os.path.join('logs') 
writer     = SummaryWriter(log_dir)

# write data to tensorboard
def write_to_tensorboard(writer, tag, value, step):
    writer.add_scalar(tag, value, step)

    

if __name__ == '__main__':
    dataloader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    
    val_loader = DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
    
    acc = 0
    history_train_loss = []
    history_val_loss = []
    history_train_acc = []
    history_val_acc = []
    
    # 训练
    for epoch in range(opt.epoch, opt.n_epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0
        correct = 0
        total = 0
        Net.train()
        print()
        for i, batch in enumerate(dataloader):
            input_img = Variable(batch[0].type(Tensor))
            label     = Variable(batch[1].to(device))

            optimizer.zero_grad()
            output = Net(input_img)

            
            loss = loss_fun(output,label)
            # 更新梯度
            loss.backward()
            optimizer.step()

            # 查看准确率
            prediction = torch.argmax(torch.softmax(output,-1), -1)
            
            correct += (prediction == label).sum().float()
            total += len(label)            

            epoch_train_loss += loss.item()
            sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [acc: %f]"
                    % (
                        epoch + opt.epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        epoch_train_loss / (i+1),
                        correct/total,
                    )
                )
        history_train_loss.append(epoch_train_loss/len(dataloader))
        history_train_acc.append((correct/total).cpu().numpy())
        
        write_to_tensorboard(writer, 'train_loss', history_train_loss[-1], epoch)
        write_to_tensorboard(writer, 'train_acc', history_train_acc[-1], epoch)
        # 
        print()
        # 
        
        Net.eval()
        total = 0
        correct = 0
        for i, batch in enumerate(val_loader):
            with torch.no_grad():
                input_img = batch[0].type(Tensor)
                label     = batch[1].to(device)

                output = Net(input_img)
    
                loss = loss_fun(output,label)
                # 更新梯度
                # 查看准确率
                prediction = torch.argmax(torch.softmax(output,-1), -1)
                correct += (prediction ==label).sum().float()
                total += len(label)            

                epoch_val_loss += loss.item()
                sys.stdout.write(
                        "\rvalidation [Epoch %d/%d] [Batch %d/%d] [loss: %f] [acc: %f]"
                        % (
                            epoch + opt.epoch,
                            opt.n_epochs,
                            i,
                            len(val_loader),
                            epoch_val_loss / (i+1),
                            correct/total,
                        )
                    )
            
                # 保存在验证集中准确率最高的一代
        if(correct/total > acc):
            acc = correct/total
            torch.save(Net.state_dict(),'saved_models/model.pth')
        
        history_val_loss.append(epoch_val_loss / len(val_loader))
        history_val_acc.append((correct/total).cpu().numpy())
    
        write_to_tensorboard(writer, 'val_loss', history_val_loss[-1], epoch)
        write_to_tensorboard(writer, 'val_acc', history_val_acc[-1], epoch)
    
    
    with torch.no_grad():
        true_labels = []
        predict_labels = []
        
        for i, batch in enumerate(val_loader):
            input_img = batch[0].type(Tensor)
            label     = batch[1].to(device)

            output = Net(input_img)

            
            
            true_labels += label.cpu().numpy().reshape(-1).tolist()
            predict_labels += torch.argmax(torch.softmax(output,-1),-1).cpu().numpy().reshape(-1).tolist()
        
        print('Acc Score is :' , accuracy_score(true_labels , predict_labels))
        print('confusion matrix')
        print(confusion_matrix(true_labels , predict_labels))
        
        plt.subplot(121)
        plt.plot(history_train_loss,label = 'Train Loss')
        plt.plot(history_val_loss,label = 'Validation Loss')
        plt.legend()
        plt.subplot(122)
        plt.plot(history_train_acc,label = 'Train Acc')
        plt.plot(history_val_acc,label = 'Validation Acc')
        plt.legend()
        plt.tight_layout()
        plt.savefig('train_acc.png')