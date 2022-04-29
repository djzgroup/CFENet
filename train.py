import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6,7'
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils.utils import clip_gradient, adjust_lr
import utils.visualization as visual
from utils import data_loader
from torch.optim import lr_scheduler
from utils.evaluation import *
from utils.loss import *
#from utils import tools
from network import model
batch_size = 8
n_epoch = 100
model_name = 'CLNet'
train_data = './dataset/train/'
val_data = './dataset/valid/'
img_size = 512
model_path = './train_model'
op_lr = 1e-4
op_decay_rate = 0.1
op_decay_epoch = 50
if not os.path.exists(model_path):
    os.makedirs(model_path)

def main():
    net = model.CLNet().to(device)
    train_loader = data_loader.get_loader(train_data , batch_size , img_size,num_workers=4, mode='train',augmentation_prob=0.4,shuffle=True, pin_memory=True)
    val_loader = data_loader.get_loader(val_data , batch_size , img_size,num_workers=4, mode='val',augmentation_prob=0,shuffle=True, pin_memory=True)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr = op_lr)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    print("Strat Training!")
    train(train_loader, val_loader,net, criterion, optimizer, n_epoch, 0)


def train(train_loader, val_loader, net, criterion, optimizer, num_epoches, iters):
    vis = visual.Visualization()
    vis.create_summary(model_name)
    best_iou = 0
    for epoch in range(1, num_epoches + 1):
        cur_lr = adjust_lr(optimizer, op_lr, epoch, op_decay_rate, op_decay_epoch)
        epoch_loss = 0
        net.train(True)
        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        FWIOU = 0.

        length = 0
        st = time.time()
        for i,(inputs, mask) in enumerate(tqdm(train_loader)):
            X = inputs.to(device)
            Y = mask.to(device)
            optimizer.zero_grad()
            output = net(X)
            SR_probs = SR_eva = F.sigmoid(output)
            SR_flat = SR_probs.view(SR_probs.size(0), -1)
            GT_flat = Y.view(Y.size(0), -1)
            # loss = losses.mean()
            loss = criterion(SR_flat, GT_flat)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            epoch_loss += loss.item()
            st = time.time()
            acc += get_accuracy(SR_eva, Y)
            SE += get_sensitivity(SR_eva, Y)
            SP += get_specificity(SR_eva, Y)
            PC += get_precision(SR_eva, Y)
            F1 += get_F1(SR_eva, Y)
            JS += get_JS(SR_eva, Y)
            DC += get_DC(SR_eva, Y)
            FWIOU += get_fwiou(SR_eva, Y)

            length += 1
        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        train_loss = epoch_loss/length
        FWIOU = FWIOU / length


        vis.add_scalar(epoch, JS, 'IOU')
        vis.add_scalar(epoch, FWIOU, 'FWIOU')
        vis.add_scalar(epoch, acc, 'acc')
        vis.add_scalar(epoch, SE, 'SE')
        vis.add_scalar(epoch, SP, 'SP')
        vis.add_scalar(epoch, PC, 'PC')
        vis.add_scalar(epoch, F1, 'F1')
        vis.add_scalar(epoch, train_loss, 'train_loss')


        print(
            'Epoch [%d/%d], Loss: %.4f,learing_rate: %.4f, \n[Training] acc: %.4f, PC: %.4f, SP: %.4f, SE: %.4f, F1: %.4f, IOU: %.4f, DC: %.4f, FWIOU: %.4f' % (
                epoch, num_epoches, \
                train_loss, cur_lr,\
                acc, PC, SP, SE, F1, JS, DC, FWIOU))
        print("Strat validing!")
        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        FWIOU = 0.
        valid_loss = 0.

        length = 0
        net.train(False)
        net.eval()
        for i, (inputs, mask) in enumerate(tqdm(val_loader)):
            with torch.no_grad():

                X = inputs.to(device)
                Y = mask.to(device)
                optimizer.zero_grad()
                output = net(X)
                SR_probs = SR_eva = F.sigmoid(output)
                SR_flat = SR_probs.view(SR_probs.size(0), -1)
                GT_flat = Y.view(Y.size(0), -1)
            # loss = losses.mean()
                Valid_loss = criterion(SR_flat, GT_flat)
                valid_loss +=Valid_loss.item()
                acc += get_accuracy(SR_eva, Y)
                SE += get_sensitivity(SR_eva, Y)
                SP += get_specificity(SR_eva, Y)
                PC += get_precision(SR_eva, Y)
                F1 += get_F1(SR_eva, Y)
                JS += get_JS(SR_eva, Y)
                DC += get_DC(SR_eva, Y)
                FWIOU += get_fwiou(SR_eva, Y)

                length += 1
        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        FWIOU = FWIOU / length
        valid_loss = valid_loss/length

        unet_score = JS + DC
        print('[Validation] valid_loss: %.4f, Acc: %.4f, PC: %.4f, SP: %.4f, SE: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, FWIOU: %.4f' % (
            valid_loss, acc, PC, SP, SE, F1, JS, DC, FWIOU))
        new_iou = JS
        if new_iou >= best_iou:
            best_iou = new_iou
            best_epoch = epoch
            best_net = net.state_dict()
            print('Best %s Model Iou :%.4f; FWIOU : %.4f; Best epoch : %d' % (model_name, JS, FWIOU, best_epoch))
            torch.save(best_net, model_path+'/'+ model_name +'building_extraction.pth')
    vis.close_summary()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()