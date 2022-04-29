import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import torch.nn as nn
import data_loader
import numpy as np
from PIL import Image
from tqdm import tqdm
from metric import *

batch_size =8
data_root = ''
test_data = ''
img_size = 512
model_name = ''
save_path = './'+'result/'+model_name + '/'
txt_name = './'+ model_name + '_test_result.txt'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def main():
    #net = unet_resnet50(1, batch_size, pretrained=False, fixed_feature=True).cuda()
    test_load = data_loader.get_loader(data_root,test_data, batch_size, img_size, num_workers=4, mode='test', \
                                             augmentation_prob=0, shuffle=False, pin_memory=True)
    print("Strat Testing!")
    test(test_load, save_path)
def test(test_load, save_path):
    acc = 0.  # Accuracy
    SE = 0.  # Sensitivity (Recall)
    SP = 0.  # Specificity
    PC = 0.  # Precision
    F1 = 0.  # F1 Score
    JS = 0.  # Jaccard Similarity
    DC = 0.  # Dice Coefficient
    FWIOU = 0.
    length = 0
    for i, (inputs, mask, filename) in enumerate(tqdm(test_load)):
        X = inputs.to(device)
        Y = mask.to(device)
        output = X
        # output = net(X)
        acc += get_accuracy(output, Y)
        SE += get_sensitivity(output, Y)
        SP += get_specificity(output, Y)
        PC += get_precision(output, Y)
        F1 += get_F1(output, Y)
        JS += get_JS(output, Y)
        DC += get_DC(output, Y)
        FWIOU += get_fwiou(output, Y)
        length += 1
        for i in range(output.shape[0]):
            probs_array = (torch.squeeze(output[i])).data.cpu().numpy()
            mask_array = (probs_array > 0.5)
            final_mask = mask_array.astype(np.float32)
            final_mask = final_mask * 255
            final_mask = final_mask.astype(np.uint8)
            final_savepath = save_path + '/' + filename[i] + '.png'
            im = Image.fromarray(final_mask)
            im.save(final_savepath)
    acc = acc / length
    SE = SE / length
    SP = SP / length
    PC = PC / length
    F1 = F1 / length
    JS = JS / length
    DC = DC / length
    FWIOU = FWIOU / length
    print('[Test] acc: %.4f, PC: %.4f, SP: %.4f, SE: %.4f, F1: %.4f, IOU: %.4f, DC: %.4f, FWIOU: %.4f' % (acc, PC, SP, SE, F1, JS, DC, FWIOU))
    string_print = 'acc: %.4f, PC: %.4f, SP: %.4f, SE: %.4f, F1: %.4f, IOU: %.4f, DC: %.4f, FWIOU: %.4f' % (acc, PC, SP, SE, F1, JS, DC, FWIOU)
    with open(txt_name, 'w') as f:
        f.write(string_print)
        f.write('\n')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()