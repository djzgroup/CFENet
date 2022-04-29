import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import test_loader
import numpy as np
from PIL import Image
from tqdm import tqdm
from network import model
from network.seg_hrnet import HighResolutionNet
from network import deeplabv3
from network import model
from network import unet
from network import pspnet
batch_size =8
test_data = './dataset/test/'
img_size = 512
model_name = 'CLNet'
save_path = './'+'result/'+model_name + '/'
model_path = './train_model/' + model_name + 'building_extraction.pth'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def main():
    with torch.no_grad():
        # net = unet.unet_resnet101(n_classes=1, batch_size=batch_size, pretrained=True, fixed_feature=True).to(device)
        # net = deeplabv3.DeepLabV3().to(device)
        # net = HighResolutionNet(input_channels=3, output_channels=1).to(device)
        # net = pspnet.pspnet_resnet101(n_classes=1, batch_size =batch_size , pretrained=True, fixed_feature=True).to(device)
        net = model.CLNET().to(device)
        net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        test_load = test_loader.get_loader(test_data, batch_size, img_size, num_workers=4, mode='test', \
                                             augmentation_prob=0, shuffle=False, pin_memory=True)
        print("Strat Testing!")
        test(test_load, net, save_path)
def test(test_load, net, save_path):
    net.train(False)
    net.eval()
    for i, (inputs, mask, filename) in enumerate(tqdm(test_load)):
        X = inputs.to(device)
        Y = mask.to(device)
        output = net(X)
        # output = net(X)
        output = F.sigmoid(output)
        for i in range(output.shape[0]):
            probs_array = (torch.squeeze(output[i])).data.cpu().numpy()
            mask_array = (probs_array > 0.5)
            final_mask = mask_array.astype(np.float32)
            final_mask = final_mask * 255
            final_mask = final_mask.astype(np.uint8)
            final_savepath = save_path + '/' + filename[i] + '.png'
            im = Image.fromarray(final_mask)
            im.save(final_savepath)
   

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()