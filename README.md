# CFENet

## A Context Feature Enhancement Network for Building Extraction from High-Resolution Remote Sensing Imagery

The complexity and diversity of buildings make it challenging to extract low-level and high-level features with strong feature representation using deep neural networks in building extraction tasks. Meanwhile, deep neural network-based methods have many network parameters, which take up a lot of memory and time in training and testing. We propose a novel fully convolutional neural network called Context Feature Enhancement Network (CFENet) to address these issues. CFENet comprises three modules: the spatial fusion module, the focus enhancement module, and the feature decoder module. Firstly, the spatial fusion module aggregates the spatial information of low-level features to obtain buildings’ outline and edge information. Secondly, the focus enhancement module fully aggregates the semantic information of high-level features to filter the information of building-related attribute categories. Finally, the feature decoder module decodes the output of the above two modules to segment the buildings more accurately. In a series of experiments on the WHU Building and the Massachusetts Building Dataset, the efficiency and accuracy of our CFENet balance on all five evaluation metrics: PA, Recall, F1, IoU, and FWIoU. Moreover, our model is smaller, which means it occupies less memory and runs faster. This indicates that CFENet can effectively enhance and fuse buildings’ low-level and high-level features, improving building extraction accuracy.

**Our main contributions include the following:**

- An end-to-end context feature enhancement network, namely CFENet, is proposed to address the challenges of complexity and diversity of buildings encountered in building extraction from remote sensing images.
- CFENet achieves more accurate building extraction results on the WHU Building Dataset and the Massachusetts Building Dataset by explicitly establishing rich contextual relationships on low-level and high-level features.
- CFENet balances efficiency and accuracy by employing dilated convolution in the spatial fusion module and asymmetric convolution in the focus enhancement module.

### Requirements:

An Pytorch implementation of our work.

- pytorch 1.7.0
- CUDA 11.0
- python 3.7

### Overview code directory:

${ROOT}/ \
 ├── dataset/ \
 ├── network/ :contains model definition. \
 ├── pretrained_model/ :includes training model. \
 ├── train_model/ :includes training model. \
 ├── utils/ :contains some utility functions. \
 ├── train.py/ : training scripts for building extraction. \
 ├── metric.py \
 ├── README.md

### Implementation details:

Our proposed method is implemented based on pytorch 1.7.0 and cuda 11.0. An Adam optimizer is applied to train our network with a learning rate initialized to 0.0001, and then decayed to 0.1 of the current learning rate every 50 epochs. Our network is trained on the GPU for a number of 200 epochs.

## Our CFENet model：

**You can download the trained model to verify our results.**

- Our train_model (CFENet)\
  [[CSNetbuilding_extraction.pth]](https://drive.google.com/file/d/1lBFwepbbZjTcmf4WCiHPnbNpbBS-wxEf/view?usp=sharing)
- The pretrained_model\
  [[resnet101-5be5422a.pth]](https://drive.google.com/file/d/1W-bKdYJCyunaKDVU-zucyJ20vKBryX4J/view?usp=sharing)

### Dataset:

- WHU Building Dataset: download from this link\
  [[gpcv.whu.edu.cn/data/building_dataset.html]](http://gpcv.whu.edu.cn/data/building_dataset.html)
  
- Massachusetts Building Dataset: download from this link\
  [[Road and Building Detection Datasets (toronto.edu)]](https://www.cs.toronto.edu/~vmnih/data/)
  

### References:

- [1] Yu F, Koltun V. Multi-scale context aggregation by dilated convolutions[J]. arXiv preprint arXiv:1511.07122, 2015.
  
- [2] Denton E L, Zaremba W, Bruna J, et al. Exploiting linear structure within convolutional networks for efficient evaluation[J]. Advances in neural information processing systems, 2014, 27.
  
- [3] Ji S, Wei S, Lu M. Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set[J]. IEEE Transactions on Geoscience and Remote Sensing, 2018, 57(1): 574-586.
  
- [4] Mnih V. Machine learning for aerial image labeling[M]. University of Toronto (Canada), 2013.
