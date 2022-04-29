import torch

# SR : Segmentation Result
# GT : Ground Truth
# TP : True Positive
# FP : False Positive

def get_precision(SR,GT,threshold=0.5):
    """训练精度，标签必须转换成0-1，容差0.5"""
    # Precision=TP/(TP+FP)
    SR[SR >  threshold] = 1
    SR[SR <= threshold] = 0
    GT[GT > 0.5] = 1
    GT[GT <= 0.5] = 0

    TP = ((SR==1) & (GT==1))
    FP = ((SR==1) & (GT==0))

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC


def get_sensitivity(SR, GT, threshold=0.5):
    """敏感度=召回率"""
    # Recall=sensitivity=TPR=TP/(TP+FN)
    SR[SR > threshold] = 1
    SR[SR <= threshold] = 0
    GT[GT > 0.5] = 1
    GT[GT <= 0.5] = 0

    TP = ((SR == 1) & (GT == 1))
    FN = ((SR == 0) & (GT == 1))

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE

def get_accuracy(SR,GT,threshold=0.5):
    """准确率 如果GT受双线性内插的影响可能有一部分内插后的值不为1"""
    # Accuracy = TP / (TP + FP + TN + FN)
    SR[SR > threshold] = 1
    SR[SR <= threshold] = 0
    GT[GT > 0.5] = 1
    GT[GT <= 0.5] = 0

    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_specificity(SR,GT,threshold=0.5):
    """真负率，可理解为错误的被判断为错误的"""
    # specificity=TN/(TN+FP)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TN = ((SR==0)&(GT==0))
    FP = ((SR==1)&(GT==0))

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_F1(SR,GT,threshold=0.5):
    """F1系数"""
    # F1 = 2 * r * p / (r + p)
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    """IOU，交并比"""
    # JS : Jaccard similarity
    SR[SR > threshold] = 1
    SR[SR <= threshold] = 0
    GT[GT > 0.5] = 1
    GT[GT <= 0.5] = 0
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS
def get_iou(SR, GT, threshold=0.5):
    SR[SR > threshold] = 1
    SR[SR <= threshold] = 0
    GT[GT > 0.5] = 1
    GT[GT <= 0.5] = 0

    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)

    iou = float(Inter) / (float(Union) + 1e-6)
    return iou



def get_fwiou(SR, GT, threshold=0.5):
    SR[SR > threshold] = 1
    SR[SR <= threshold] = 0
    GT[GT > 0.5] = 1
    GT[GT <= 0.5] = 0
    TP = torch.sum(((SR == 1) & (GT == 1)))
    FP = torch.sum(((SR == 1) & (GT == 0)))
    TN = torch.sum(((SR == 0) & (GT == 0)))
    FN = torch.sum(((SR == 0) & (GT == 1)))
    fwiou1 = float(((TP + FN)/(TP+FP+TN+FN + 1e-6))*(TP / (TP + FP + FN + 1e-6)))
    fwiou2 = float(((TN + FP)/(TP+FP+TN+FN + 1e-6))*(TN / (TN + FP + FN + 1e-6)))
    fwiou =fwiou1 + fwiou2
    return fwiou


def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR[SR > threshold] = 1
    SR[SR <= threshold] = 0
    GT[GT > 0.5] = 1
    GT[GT <= 0.5] = 0

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC



