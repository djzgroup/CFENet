import torch

# SR : Segmentation Result
# GT : Ground Truth
# TP : True Positive
# FP : False Positive

def get_precision(SR,GT):
    """训练精度，标签必须转换成0-1，容差0.5"""
    # Precision=TP/(TP+FP)

    TP = ((SR==1) & (GT==1))
    FP = ((SR==1) & (GT==0))

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC


def get_sensitivity(SR, GT):
    """敏感度=召回率"""
    # Recall=sensitivity=TPR=TP/(TP+FN)

    TP = ((SR == 1) & (GT == 1))
    FN = ((SR == 0) & (GT == 1))

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE

def get_accuracy(SR,GT):
    """准确率 如果GT受双线性内插的影响可能有一部分内插后的值不为1"""
    # Accuracy = TP / (TP + FP + TN + FN)

    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_specificity(SR,GT):
    """真负率，可理解为错误的被判断为错误的"""
    # specificity=TN/(TN+FP)

    TN = ((SR==0)&(GT==0))
    FP = ((SR==1)&(GT==0))

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_F1(SR,GT):
    """F1系数"""
    # F1 = 2 * r * p / (r + p)
    SE = get_sensitivity(SR,GT)
    PC = get_precision(SR,GT)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT):
    """IOU，交并比"""
    # JS : Jaccard similarity
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS
def get_iou(SR, GT):

    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)

    iou = float(Inter) / (float(Union) + 1e-6)
    return iou



def get_fwiou(SR, GT):
    TP = torch.sum(((SR == 1) & (GT == 1)))
    FP = torch.sum(((SR == 1) & (GT == 0)))
    TN = torch.sum(((SR == 0) & (GT == 0)))
    FN = torch.sum(((SR == 0) & (GT == 1)))
    fwiou1 = float(((TP + FN)/(TP+FP+TN+FN + 1e-6))*(TP / (TP + FP + FN + 1e-6)))
    fwiou2 = float(((TN + FP)/(TP+FP+TN+FN + 1e-6))*(TN / (TN + FP + FN + 1e-6)))
    fwiou =fwiou1 + fwiou2
    return fwiou


def get_DC(SR,GT):
    # DC : Dice Coefficient

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC



