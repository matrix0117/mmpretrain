import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpretrain.registry import MODELS
from mmpretrain.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmpretrain.models.losses.label_smooth_loss import LabelSmoothLoss



@MODELS.register_module()
class TransFGLoss(nn.Module):

    def __init__(self, smoothing_value=0):
        super(TransFGLoss, self).__init__()
        self.smoothing_value=smoothing_value

    def forward(self,
                pred,
                target,
                **kwargs):
        if self.smoothing_value == 0:
            loss_fct = CrossEntropyLoss()
        else:
            loss_fct = LabelSmoothLoss(label_smooth_val=self.smoothing_value)
        part_loss=loss_fct(pred[0],target.view(-1),avg_factor=pred[0].size(0))
        contrast_loss = con_loss(pred[1],target.view(-1))
        return part_loss+contrast_loss

def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss