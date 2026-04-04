import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from encoder import PointNetEncoder
from loss.OrthogonalRegularizationLoss import feature_transform_regularizer

class get_model(nn.Module):
    def __init__(self,num_class):
        super(get_model, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=9)
        self.conv1 = torch.nn.Conv1d(1088,512,1)
        self.conv2 = torch.nn.Conv1d(512,256,1)
        self.conv3 = torch.nn.Conv1d(256,128,1)
        self.conv4 = torch.nn.Conv1d(128,self.k,1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self,x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x,trans,trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(1,2).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize,n_pts,self.k)
        return x,trans,trans_feat

class get_loss(nn.Module):
    def __init__(self,mat_diff_loss_scale=0.001):
        super(get_loss,self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self,pred,target,trans_feat,weight):
        loss=F.nll_loss(pred,target,weight=weight)
        mat_diff_loss = feature_transform_regularizer(trans_feat)
        total_loss = loss+mat_diff_loss*self.mat_diff_loss_scale
        return total_loss

