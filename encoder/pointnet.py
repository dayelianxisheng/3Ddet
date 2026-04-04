import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self,channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.register_buffer('iden', torch.eye(3).view(1, 9))

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x + self.iden.repeat(batchsize, 1)
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.register_buffer('iden', torch.eye(k).view(1, k * k))
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x + self.iden.repeat(batchsize, 1)
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self,global_feat = True,feature_transform = False,channel = 3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self,x):
        B,D,N = x.size()
        # Spatial transform is defined on XYZ only; extra channels are concatenated back after alignment.
        trans = self.stn(x[:, :3, :])
        x = x.transpose(1,2)
        if D>3:
            feature =x[:,:,3:] # 剥离出 3 维以后的特征
            x = x[:,:,:3]

        # 执行空间变换：[B, N, 3] 乘以 [B, 3, 3]
        # 物理意义：根据模型学习到的角度，旋转/平移整个点云
        x = torch.bmm(x,trans) # 形状依然是: [B, N, 3]
        if D>3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x =self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1,1024,1).repeat(1,1,N)
            return torch.cat([x,pointfeat],1),trans,trans_feat


if __name__ == '__main__':
    sim_data = torch.rand(2, 3, 1024)

    # 情况 1: 测试分类模式 (Global Feature)
    print("--- 测试分类模式 ---")
    net_cls = PointNetEncoder(global_feat=True, feature_transform=True)
    # 解包返回值
    global_feat, trans, trans_feat = net_cls(sim_data)
    print(f"全局特征形状: {global_feat.size()}")  # [2, 1024]
    print(f"输入变换矩阵: {trans.size()}")  # [2, 3, 3]

    # 情况 2: 测试分割模式 (Point-wise Feature)
    print("\n--- 测试分割模式 ---")
    net_seg = PointNetEncoder(global_feat=False, feature_transform=True)
    seg_feat, trans, trans_feat = net_seg(sim_data)
    print(f"分割特征形状: {seg_feat.size()}")  # [2, 1088, 1024]
