import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
import wandb

wandb.init(
    project="ISTD",
    config = {
    "epochs": 20,
    }
)

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k = 64

class ScaleFusionModule(nn.Module):
    def __init__(self, out_channels, out_size):
        super(ScaleFusionModule, self).__init__()

        # 多尺度特征融合子模块
        self.conv_low = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_high = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool_mid = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_high = nn.MaxPool2d(2)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.BN3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.attention_BN1 = nn.BatchNorm2d(3 * out_channels)
        self.attention_BN2 = nn.BatchNorm2d(3 * out_channels)
        self.attention_BN3 = nn.BatchNorm2d(3 * out_channels)

        # 尺度注意力生成子模块
        self.attention_conv1 = nn.Conv2d(out_channels * 3, out_channels * 3, kernel_size=3, padding=1)
        self.attention_conv2 = nn.Conv2d(out_channels * 3, out_channels * 3, kernel_size=3, padding=1)
        self.attention_conv3 = nn.Conv2d(out_channels * 3, out_channels * 3, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)

        self.out_size = out_size  # 输出标准化尺寸

    def forward(self, low, mid, high):
        low = self.conv_low(low)
        mid = self.conv_mid(mid)
        high = self.conv_high(high)
        low = self.BN1(low)
        mid = self.BN2(mid)
        high = self.BN3(high)
        low = self.relu(low)
        mid = self.relu(mid)
        high = self.relu(high)

        mid = nn.functional.interpolate(mid, size=low.shape[2:], mode='bilinear', align_corners=False)
        high = nn.functional.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)

        # 合并特征
        merged = torch.cat((low, mid, high), dim=1)

        # 尺度注意力生成子模块
        attention_weights = self.relu(self.attention_BN1(self.attention_conv1(merged)))
        attention_weights = self.relu(self.attention_BN2(self.attention_conv2(attention_weights)))
        attention_weights = self.softmax(self.attention_conv3(attention_weights))  # 生成尺度注意力权重
        attention_weights = nn.functional.adaptive_avg_pool2d(attention_weights, (1, 1))  # 输出形状 (B, channels, 1, 1)
        attention_low, attention_mid, attention_high = torch.split(attention_weights, low.shape[1], dim=1)

        # 权重作用到不同尺度特征
        low = attention_low * low
        mid = attention_mid * mid
        high = attention_high * high

        # 最终输出
        output = low + mid + high
        return output

# ResNet Block
class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1)

        self.upsample = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(2 * in_channels)
        self.bn2 = nn.BatchNorm2d(2 * in_channels)

    def forward(self, x):
        x_1 = self.relu(self.conv1(x))
        x_sample = self.upsample(x)
        x_2 = self.relu(self.bn1(self.conv2(x_1) + x_sample))

        x_3 = self.relu(self.conv3(x_2))
        x_4 = self.conv4(x_3)

        return self.relu(self.bn2(x_4 + x_2))

# CFM模块
class ChannelFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelFusionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(3 * in_channels, in_channels, kernel_size=3, padding=1)

        self.attention_conv1 = nn.Conv2d(3 * in_channels, 3 * in_channels, kernel_size=3, padding=1)
        self.attention_conv2 = nn.Conv2d(3 * in_channels, 3 * in_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.BN1 = nn.BatchNorm2d(3 * in_channels)
        self.BN2 = nn.BatchNorm2d(2 * in_channels)
        self.BN3 = nn.BatchNorm2d(2 * in_channels)
        self.BN4 = nn.BatchNorm2d(2 * in_channels)
        self.BN5 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x_extend = self.relu(self.BN1(self.conv(x)))
        f_1, f_2, f_3 = torch.split(x_extend, x.shape[1], dim=1)

        vk_1 = self.relu(self.BN2(self.conv1(f_1)))
        v_1, k_1 = torch.split(vk_1, x.shape[1], dim=1)
        vk_2 = torch.cat((f_2, k_1), dim=1)
        vk_2 = self.relu(self.BN3(self.conv2(vk_2)))
        v_2, k_2 = torch.split(vk_2, x.shape[1], dim=1)
        vk_3 = torch.cat((f_3, k_2), dim=1)
        vk_3 = self.relu(self.BN4(self.conv3(vk_3)))
        v_3, k_3 = torch.split(vk_3, x.shape[1], dim=1)

        k = torch.cat((k_1, k_2, k_3), dim=1)
        v = torch.cat((v_1, v_2, v_3), dim=1)

        attention = self.relu(self.attention_conv1(k))
        attention = self.softmax(self.attention_conv2(attention))
        attention = nn.functional.adaptive_avg_pool2d(attention, (1, 1))

        c = attention * v
        c = self.relu(self.BN5(self.final_conv(c)))
        return c

# BPR-Net模型
class BPRNet(nn.Module):
    def __init__(self):
        super(BPRNet, self).__init__()

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        #Resnet Block
        self.encoder2_1 = ResNet(16)
        self.encoder2_2 = ResNet(16)
        self.encoder2_3 = ResNet(16)
        self.encoder3_1 = ResNet(32)
        self.encoder3_2 = ResNet(32)
        self.encoder3_3 = ResNet(32)
        self.encoder4_1 = ResNet(64)
        self.encoder4_2 = ResNet(64)
        self.encoder4_3 = ResNet(64)
        self.encoder5_1 = ResNet(128)
        self.encoder5_2 = ResNet(128)
        self.encoder5_3 = ResNet(128)

        # 编码器第一个卷积层
        self.encoder_conv1 = nn.Conv2d(1, 16, kernel_size=7, padding=3)

        # 将所有特征都变为k通道
        self.low_conv1_1 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.low_conv1_2 = nn.Conv2d(32, k, kernel_size=5, padding=2)
        self.low_conv1_3 = nn.Conv2d(k, k, kernel_size=5, padding=2)
        self.low_conv1_4 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.low_conv1_5 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.mid_conv1_1 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.mid_conv1_2 = nn.Conv2d(32, k, kernel_size=5, padding=2)
        self.mid_conv1_3 = nn.Conv2d(k, k, kernel_size=5, padding=2)
        self.mid_conv1_4 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.mid_conv1_5 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.high_conv1_1 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.high_conv1_2 = nn.Conv2d(32, k, kernel_size=5, padding=2)
        self.high_conv1_3 = nn.Conv2d(k, k, kernel_size=5, padding=2)
        self.high_conv1_4 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.high_conv1_5 = nn.Conv2d(k, k, kernel_size=3, padding=1)

        self.low_conv2_1 = nn.Conv2d(32, k, kernel_size=3, padding=1)
        self.low_conv2_2 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.mid_conv2_1 = nn.Conv2d(32, k, kernel_size=3, padding=1)
        self.mid_conv2_2 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.high_conv2_1 = nn.Conv2d(32, k, kernel_size=3, padding=1)
        self.high_conv2_2 = nn.Conv2d(k, k, kernel_size=3, padding=1)

        self.low_conv3_1 = nn.Conv2d(64, k, kernel_size=3, padding=1)
        self.mid_conv3_1 = nn.Conv2d(64, k, kernel_size=3, padding=1)
        self.high_conv3_1 = nn.Conv2d(64, k, kernel_size=3, padding=1)

        self.low_conv4_2 = nn.Conv2d(128, k, kernel_size=3, padding=1)
        self.mid_conv4_2 = nn.Conv2d(128, k, kernel_size=3, padding=1)
        self.high_conv4_2 = nn.Conv2d(128, k, kernel_size=3, padding=1)

        self.low_conv5_1 = nn.Conv2d(256, k, kernel_size=3, padding=1)
        self.mid_conv5_1 = nn.Conv2d(256, k, kernel_size=3, padding=1)
        self.high_conv5_1 = nn.Conv2d(256, k, kernel_size=3, padding=1)

        # SFM模块
        self.sfm_1 = ScaleFusionModule(k, 64)
        self.sfm_2 = ScaleFusionModule(k, 32)
        self.sfm_3 = ScaleFusionModule(k, 16)
        self.sfm_4 = ScaleFusionModule(k, 8)
        self.sfm_5 = ScaleFusionModule(k, 4)

        # CFM模块
        self.CFM_1 = ChannelFusionModule(k)
        self.CFM_2 = ChannelFusionModule(k)
        self.CFM_3 = ChannelFusionModule(k)
        self.CFM_4 = ChannelFusionModule(k)
        self.CFM_5 = ChannelFusionModule(k)

        self.final_conv = nn.Conv2d(k, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.bnlow1_1 = nn.BatchNorm2d(32)
        self.bnlow1_2 = nn.BatchNorm2d(64)
        self.bnlow1_3 = nn.BatchNorm2d(64)
        self.bnlow1_4 = nn.BatchNorm2d(64)
        self.bnlow1_5 = nn.BatchNorm2d(64)

        self.bnmid1_1 = nn.BatchNorm2d(32)
        self.bnmid1_2 = nn.BatchNorm2d(64)
        self.bnmid1_3 = nn.BatchNorm2d(64)
        self.bnmid1_4 = nn.BatchNorm2d(64)
        self.bnmid1_5 = nn.BatchNorm2d(64)

        self.bnhigh1_1 = nn.BatchNorm2d(32)
        self.bnhigh1_2 = nn.BatchNorm2d(64)
        self.bnhigh1_3 = nn.BatchNorm2d(64)
        self.bnhigh1_4 = nn.BatchNorm2d(64)
        self.bnhigh1_5 = nn.BatchNorm2d(64)

        self.bnlow2_1 = nn.BatchNorm2d(64)
        self.bnlow2_2 = nn.BatchNorm2d(64)

        self.bnmid2_1 = nn.BatchNorm2d(64)
        self.bnmid2_2 = nn.BatchNorm2d(64)

        self.bnhigh2_1 = nn.BatchNorm2d(64)
        self.bnhigh2_2 = nn.BatchNorm2d(64)

        self.bnlow3_1 = nn.BatchNorm2d(64)
        self.bnmid3_1 = nn.BatchNorm2d(64)
        self.bnhigh3_1 = nn.BatchNorm2d(64)

        self.bnlow4_1 = nn.BatchNorm2d(64)
        self.bnmid4_1 = nn.BatchNorm2d(64)
        self.bnhigh4_1 = nn.BatchNorm2d(64)

        self.bnlow5_1 = nn.BatchNorm2d(64)
        self.bnmid5_1 = nn.BatchNorm2d(64)
        self.bnhigh5_1 = nn.BatchNorm2d(64)

        self.final_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x_mid = nn.functional.interpolate(x, scale_factor=1.5)
        x_high = nn.functional.interpolate(x, scale_factor=2)

        # 编码器生成特征
        low_feat_1 = self.relu(self.encoder_conv1(x))
        low_feat_2 = self.encoder2_1(low_feat_1)
        low_feat_3 = self.encoder3_1(low_feat_2)
        low_feat_4 = self.encoder4_1(low_feat_3)
        low_feat_5 = self.encoder5_1(low_feat_4)

        mid_feat_1 = self.relu(self.encoder_conv1(x_mid))
        mid_feat_2 = self.encoder2_2(mid_feat_1)
        mid_feat_3 = self.encoder3_2(mid_feat_2)
        mid_feat_4 = self.encoder4_2(mid_feat_3)
        mid_feat_5 = self.encoder5_2(mid_feat_4)

        high_feat_1 = self.relu(self.encoder_conv1(x_high))
        high_feat_2 = self.encoder2_3(high_feat_1)
        high_feat_3 = self.encoder3_3(high_feat_2)
        high_feat_4 = self.encoder4_3(high_feat_3)
        high_feat_5 = self.encoder5_3(high_feat_4)

        low_feat_1 = self.relu(self.bnlow1_1(self.low_conv1_1(low_feat_1)))
        low_feat_1 = self.relu(self.bnlow1_2(self.low_conv1_2(low_feat_1)))
        low_feat_1 = self.relu(self.bnlow1_3(self.low_conv1_3(low_feat_1)))
        low_feat_1 = self.relu(self.bnlow1_4(self.low_conv1_4(low_feat_1)))
        low_feat_1 = self.relu(self.bnlow1_5(self.low_conv1_5(low_feat_1)))
        mid_feat_1 = self.relu(self.bnmid1_1(self.mid_conv1_1(mid_feat_1)))
        mid_feat_1 = self.relu(self.bnmid1_2(self.mid_conv1_2(mid_feat_1)))
        mid_feat_1 = self.relu(self.bnmid1_3(self.mid_conv1_3(mid_feat_1)))
        mid_feat_1 = self.relu(self.bnmid1_4(self.mid_conv1_4(mid_feat_1)))
        mid_feat_1 = self.relu(self.bnmid1_5(self.mid_conv1_5(mid_feat_1)))
        high_feat_1 = self.relu(self.bnhigh1_1(self.high_conv1_1(high_feat_1)))
        high_feat_1 = self.relu(self.bnhigh1_2(self.high_conv1_2(high_feat_1)))
        high_feat_1 = self.relu(self.bnhigh1_3(self.high_conv1_3(high_feat_1)))
        high_feat_1 = self.relu(self.bnhigh1_4(self.high_conv1_4(high_feat_1)))
        high_feat_1 = self.relu(self.bnhigh1_5(self.high_conv1_5(high_feat_1)))

        low_feat_2 = self.relu(self.bnlow2_1(self.low_conv2_1(low_feat_2)))
        low_feat_2 = self.relu(self.bnlow2_2(self.low_conv2_2(low_feat_2)))
        mid_feat_2 = self.relu(self.bnmid2_1(self.mid_conv2_1(mid_feat_2)))
        mid_feat_2 = self.relu(self.bnmid2_2(self.mid_conv2_2(mid_feat_2)))
        high_feat_2 = self.relu(self.bnhigh2_1(self.high_conv2_1(high_feat_2)))
        high_feat_2 = self.relu(self.bnhigh2_2(self.high_conv2_2(high_feat_2)))

        low_feat_3 = self.relu(self.bnlow3_1(self.low_conv3_1(low_feat_3)))
        mid_feat_3 = self.relu(self.bnmid3_1(self.mid_conv3_1(mid_feat_3)))
        high_feat_3 = self.relu(self.bnhigh3_1(self.high_conv3_1(high_feat_3)))

        low_feat_4 = self.relu(self.bnlow4_1(self.low_conv4_2(low_feat_4)))
        mid_feat_4 = self.relu(self.bnmid4_1(self.mid_conv4_2(mid_feat_4)))
        high_feat_4 = self.relu(self.bnhigh4_1(self.high_conv4_2(high_feat_4)))

        low_feat_5 = self.relu(self.bnlow5_1(self.low_conv5_1(low_feat_5)))
        mid_feat_5 = self.relu(self.bnmid5_1(self.mid_conv5_1(mid_feat_5)))
        high_feat_5 = self.relu(self.bnhigh5_1(self.high_conv5_1(high_feat_5)))

        # SFM
        f1 = self.sfm_5(low_feat_1, mid_feat_1, high_feat_1)
        f2 = self.sfm_4(low_feat_2, mid_feat_2, high_feat_2)
        f3 = self.sfm_3(low_feat_3, mid_feat_3, high_feat_3)
        f4 = self.sfm_2(low_feat_4, mid_feat_4, high_feat_4)
        f5 = self.sfm_1(low_feat_5, mid_feat_5, high_feat_5)

        # CFM
        cfm_5 = self.CFM_5(f5)
        c5 = cfm_5 + f5

        cfm_4 = self.CFM_4(f4 + self.upsample(c5))
        c4 = cfm_4 + f4

        cfm_3 = self.CFM_3(f3 + self.upsample(c4))
        c3 = cfm_3 + f3

        cfm_2 = self.CFM_2(f2 + self.upsample(c3))
        c2 = cfm_2 + f2

        cfm_1 = self.CFM_1(f1 + self.upsample(c2))
        c1 = cfm_1 + f1

        output = self.sigmoid(self.final_bn(self.final_conv(c1)))
        return output

# Dice损失函数
def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs):
    F1_BEST = 0
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.to(device).float(), masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)

            loss = 10 * criterion(outputs, masks) + 0.2 * dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 更新学习率
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        F1_score = 0

        for img_path, mask_path in zip(val_images, val_masks):
            # 加载图片和掩码
            image, original_size = preprocess_image(img_path)
            mask = Image.open(mask_path).convert('L')
            mask = transforms.ToTensor()(mask)  # 转为张量

            # 执行模型预测
            image = image.unsqueeze(0).to(device)  # 增加 batch 维度
            with torch.no_grad():
                output = model(image)  # 模型输出 (128x128)

            # 将输出调整回原始尺寸
            output_resized = resize_to_original(output, original_size).cpu()
            preds_binary = (output_resized > 0.7).int()  # 二值化预测结果

            # 收集预测和真实值
            all_preds.append(preds_binary.flatten())
            all_labels.append(mask.flatten())

        # 拼接所有预测和真实值
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # 计算总的 F1 score
        output_image = np.squeeze(all_preds)
        gt_image = np.squeeze(all_labels)
        out_bin = output_image
        gt_bin = gt_image
        recall_SIRST = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(gt_bin))
        prec_SIRST = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(out_bin))
        F1_SIRST = 2 * recall_SIRST * prec_SIRST / np.maximum(0.001, recall_SIRST + prec_SIRST)
        F1_score += recall_SIRST
        F1_score += prec_SIRST

        IOU_SIRST = compute_iou(all_labels, all_preds)

        print(f"recall Score: {recall_SIRST:.4f}")
        print(f"prec Score: {prec_SIRST:.4f}")
        print(f"F1 Score in SIRST: {F1_SIRST:.4f}")
        print(f"IOU in SIRST: {IOU_SIRST:.4f}")

        all_preds = []
        all_labels = []

        for img_path, mask_path in zip(val_images1, val_masks1):
            # 加载图片和掩码
            image, original_size = preprocess_image(img_path)
            mask = Image.open(mask_path).convert('L')
            mask = transforms.ToTensor()(mask)  # 转为张量

            # 执行模型预测
            image = image.unsqueeze(0).to(device)  # 增加 batch 维度
            with torch.no_grad():
                output = model(image)  # 模型输出 (128x128)

            # 将输出调整回原始尺寸
            output_resized = resize_to_original(output, original_size).cpu()
            preds_binary = (output_resized > 0.6).int()  # 二值化预测结果

            # 收集预测和真实值
            all_preds.append(preds_binary.flatten())
            all_labels.append(mask.flatten())

        # 拼接所有预测和真实值
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        output_image = np.squeeze(all_preds)
        gt_image = np.squeeze(all_labels)
        out_bin = output_image
        gt_bin = gt_image
        recall_MDvsFA = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(gt_bin))
        prec_MDvsFA = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(out_bin))
        F1_MDvsFA = 2 * recall_MDvsFA * prec_MDvsFA / np.maximum(0.001, recall_MDvsFA + prec_MDvsFA)
        F1_score += recall_MDvsFA
        F1_score += prec_MDvsFA

        IOU_MDvsFA = compute_iou(all_labels, all_preds)

        print(f"recall Score: {recall_MDvsFA:.4f}")
        print(f"prec Score: {prec_MDvsFA:.4f}")
        print(f"F1 Score in FDvsMA: {F1_MDvsFA:.4f}")
        print(f"IOU in FDvsMA: {IOU_MDvsFA:.4f}")

        if F1_score > F1_BEST:
            F1_BEST = F1_score
            print(f"BEST_SCORE: {F1_BEST:.4f}")
            torch.save(model.state_dict(), model_path)
            print("Model weights saved to 'model_weights.pth'")

        wandb.log(
            {
                "epoch": epoch + 1,
                "f1_score in MDvsFA": F1_MDvsFA,
                "f1_score in SIRST": F1_SIRST,
                "IOU in MDvsFA": IOU_MDvsFA,
                "IOU in SIRST": IOU_SIRST,
                "train_loss": running_loss,
            }
        )

def preprocess_image(image_path):
    """加载图片并调整为 128x128 大小"""
    image = Image.open(image_path).convert('L')  # 灰度化
    original_size = image.size  # 保存原始尺寸 (width, height)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image, original_size

def resize_to_original(output, original_size):
    height, width = original_size[1], original_size[0]  # 转换为 (height, width)
    return torch.nn.functional.interpolate(output, size=(height, width), mode='bilinear', align_corners=False)

def compute_iou(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    iou = intersection / union if union > 0 else 0.0
    return iou

# 数据路径
image_dir = 'Dataset/数据集/训练集/image'
mask_dir = 'Dataset/数据集/训练集/mask'
train_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
train_masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]

image_dir1 = 'Dataset/数据集/测试集/SIRST/image'
mask_dir1 = 'Dataset/数据集/测试集/SIRST/mask'
image_dir2 = 'Dataset/数据集/测试集/MDvsFA/image'
mask_dir2 = 'Dataset/数据集/测试集/MDvsFA/mask'

val_images = [os.path.join(image_dir1, f) for f in os.listdir(image_dir1) if f.endswith('.png')]
val_masks = [os.path.join(mask_dir1, f) for f in os.listdir(mask_dir1) if f.endswith('.png')]
val_images1 = [os.path.join(image_dir2, f) for f in os.listdir(image_dir2) if f.endswith('.png')]
val_masks1 = [os.path.join(mask_dir2, f) for f in os.listdir(mask_dir2) if f.endswith('.png')]

save_dir = 'result/MDvsFA'
os.makedirs(save_dir, exist_ok=True)

# 模型保存路径
model_dir = 'result'
os.makedirs(model_dir, exist_ok=True)  # 如果路径不存在，自动创建
model_path = os.path.join(model_dir, "model_weights.pth")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(train_images, train_masks, transform)
val_dataset = CustomDataset(val_images, val_masks, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

if __name__ == "__main__":

    # 训练模型
    model = BPRNet().to(device)
    train_model(model, train_loader, val_loader, 20)
