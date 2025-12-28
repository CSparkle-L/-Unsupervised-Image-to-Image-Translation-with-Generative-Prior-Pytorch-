import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm

# 1. 定义数据集加载类
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# 2. 加载Inception v3模型（提取2048维特征）
def load_inception_model():
    model = models.inception_v3(pretrained=True, transform_input=False)
    # 移除最后一层分类层，保留池化层输出
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

# 3. 提取图像特征
def extract_features(dataloader, model, device):
    features = []
    with torch.no_grad():
        for img in dataloader:
            img = img.to(device)
            # Inception v3要求输入至少299×299，且需要batch维度
            if img.shape[2] != 299 or img.shape[3] != 299:
                img = nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
            feat = model(img)
            features.append(feat.view(feat.size(0), -1).cpu().numpy())
    return np.concatenate(features, axis=0)

# 4. 计算FID值
def calculate_fid(real_features, gen_features):
    # 计算均值和协方差
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    # 计算均值差的平方和
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # 计算协方差的平方根（使用scipy的sqrtm）
    covmean = sqrtm(sigma1.dot(sigma2))
    # 处理复数情况
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # 计算FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

if __name__ == "__main__":
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    real_img_dir = './data/real/'  # 真实目标域图像
    gen_img_dir = './data/generated/'  # 生成的图像

    # 定义图像变换（匹配Inception v3的预处理）
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    real_dataset = ImageDataset(real_img_dir, transform)
    gen_dataset = ImageDataset(gen_img_dir, transform)
    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    inception_model = load_inception_model().to(device)

    # 提取特征
    print("Extracting real features...")
    real_features = extract_features(real_dataloader, inception_model, device)
    print("Extracting generated features...")
    gen_features = extract_features(gen_dataloader, inception_model, device)

    # 计算FID
    fid_value = calculate_fid(real_features, gen_features)
    print(f"FID Value: {fid_value:.2f}")