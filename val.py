from train import *

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型路径
model_dir = 'result'
model_path = os.path.join(model_dir, "model_weights.pth")

# 数据集路径
image_dir = 'Dataset/数据集/训练集/image'
mask_dir = 'Dataset/数据集/训练集/mask'
train_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
train_masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]

image_dir1 = 'Dataset/数据集/测试集/MDvsFA/image'
mask_dir1 = 'Dataset/数据集/测试集/MDvsFA/mask'
# image_dir1 = 'Dataset/数据集/测试集/SIRST/image'
# mask_dir1 = 'Dataset/数据集/测试集/SIRST/mask'
val_images = [os.path.join(image_dir1, f) for f in os.listdir(image_dir1) if f.endswith('.png')]
val_masks = [os.path.join(mask_dir1, f) for f in os.listdir(mask_dir1) if f.endswith('.png')]

# 加载模型
model = BPRNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print("Model loaded successfully.")

# 打印数据集大小
print(f"Number of test images: {len(val_images)}")
print(f"Number of test masks: {len(val_masks)}")

# **5. 逐张处理并计算 F1 score**

all_preds = []
all_labels = []

for img_path, mask_path in zip(val_images, val_masks):
# for img_path, mask_path in zip(train_images, train_masks):
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
f1_score_test = calculateF1Measure(all_preds, all_labels, thre=0.5)
# f1_score_test = f1_score(all_labels, all_preds)
print(f"Test Set F1 Score: {f1_score_test:.4f}")