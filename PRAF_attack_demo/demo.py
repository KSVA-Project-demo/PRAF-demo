import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation

class PRAF_Attack:
    def __init__(self, device='cuda'):
        self.device = device
        # 1. 加载替代模型 (Substitute Model: ResNet-50 based)
        # 这里使用预训练的 DeepLabV3-ResNet50 作为示例
        self.substitute_model = segmentation.deeplabv3_resnet50(pretrained=True).to(device)
        self.substitute_model.eval() # 固定替代模型参数

    # --- 模块 1: 潜变量驱动生成器 (Latent-Driven Generator) ---
    # 对应论文公式 (2)
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            # 简单的 U-Net 风格或 ResNet block 结构
            # 输入通道 = 3 (图像) + 噪声维度 (假设为 1 维扩展后 concat)
            self.conv1 = nn.Conv2d(3 + 1, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1) # 输出 3 通道补丁
            
        def forward(self, x_local, z_noise):
            # 将噪声 z 扩展到与图像相同的空间维度并拼接
            z_expanded = z_noise.view(z_noise.size(0), 1, 1, 1).expand(-1, 1, x_local.size(2), x_local.size(3))
            input_tensor = torch.cat([x_local, z_expanded], dim=1)
            
            x = F.relu(self.conv1(input_tensor))
            patch = torch.tanh(self.conv2(x)) # 限制在 [-1, 1] 或 [0, 1]
            return patch

    # --- 模块 2: 可微分物理仿真层 (Physical Simulation Layer) ---
    # 对应论文公式 (3) 和 (4) 
    class PhysicalSimulationLayer(nn.Module):
        def __init__(self):
            super().__init__()

        def motion_blur(self, x, kernel_size=15):
            # 模拟运动模糊：创建一个可微分的均值滤波核或方向性核
            # 这里的梯度流很重要：\partial L / \partial x_phys * k^T 
            k = torch.eye(kernel_size) / kernel_size
            k = k.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1).to(x.device)
            # 使用 depthwise卷积模拟模糊
            return F.conv2d(x, k, padding=kernel_size//2, groups=3)

        def forward(self, x):
            # 1. 应用运动模糊
            x_blur = self.motion_blur(x)
            
            # 2. 注入环境噪声 (Gaussian)
            noise = torch.randn_like(x) * 0.05 # sigma_env
            x_phys = x_blur + noise
            
            # 3. (可选) 还可以加入仿射变换 (Affine Trans)
            return x_phys

    # --- 模块 3: 损失函数定义 ---
    # 对应论文公式 (6) 和 (7)
    def calculate_loss(self, pred_adv, target, patch, clean_image):
        # A. 对抗损失 (Adversarial Loss): CrossEntropy + Feature Disruption
        # 这里简化为只用 CrossEntropy 让模型预测错误的目标（例如让它把路识别为背景）
        adv_loss = F.cross_entropy(pred_adv, target)
        
        # B. 隐蔽性损失 (Stealthiness Loss): TV Loss + Color Consistency
        tv_loss = torch.sum(torch.abs(patch[:, :, :, :-1] - patch[:, :, :, 1:])) + \
                  torch.sum(torch.abs(patch[:, :, :-1, :] - patch[:, :, 1:, :]))
        
        # 简单模拟 Color Consistency (L2 distance to local background)
        color_loss = F.mse_loss(patch, clean_image) 
        
        # 总损失 
        total_loss = adv_loss + 0.1 * tv_loss + 0.1 * color_loss
        return total_loss

    def train_step(self, image, target_label):
        generator = self.Generator().to(self.device)
        simulator = self.PhysicalSimulationLayer().to(self.device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)

        # 训练循环 (对应 Algorithm 1) 
        for i in range(100): # 简化迭代次数
            optimizer.zero_grad()
            
            # 1. 采样潜变量 z 
            z = torch.randn(image.size(0), 100).to(self.device) # 假设噪声维100
            
            # 2. 生成补丁 (Generate)
            raw_patch = generator(image, z)
            
            # 3. 贴图 (Masking) - 简单假设贴在图像中心
            # 实际需根据论文公式 (5) 实现 M * P + (1-M) * x 
            x_adv = image.clone()
            x_adv[:, :, 100:200, 100:200] = raw_patch[:, :, 0:100, 0:100] # 简化的贴图逻辑
            
            # 4. 物理仿真 (Sim-to-Real)
            # 关键：这里必须保留梯度
            x_phys = simulator(x_adv)
            
            # 5. 替代模型前向传播
            # 注意：需归一化适配 ResNet 输入
            pred = self.substitute_model(x_phys)['out']
            
            # 6. 计算损失并反向传播
            loss = self.calculate_loss(pred, target_label, raw_patch, image)
            loss.backward()
            
            # 7. 更新生成器参数
            optimizer.step()
            
            print(f"Iter {i}, Loss: {loss.item()}")

        return generator

# 使用示例
# praf = PRAF_Attack()
# praf.train_step(dummy_image, dummy_target)