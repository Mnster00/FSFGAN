import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
from PIL import Image
import random
import torchvision.utils as vutils
import os

class Generator(nn.Module):
    def __init__(self, latent_dim, output_channels, img_size=128):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            self._make_gen_block(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            self._make_gen_block(512, 256, kernel_size=4, stride=2, padding=1),
            self._make_gen_block(256, 128, kernel_size=4, stride=2, padding=1),
            self._make_gen_block(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _make_gen_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0)),
        )

    def forward(self, x):
        return self.model(x).view(-1)

class WildImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

def kmeans(X, num_clusters, distance='euclidean', tol=1e-4, max_iter=300, device=None):
    if device is None:
        device = X.device
    
    n_samples, n_features = X.shape
    
    idx = torch.randperm(n_samples)[:num_clusters]
    cluster_centers = X[idx].clone()
    
    for _ in range(max_iter):
        if distance == 'euclidean':
            dists = torch.cdist(X, cluster_centers)
        elif distance == 'cosine':
            dists = 1 - F.cosine_similarity(X.unsqueeze(1), cluster_centers.unsqueeze(0), dim=2)
        else:
            raise ValueError("Invalid distance metric")
        
        cluster_ids = torch.argmin(dists, dim=1)
        
        new_cluster_centers = torch.zeros_like(cluster_centers)
        for k in range(num_clusters):
            if (cluster_ids == k).sum() > 0:
                new_cluster_centers[k] = X[cluster_ids == k].mean(dim=0)
            else:
                new_cluster_centers[k] = cluster_centers[k]
        
        if torch.norm(new_cluster_centers - cluster_centers) < tol:
            break
        
        cluster_centers = new_cluster_centers
    
    return cluster_ids, cluster_centers

def enhance_color_blocks(image, block_size=16):
    """
    Enhance large color blocks in the image
    """
    b, c, h, w = image.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[:, :, i:i+block_size, j:j+block_size]
            mean_color = block.mean(dim=(2, 3), keepdim=True)
            image[:, :, i:i+block_size, j:j+block_size] = mean_color
    return image

def color_quantization(image, num_colors=4):
    b, c, h, w = image.shape
    image_flat = image.view(b, c, -1).permute(0, 2, 1).contiguous()
    
    cluster_ids_x, cluster_centers = kmeans(
        X=image_flat.view(-1, c), num_clusters=num_colors, distance='euclidean', device=image.device
    )
    
    quantized_image = cluster_centers[cluster_ids_x].view(b, h, w, c)
    return quantized_image.permute(0, 3, 1, 2).contiguous()

def edge_color_interpolation(image, edge_width=10):
    b, c, h, w = image.shape
    
    edge_color = torch.cat([
        image[:, :, :edge_width, :].mean(dim=(2, 3), keepdim=True),
        image[:, :, -edge_width:, :].mean(dim=(2, 3), keepdim=True),
        image[:, :, :, :edge_width].mean(dim=(2, 3), keepdim=True),
        image[:, :, :, -edge_width:].mean(dim=(2, 3), keepdim=True)
    ], dim=3).mean(dim=3, keepdim=True)
    
    x = torch.linspace(0, 1, w).view(1, 1, 1, -1).to(image.device)
    y = torch.linspace(0, 1, h).view(1, 1, -1, 1).to(image.device)
    weight = torch.min(x, 1-x) * torch.min(y, 1-y)
    weight = weight.expand(b, c, h, w)
    
    interpolated_image = image * weight + edge_color * (1 - weight)
    return interpolated_image

def apply_camouflage(image, camouflage):
    applied = image.clone()
    b, c, h, w = applied.size()
    
    # 随机选择覆盖区域的位置
    h_start = random.randint(0, h - camouflage.size(2))
    w_start = random.randint(0, w - camouflage.size(3))
    
    # 计算应用区域的大小
    h_size = camouflage.size(2)
    w_size = camouflage.size(3)
    
    # 应用颜色量化和边缘插值
    camouflage = color_quantization(camouflage)
    camouflage = enhance_color_blocks(camouflage, block_size=4)
    # 替换图像的随机部分
    applied[:, :, h_start:h_start+h_size, w_start:w_start+w_size] = camouflage
    

    
    return applied, (h_start, w_start)


def test_and_save_images(generator, dataloader, device, latent_dim, output_dir):
    generator.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            
            z = torch.randn(real_images.size(0), latent_dim, 1, 1).to(device)
            fake_camouflage = generator(z)
            
            # Ensure fake_camouflage has the same size as real_images
            if fake_camouflage.size() != real_images.size():
                fake_camouflage = F.interpolate(fake_camouflage, size=real_images.size()[2:], mode='bilinear', align_corners=False)
            
            # Enhance color blocks in fake_camouflage
            fake_camouflage = enhance_color_blocks(fake_camouflage, block_size=32)
            
            # Apply camouflage at random position
            h_max = max(0, real_images.size(2) - fake_camouflage.size(2))
            w_max = max(0, real_images.size(3) - fake_camouflage.size(3))
            position = (random.randint(0, h_max), random.randint(0, w_max))

            camouflaged_images, position = apply_camouflage(real_images, fake_camouflage)
            
            # Create mask
            mask = torch.zeros_like(real_images)
            h_start, w_start = position
            h_size, w_size = fake_camouflage.size(2), fake_camouflage.size(3)
            mask[:, :, h_start:h_start+h_size, w_start:w_start+w_size] = 1

            
            # Move tensors to CPU for saving
            real_images = real_images.cpu()
            fake_camouflage = fake_camouflage.cpu()
            camouflaged_images = camouflaged_images.cpu()
            mask = mask.cpu()
            
            for j in range(real_images.size(0)):
                vutils.save_image(real_images[j], f'{output_dir}/real_image_{i}_{j}.png', normalize=True)
                vutils.save_image(fake_camouflage[j], f'{output_dir}/fake_camouflage_{i}_{j}.png', normalize=True)
                vutils.save_image(camouflaged_images[j], f'{output_dir}/camouflaged_image_{i}_{j}.png', normalize=True)
                vutils.save_image(mask[j], f'{output_dir}/mask_{i}_{j}.png', normalize=False)
            
            print(f'Saved images for batch {i}')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    latent_dim = 100
    image_size = 256#128
    batch_size = 1

    generator = Generator(latent_dim, 3).to(device)
    generator.load_state_dict(torch.load('camouflage_generator.pth'))

    # Prepare dataset
    transform = transforms.Compose([
        transforms.RandomCrop(image_size),
        #transforms.Resize(image_size),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = WildImageDataset('G:/cam-pytorch/dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    test_and_save_images(generator, dataloader, device, latent_dim, 'output_images')

if __name__ == "__main__":
    main()
