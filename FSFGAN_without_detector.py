import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
from PIL import Image
import random


import os




class Generator(nn.Module):
    def __init__(self, latent_dim, output_channels, img_size=256):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            self._make_gen_block(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            self._make_gen_block(512, 256, kernel_size=4, stride=2, padding=1),
            self._make_gen_block(256, 128, kernel_size=4, stride=2, padding=1),
            self._make_gen_block(128, 64, kernel_size=4, stride=2, padding=1),
            self._make_gen_block(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
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

'''
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
        return self.model(x).view(-1)  # 将输出展平为一维
'''

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.model(x).view(-1)



def gradient_penalty(discriminator, real_samples, fake_samples, device):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    if fake_samples.size() != real_samples.size():
        fake_samples = F.interpolate(fake_samples, size=real_samples.size()[2:], mode='bilinear', align_corners=False)
    
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty







class WildImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        #print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        #print(f"Trying to open image: {image_path}")
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            #print(f"Error opening image {image_path}: {str(e)}")
            raise


def enhance_color_blocks(image, block_size=4):
    """
    增强图像中的大色块
    """
    b, c, h, w = image.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[:, :, i:i+block_size, j:j+block_size]
            mean_color = block.mean(dim=(2, 3), keepdim=True)
            image[:, :, i:i+block_size, j:j+block_size] = mean_color
    return image


def color_quantization(image, num_colors=4):
    """
    将图像量化为指定数量的颜色
    """
    b, c, h, w = image.shape
    image_flat = image.view(b, c, -1).permute(0, 2, 1).contiguous()
    
    # 使用 K-means 进行颜色聚类
    cluster_ids_x, cluster_centers = kmeans(
        X=image_flat.view(-1, c), num_clusters=num_colors, distance='euclidean', device=image.device
    )
    
    # 将每个像素分配到最近的聚类中心
    quantized_image = cluster_centers[cluster_ids_x].view(b, h, w, c)
    return quantized_image.permute(0, 3, 1, 2).contiguous()

def edge_color_interpolation(image, edge_width=10):
    """
    在图像边缘应用颜色插值
    """
    b, c, h, w = image.shape
    
    # 计算边缘的平均颜色
    edge_color = torch.cat([
        image[:, :, :edge_width, :].mean(dim=(2, 3), keepdim=True),
        image[:, :, -edge_width:, :].mean(dim=(2, 3), keepdim=True),
        image[:, :, :, :edge_width].mean(dim=(2, 3), keepdim=True),
        image[:, :, :, -edge_width:].mean(dim=(2, 3), keepdim=True)
    ], dim=3).mean(dim=3, keepdim=True)
    
    # 创建插值权重
    x = torch.linspace(0, 1, w).view(1, 1, 1, -1).to(image.device)
    y = torch.linspace(0, 1, h).view(1, 1, -1, 1).to(image.device)
    weight = torch.min(x, 1-x) * torch.min(y, 1-y)
    weight = weight.expand(b, c, h, w)
    
    # 应用插值
    interpolated_image = image * weight + edge_color * (1 - weight)
    return interpolated_image

def apply_camouflage(image, camouflage, position):
    applied = image.clone()
    h, w = position
    
    # 计算应用区域的大小
    h_size = min(camouflage.size(2), applied.size(2) - h)
    w_size = min(camouflage.size(3), applied.size(3) - w)
    
    # 如果必要，调整camouflage的大小
    if camouflage.size(2) != h_size or camouflage.size(3) != w_size:
        camouflage = F.interpolate(camouflage, size=(h_size, w_size), mode='bilinear', align_corners=False)


    # 应用颜色量化和边缘插值
    #camouflage = color_quantization(camouflage, num_colors=4)
    #applied = edge_color_interpolation(applied)
    camouflage = enhance_color_blocks(camouflage,block_size=8)
    
    # 替换图像的一部分，而不是叠加
    applied[:, :, h:h+h_size, w:w+w_size] = camouflage[:, :, :h_size, :w_size]
    
    return applied

def kmeans(X, num_clusters, distance='euclidean', tol=1e-4, max_iter=300, device=None):
    """
    Perform K-means clustering using PyTorch.
    
    Args:
    X: tensor of shape (n_samples, n_features)
    num_clusters: number of clusters
    distance: distance metric ('euclidean' or 'cosine')
    tol: tolerance for stopping criterion
    max_iter: maximum number of iterations
    device: device to run the algorithm on
    
    Returns:
    cluster_ids: tensor of shape (n_samples,)
    cluster_centers: tensor of shape (num_clusters, n_features)
    """
    if device is None:
        device = X.device
    
    n_samples, n_features = X.shape
    
    # Randomly initialize cluster centers
    idx = torch.randperm(n_samples)[:num_clusters]
    cluster_centers = X[idx].clone()
    
    for _ in range(max_iter):
        # Compute distances
        if distance == 'euclidean':
            dists = torch.cdist(X, cluster_centers)
        elif distance == 'cosine':
            dists = 1 - F.cosine_similarity(X.unsqueeze(1), cluster_centers.unsqueeze(0), dim=2)
        else:
            raise ValueError("Invalid distance metric")
        
        # Assign samples to nearest cluster
        cluster_ids = torch.argmin(dists, dim=1)
        
        # Update cluster centers
        new_cluster_centers = torch.zeros_like(cluster_centers)
        for k in range(num_clusters):
            if (cluster_ids == k).sum() > 0:
                new_cluster_centers[k] = X[cluster_ids == k].mean(dim=0)
            else:
                new_cluster_centers[k] = cluster_centers[k]
        
        # Check for convergence
        if torch.norm(new_cluster_centers - cluster_centers) < tol:
            break
        
        cluster_centers = new_cluster_centers
    
    return cluster_ids, cluster_centers




def train(generator, discriminator, dataloader, pre_epochs, num_epochs, device, latent_dim):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    
    mse_loss = nn.MSELoss()
    
    # Pre-training generator
    print("Pre-training generator...")
    for epoch in range(pre_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_camouflage = generator(z)
            
            if fake_camouflage.size() != real_images.size():
                fake_camouflage = F.interpolate(fake_camouflage, size=real_images.size()[2:], mode='bilinear', align_corners=False)
            
            g_loss = mse_loss(fake_camouflage, real_images)
            g_loss.backward()
            g_optimizer.step()
            
            if i % 10 == 0:
                print(f'Pre-training Epoch [{epoch+1}/{pre_epochs}], Step [{i}/{len(dataloader)}], g_loss: {g_loss.item():.4f}')

               # Save generated image every 5 epochs
        if (epoch + 1) % 5 == 0:
            position = (random.randint(0, max(0, real_images.size(2) - fake_camouflage.size(2))), 
                        random.randint(0, max(0, real_images.size(3) - fake_camouflage.size(3))))
            with torch.no_grad():
                fake_image = generator(torch.randn(1, latent_dim, 1, 1).to(device))
                camouflaged_images = apply_camouflage(real_images[:1], fake_image, position)
            vutils.save_image(fake_image, f'generated_images/fake_image_epoch_{epoch+1}.png', normalize=True)
            vutils.save_image(camouflaged_images, f'generated_images/fake_camouflage_epoch_{epoch+1}.png', normalize=True)
            print(f"Saved generated image for epoch {epoch+1}")

    print("Starting joint training...")
    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Train discriminator
            d_optimizer.zero_grad()
            
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_camouflage = generator(z)
            
            if fake_camouflage.size() != real_images.size():
                fake_camouflage = F.interpolate(fake_camouflage, size=real_images.size()[2:], mode='bilinear', align_corners=False)
            
            position = (random.randint(0, max(0, real_images.size(2) - fake_camouflage.size(2))), 
                        random.randint(0, max(0, real_images.size(3) - fake_camouflage.size(3))))
            
            camouflaged_images = apply_camouflage(real_images, fake_camouflage, position)
            
            real_validity = discriminator(real_images)
            fake_validity = discriminator(camouflaged_images.detach())
            
            gp = gradient_penalty(discriminator, real_images, camouflaged_images, device)
            
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + 10 * gp
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            if i % 5 == 0:
                g_optimizer.zero_grad()
                gen_camouflage = generator(z)
                if gen_camouflage.size() != real_images.size():
                    gen_camouflage = F.interpolate(gen_camouflage, size=real_images.size()[2:], mode='bilinear', align_corners=False)
                
                camouflaged_gen_images = apply_camouflage(real_images, gen_camouflage, position)
                gen_validity = discriminator(camouflaged_gen_images)
                
                mse = mse_loss(gen_camouflage, real_images)
                g_loss = -0.01*torch.mean(gen_validity)# + 0.1 * mse  # Add MSE loss with a smaller weight
                g_loss.backward()
                g_optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], '
                      f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
        
        # Save generated image every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                fake_image = generator(torch.randn(1, latent_dim, 1, 1).to(device))
                camouflaged_images = apply_camouflage(real_images[:1], fake_image, position)
            vutils.save_image(fake_image, f'generated_images/fake_image_epoch_{epoch+1}.png', normalize=True)
            vutils.save_image(camouflaged_images, f'generated_images/fake_camouflage_epoch_{epoch+1}.png', normalize=True)
            print(f"Saved generated image for epoch {epoch+1}")




# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    latent_dim = 100
    image_size = 256 # 128
    batch_size = 128  # Make sure this matches your actual batch size
    #pre_epochs = 50
    #num_epochs = 100
    pre_epochs = 30
    num_epochs = 200

    # Initialize networks
    generator = Generator(latent_dim, 3, img_size=image_size).to(device)
    discriminator = Discriminator(3).to(device)

    # Prepare dataset
    transform = transforms.Compose([
        transforms.RandomCrop(image_size),
        #transforms.Resize(image_size),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = WildImageDataset('G:/cam-pytorch/dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Train the model
    train(generator, discriminator, dataloader, pre_epochs, num_epochs, device, latent_dim)

    # Save the trained generator
    torch.save(generator.state_dict(), 'camouflage_generator.pth')


if __name__ == "__main__":
    main()