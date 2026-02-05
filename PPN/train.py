import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HazyGT_Dataset
from model import LightweightTransformer
import torchvision.transforms as T
import torch.nn as nn
from pytorch_msssim import ssim

def constraint_variance_loss(pred):
    """鼓励pred不要全同（防止单一值），用反方差惩罚方差过小"""
    mean = torch.mean(pred, dim=[1,2,3], keepdim=True)
    var = torch.mean((pred - mean) ** 2, dim=[1,2,3])
    return torch.mean(1.0 / (var + 1e-6))

def constraint_mean_loss(pred, target_mean=0.5):
    """鼓励pred均值接近目标值，防止偏置过大或过小"""
    mean = torch.mean(pred, dim=[1,2,3])
    return torch.mean((mean - target_mean) ** 2)

def composite_loss(pred_M, pred_T, pred_A, pred_C, pred_J, target,
                   lambda_var=0.1, lambda_mean=0.1):
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # 基础重建损失对J_pred和目标清晰图像
    loss_mse = mse_loss(pred_J, target)
    loss_l1 = l1_loss(pred_J, target)
    loss_ssim = 1 - ssim(pred_J, target, data_range=1.0, size_average=True)

    base_loss = loss_mse + loss_l1 + 0.5 * loss_ssim

    # 对每个物理量添加约束
    losses_var = [
        constraint_variance_loss(pred_M),
        constraint_variance_loss(pred_T),
        constraint_variance_loss(pred_A),
        constraint_variance_loss(pred_C)
    ]
    losses_mean = [
        constraint_mean_loss(pred_M, target_mean=0.5),
        constraint_mean_loss(pred_T, target_mean=0.5),
        constraint_mean_loss(pred_A, target_mean=0.5),
        constraint_mean_loss(pred_C, target_mean=0.0)  # pred_C是tanh范围，均值一般0比较合适
    ]

    loss_var_total = sum(losses_var) / len(losses_var)
    loss_mean_total = sum(losses_mean) / len(losses_mean)

    loss = base_loss + lambda_var * loss_var_total + lambda_mean * loss_mean_total

    return loss

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    for hazy, gt in dataloader:
        hazy = hazy.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        pred_M, pred_T, pred_A, pred_C, pred_J = model(hazy)

        loss = criterion(pred_M, pred_T, pred_A, pred_C, pred_J, gt)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * hazy.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for hazy, gt in dataloader:
            hazy = hazy.to(device)
            gt = gt.to(device)

            pred_M, pred_T, pred_A, pred_C, pred_J = model(hazy)

            loss = criterion(pred_M, pred_T, pred_A, pred_C, pred_J, gt)

            running_loss += loss.item() * hazy.size(0)
    return running_loss / len(dataloader.dataset)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    train_dataset = HazyGT_Dataset('G:/Dehaze-Dataset/SateHaze/Haze1k_thick/train/hazy',
                                   'G:/Dehaze-Dataset/SateHaze/Haze1k_thick/train/GT',
                                   transform=transform)
    val_dataset = HazyGT_Dataset('G:/Dehaze-Dataset/SateHaze/Haze1k_thick/test/hazy',
                                 'G:/Dehaze-Dataset/SateHaze/Haze1k_thick/test/GT',
                                 transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = LightweightTransformer(in_channels=3, out_channels=6, embed_dim=64, patch_size=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = composite_loss

    best_val_loss = float('inf')
    num_epochs = 500

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model at epoch {epoch+1}")

if __name__ == "__main__":
    main()
