# import os
# import torch
# from torch.utils.data import DataLoader
# from dataset import HazyGT_Dataset
# from model import LightweightTransformer
# import torchvision.transforms as T
# from PIL import Image
# import torchvision.utils as vutils

# def save_tensor_as_image(tensor, path):
#     # tensor shape: [3, H, W], 取值范围假设是0-1
#     tensor = tensor.clamp(0, 1)
#     vutils.save_image(tensor, path)

# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     test_hazy_dir = 'G:/Decloud-Dataset/T-Cloud/test/cloud'  # 只需雾图路径

#     transform = T.Compose([
#         T.ToTensor(),
#     ])

#     test_dataset = HazyGT_Dataset(test_hazy_dir, gt_dir=None, transform=transform)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     model = LightweightTransformer(in_channels=3, out_channels=6, embed_dim=64, patch_size=16)
#     model.load_state_dict(torch.load('best_model_OTS.pth', map_location=device))
#     model.to(device)
#     model.eval()

#     save_dir = 'test_results_C'
#     os.makedirs(save_dir, exist_ok=True)

#     with torch.no_grad():
#         for idx, hazy in enumerate(test_loader):  # 只返回雾图
#             hazy = hazy.to(device)

#             pred_M, pred_T, pred_A, pred_C, pred_J = model(hazy)

#             pred_img = pred_J.clamp(0,1)

#             save_path = os.path.join(save_dir, f"pred_{idx:04d}.png")
#             vutils.save_image(pred_img, save_path)

#             print(f"Saved prediction: {save_path}")

# if __name__ == "__main__":
#     main()

import os
import torch
from torch.utils.data import DataLoader
from dataset import HazyGT_Dataset
from model import LightweightTransformer
import torchvision.transforms as T
from PIL import Image
import torchvision.utils as vutils
import time  # For timing the inference
from thop import profile  # For calculating FLOPs

def save_tensor_as_image(tensor, path):
    # tensor shape: [3, H, W], 取值范围假设是0-1
    tensor = tensor.clamp(0, 1)
    vutils.save_image(tensor, path)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_hazy_dir = 'G:/Decloud-Dataset/T-Cloud/test/cloud'  # 只需雾图路径

    transform = T.Compose([
        T.ToTensor(),
    ])

    test_dataset = HazyGT_Dataset(test_hazy_dir, gt_dir=None, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = LightweightTransformer(in_channels=3, out_channels=6, embed_dim=64, patch_size=16)
    model.load_state_dict(torch.load('best_model_OTS.pth', map_location=device))
    model.to(device)
    model.eval()

    # Calculate model parameters and FLOPs
    _print_model_params_and_flops(model, device)

    save_dir = 'test'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, hazy in enumerate(test_loader):  # 只返回雾图
            hazy = hazy.to(device)

            # Start timing the inference for each image
            start_time = time.time()

            pred_M, pred_T, pred_A, pred_C, pred_J = model(hazy)

            # End timing
            end_time = time.time()
            elapsed_time = end_time - start_time  # Inference time for one image

            pred_img = pred_J.clamp(0, 1)

            save_path = os.path.join(save_dir, f"pred_{idx:04d}.png")
            vutils.save_image(pred_img, save_path)

            print(f"Saved prediction: {save_path} - Inference time: {elapsed_time:.4f} seconds")

def _print_model_params_and_flops(model, device):
    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_params}")

    # Calculate FLOPs using thop
    x_dummy = torch.randn(1, 3, 256, 256).to(device)  # A dummy input tensor
    flops, _ = profile(model, inputs=(x_dummy,))
    print(f"FLOPs: {flops / 1e9} GFLOPs")  # Convert FLOPs to GFLOPs (Giga FLOPs)

if __name__ == "__main__":
    main()
