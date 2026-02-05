import torch
import numpy as np
import os
import torch.nn.functional as F
import time
from thop import profile
import utils

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

        # Ensure the model is on the correct device (cuda:0)
        self.diffusion.model.to('cuda:0')  # Move model to CUDA device 0 explicitly

        # If the model is wrapped in DataParallel, make sure it's moved to the correct device
        if isinstance(self.diffusion.model, torch.nn.DataParallel):
            self.diffusion.model = self.diffusion.model.module
        self.diffusion.model.to('cuda:0')  # Ensure it is explicitly on cuda:0

        # Print model parameters and FLOPs
        self._print_model_params_and_flops()

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.test_dataset)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x_cond = x[:, :3, :, :].to(self.diffusion.device)

                # Ensure input has a batch size of 1
                if x_cond.ndimension() == 3:
                    x_cond = x_cond.unsqueeze(0)  # Add batch dimension if missing

                # Measure the inference time for a single batch
                start_time = time.time()  # Start timing

                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_output = self.diffusive_restoration(x_cond)
                x_output = x_output[:, :, :h, :w]
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))
                
                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time  # Calculate the elapsed time
                print(f"Processing image {y[0]} - Inference time: {elapsed_time:.4f} seconds")

    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]

    def _print_model_params_and_flops(self):
        # Calculate model parameters
        num_params = sum(p.numel() for p in self.diffusion.model.parameters())
        print(f"Total Parameters: {num_params}")

        # Calculate FLOPs (using thop)
        # Move the dummy input to the correct device (cuda:0)
        x_dummy = torch.randn(1, 3, 256, 256).to('cuda:0')  # Ensure input is on the same device
        flops, _ = profile(self.diffusion.model, inputs=(x_dummy,))
        print(f"FLOPs: {flops / 1e9} GFLOPs")  # FLOPs in GFLOPs (giga FLOPs)
