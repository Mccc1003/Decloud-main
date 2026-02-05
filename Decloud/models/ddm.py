import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from pytorch_msssim import ssim
from .attention import StructureAwareMeasure
from .model import LightweightTransformer

dehaze_model = LightweightTransformer(in_channels=3, out_channels=6, embed_dim=64, patch_size=16)
dehaze_model.load_state_dict(torch.load('/root/Decloud/models/best_model.pth'))
dehaze_model.eval()

def rgb_to_grayscale(tensor):
    grayscale = 0.2989 * tensor[:, 0, :, :] + 0.5870 * tensor[:, 1, :, :] + 0.1140 * tensor[:, 2, :, :]
    grayscale = grayscale.squeeze(0)
    return grayscale

def tensor_to_numpy(tensor):
    tensor = tensor.cpu()
    if tensor.requires_grad:
        tensor = tensor.detach()
    numpy_array = tensor.numpy()
    return numpy_array

def numpy_to_tensor(numpy_array):
    tensor = torch.from_numpy(numpy_array)
    tensor = tensor.cuda()
    return tensor

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Net(nn.Module):
    def __init__(self, args, config, dehaze_model):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DiffusionUNet(config)
        self.SAM = StructureAwareMeasure()
        self.dehaze_model = dehaze_model
        if self.dehaze_model is not None:
            for param in self.dehaze_model.parameters():
                param.requires_grad = False
            self.dehaze_model.eval()

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        Structure = self.SAM(x_cond)
        if self.dehaze_model is not None:
                with torch.no_grad():
                    pred_M, pred_T, pred_A, pred_C, pred_J = self.dehaze_model(x_cond)
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            
            et = self.Unet(torch.cat([x_cond, xt], dim=1), t, Structure=Structure, pred_T = pred_T, pred_J = pred_J)
            # et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        if self.dehaze_model is not None:
                with torch.no_grad():
                    pred_M, pred_T, pred_A, pred_C, pred_J = self.dehaze_model(input_img)

        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_img.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_img.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_img)

        if self.training:
            gt_img = x[:, 3:, :, :]
            x = gt_img * a.sqrt() + e * (1.0 - a).sqrt()
            diffusion_input = input_img
            Structure = self.SAM(input_img)
            noise_output = self.Unet(torch.cat([diffusion_input, x], dim=1), t.float(), Structure = Structure, pred_T = pred_T, pred_J = pred_J)
            # noise_output = self.Unet(torch.cat([diffusion_input, x], dim=1), t.float())
            denoise_y = self.sample_training(diffusion_input, b)
            pred_x = denoise_y

            gt_img_gray = tensor_to_numpy(rgb_to_grayscale(gt_img))
            pred_x_gray = tensor_to_numpy(rgb_to_grayscale(pred_x))

            data_dict["gt_img"] = gt_img
            data_dict["pred_x"] = pred_x
            data_dict["noise_output"] = noise_output
            data_dict["e"] = e

        else:
            diffusion_input = input_img
            denoise_y = self.sample_training(diffusion_input, b)
            pred_x = denoise_y

            data_dict["pred_x"] = pred_x

        return data_dict

class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config, dehaze_model=dehaze_model)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                output = self.model(x)

                image_loss, photo_loss, TV_loss = self.estimation_loss(x, output)
                loss = 10 * image_loss + 10 * photo_loss + TV_loss
                if self.step % 10 == 0:
                    print("step:{}, lr:{:.6f}, luminance_loss:{:.4f}, chroma_loss:{:.4f}, photo_loss:{:.4f}, FSIM_loss:{:.4f}, loss:{:.4f}"
                            .format(self.step, self.scheduler.get_last_lr()[0], image_loss.item(), photo_loss.item(), TV_loss.item(), loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                    utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir, 'model_latest'))
                        
            self.scheduler.step()
    
    def estimation_loss(self, x, output):
        enhanced_y, enhanced_cb, enhanced_cr, gt_y, gt_cb, gt_cr = output["enhanced_y"], output["enhanced_cb"], output["enhanced_cr"], output["gt_y"], output["gt_cb"], output["gt_cr"]
        pred_x, gt_x, noise_output, e = output["pred_x"], output["gt_img"], output["noise_output"], output["e"]
        pred_x_pc, gt_pc = output["pred_x_pc"], output["gt_pc"]
        
        # =============luminance loss==================
        image_loss = self.l1_loss(noise_output, e)

        # =============photo loss==================
        content_loss = self.l2_loss(pred_x, gt_x)
        ssim_loss = 1 - ssim(pred_x, gt_x, data_range=1.0).to(self.device)
        photo_loss = content_loss + ssim_loss

        # =============TV loss==================
        TV_loss = self.TV_loss(pred_x)

        return image_loss, photo_loss, TV_loss

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step), f"{y[0]}.png"))


