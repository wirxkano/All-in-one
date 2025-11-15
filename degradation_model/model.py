import random
import numpy as np
import albumentations as A
from degradation_model.blindsr import *
from torch import Tensor
from torchvision import transforms
from torch.nn import functional as F


class DegradationModel:
    def __init__(self, settings):
        self.blur_settings = settings["blur_settings"]
        self.sinc_settings = settings["sinc_settings"]
        self.noise_settings = settings["noise_settings"]
        self.jpeg_compression_settings = settings["jpeg_compression_settings"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_blur_kernel(self, image_tensor: Tensor, blur_range: tuple):
        image_tensor = image_tensor.to(self.device)
        kernel_list = self.blur_settings["kernel_list"]
        kernel_prob = self.blur_settings["kernel_prob"]
        kernel_size = random.choice(self.blur_settings["kernel_size"])
        sigma_x_range = blur_range[0]
        sigma_y_range = blur_range[1]
        theta_range = self.blur_settings["theta"]
        betag_range = self.blur_settings["betag"]
        betap_range = self.blur_settings["betap"]
        blur_kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            theta_range,
            betag_range,
            betap_range,
        )

        blur_kernel = torch.FloatTensor(blur_kernel).unsqueeze(0).unsqueeze(0)
        blur_kernel = blur_kernel.repeat(3, 1, 1, 1).to(self.device)

        padding = kernel_size // 2
        image_tensor = F.pad(
            image_tensor, (padding, padding, padding, padding), mode="reflect"
        )
        image_tensor = F.conv2d(image_tensor, blur_kernel, padding=0, groups=3)

        return image_tensor

    def add_noise(self, image_tensor: Tensor, noise_range: tuple):
        noise_list, noise_prob = (
            self.noise_settings["noise_list"],
            self.noise_settings["noise_prob"],
        )
        noise_type = random.choices(noise_list, noise_prob)[0]
        sigma_range = noise_range[0]
        scale_range = noise_range[1]
        if noise_type == "poisson":
            return random_add_poisson_noise_pt(image_tensor, scale_range)

        return random_add_gaussian_noise_pt(image_tensor, sigma_range)

    def add_jpeg_compression(self, image_tensor: Tensor):
        image_tensor = image_tensor.to(self.device)
        B, C, H, W = image_tensor.shape

        quality_range = self.jpeg_compression_settings["quality_range"]
        imgs = []
        for i in range(B):
            img = image_tensor[i]
            image_numpy = img.permute(1, 2, 0).cpu().numpy()
            image_numpy = random_add_jpg_compression(image_numpy, quality_range)
            img_t = torch.from_numpy(image_numpy).permute(2, 0, 1).float()
            img_t = transforms.Resize((H, W))(img_t)
            imgs.append(img_t)

        return torch.stack(imgs, dim=0)

    def add_sinc_kernel(self, image_tensor: Tensor):
        image_tensor = image_tensor.to(self.device)
        kernel_size = random.choice(self.sinc_settings["kernel_size"])
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)

        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel).unsqueeze(0)
        sinc_kernel = sinc_kernel.repeat(3, 1, 1, 1).to(self.device)
        kernel_size = sinc_kernel.size(-1)
        padding = kernel_size // 2
        image_tensor = F.pad(
            image_tensor, (padding, padding, padding, padding), mode="reflect"
        )
        image_tensor = F.conv2d(image_tensor, sinc_kernel, padding=0, groups=3)

        return image_tensor

    def _gaussian_focus_mask(self, h, w, cx, cy, sigma, device="cpu"):
        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        mask = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        return mask

    def add_region_dependent_blur(self, image_tensor: Tensor):
        B, C, H, W = image_tensor.shape

        blur_range_list = list(
            zip(self.blur_settings["sigma_x"], self.blur_settings["sigma_y"])
        )
        blurred = [
            self.add_blur_kernel(image_tensor, blur_range_list[i]) for i in range(3)
        ]
        mask = self._gaussian_focus_mask(
            H,
            W,
            cx=random.randint(0, H),
            cy=random.randint(0, W),
            sigma=random.uniform(40, 60),
        )
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W).to(self.device)

        w0 = mask
        w1 = (1 - mask) * mask
        w2 = (1 - mask) ** 2

        weights = w0 + w1 + w2 + 1e-8
        w0, w1, w2 = w0 / weights, w1 / weights, w2 / weights

        out = w0 * blurred[0] + w1 * blurred[1] + w2 * blurred[2]

        return out

    def add_region_dependent_noise(self, image_tensor: Tensor):
        B, C, H, W = image_tensor.shape

        noise_range_list = list(
            zip(self.noise_settings["sigma"], self.noise_settings["scale"])
        )
        noised = [self.add_noise(image_tensor, noise_range_list[i]) for i in range(3)]
        mask = self._gaussian_focus_mask(
            H,
            W,
            cx=random.randint(0, H),
            cy=random.randint(0, W),
            sigma=random.uniform(40, 60),
        )
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W).to(self.device)

        w0 = (1 - mask) ** 2
        w1 = (1 - mask) * mask
        w2 = mask

        weights = w0 + w1 + w2 + 1e-8
        w0, w1, w2 = w0 / weights, w1 / weights, w2 / weights

        out = (
            w0 * noised[0].to(self.device)
            + w1 * noised[1].to(self.device)
            + w2 * noised[2].to(self.device)
        )

        return out

    def add_brightness_contrast(self, image_tensor: Tensor):
        B, C, H, W = image_tensor.shape
        transform = A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.1, p=1.0
        )
        outs = []
        for i in range(B):
            img = image_tensor[i]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            out_np = transform(image=img_np)["image"]

            out_t = torch.from_numpy(out_np).permute(2, 0, 1).to(self.device).float()
            outs.append(out_t)

        return torch.stack(outs, dim=0)

    def random_shuffle_degradations(self, image_tensor: Tensor, scale_factor=0.25):
        image_tensor = image_tensor.to(self.device)
        down_sampling_modes = ["bilinear", "bicubic", "area"]
        degradation_list = [
            self.add_region_dependent_blur,
            self.add_region_dependent_noise,
            self.add_jpeg_compression,
            self.add_sinc_kernel,
            self.add_brightness_contrast,
        ]

        random.shuffle(degradation_list)

        for degradation in degradation_list:
            image_tensor = degradation(image_tensor)
            image_tensor = torch.clamp(image_tensor, 0, 1)

        image_tensor = F.interpolate(
            image_tensor,
            scale_factor=scale_factor,
            mode=random.choice(down_sampling_modes),
        )
        image_tensor = torch.clamp(image_tensor, 0, 1)

        return image_tensor
