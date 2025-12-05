from PIL import Image
import os
import subprocess
import torch
from torchvision import transforms
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor


class DINOv2:
    # Initialize the class with model parameters and execution device.
    def __init__(self, repo_name='facebookresearch/dinov2', model_name='dinov2_vitb14', half_precision=False, device="cuda"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.half_precision = half_precision
        self.device = device

        if self.half_precision:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    # Preprocess the image and extract features
    def get_features(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)

        height, width = image_tensor.shape[1:]
        cropped_width = width - width % self.model.patch_size
        cropped_height = height - height % self.model.patch_size
        image_tensor_ac = image_tensor[:, :cropped_height, :cropped_width]
        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)

        with torch.inference_mode():
            image_batch = image_tensor_ac.unsqueeze(0).half().to(self.device) if self.half_precision else image_tensor_ac.unsqueeze(0).to(self.device)
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        del image_batch
        torch.cuda.empty_cache()

        return tokens.cpu().numpy(), grid_size

class DINOv3:
    # Initialize the class with model parameters and execution device.
    def __init__(self, repo_name='facebookresearch/dinov3', model_name='dinov3_vitb16', half_precision=False, device='cuda'):
        self.repo_name = repo_name
        self.model_name = model_name
        self.half_precision = half_precision
        self.device = device

        # If environment variable is set, dont download the model again
        if os.getenv('DINOV3_MODELS_LOCATION'):
            folder = os.getenv('DINOV3_MODELS_LOCATION')
            model_checkpoints = {   # DINOv3 ViT LVD-1689M
                'dinov3_vits16' : 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
                'dinov3_vits16plus' : 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
                'dinov3_vitb16' : 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
                'dinov3_vitl16' : 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
                'dinov3_vith16plus' : 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
            }
            file_path = os.path.join(folder, model_checkpoints[model_name])
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f'The model file {file_path} does not exist.')
            else:
                self.model = torch.hub.load(
                    repo_or_dir=repo_name,
                    model=model_name,
                    weights=file_path
                )
        else:
            # Load model from GitHub releases if not found locally
            self.model = torch.hub.load(
                repo_or_dir=repo_name,
                model=model_name,
                source='github',
            )

        if self.half_precision:
            self.model = self.model.half().to(self.device)
        else:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    # Preprocess the image and extract features
    def get_features(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)

        height, width = image_tensor.shape[1:]
        cropped_width = width - width % self.model.patch_size
        cropped_height = height - height % self.model.patch_size
        image_tensor_ac = image_tensor[:, :cropped_height, :cropped_width]
        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)

        with torch.inference_mode():
            image_batch = image_tensor_ac.unsqueeze(0).half().to(self.device) if self.half_precision else image_tensor_ac.unsqueeze(0).to(self.device)
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        del image_batch
        torch.cuda.empty_cache()

        return tokens.cpu().numpy(), grid_size    

class CAPI:
    def __init__(self, repo_name='facebookresearch/capi:main', model_name='capi_vitl14_p205', half_precision=False, device="cuda"):
        self.model_name = model_name
        self.half_precision = half_precision
        self.device = device

        # Load the model from torch.hub
        self.model = torch.hub.load('facebookresearch/capi:main', model_name)
        self.model = self.model.half().to(self.device) if self.half_precision else self.model.to(self.device)
        self.model.eval()

        # Standard transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224)),  # Ensure input size
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def get_features(self, rgb_image_numpy):
        # Preprocess image
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        image_tensor = image_tensor.half().to(self.device) if self.half_precision else image_tensor.to(self.device)

        with torch.inference_mode():
            global_repr, registers, feature_map = self.model(image_tensor)
            torch.cuda.empty_cache()

        # return {
        #     'global_repr': global_repr.cpu().numpy(),
        #     'registers': registers.cpu().numpy(),
        #     'feature_map': feature_map.cpu().numpy()
        # }

        # feature_map shape: (1, H, W, C) -> (H, W, C)
        feature_map = feature_map.squeeze(0)
        H, W, C = feature_map.shape
        patch_embeddings = feature_map.reshape(H * W, C)  # Reshape to (H*W, C)

        return patch_embeddings.cpu().numpy(), (H, W)
    
class RADIO:
    def __init__(self, model_version='radio_v2.5-h', device='cuda', half_precision=False):
        self.model_version = model_version
        self.device = device
        self.half_precision = half_precision

        # Load the model from torch.hub
        self.model = torch.hub.load(
            'NVlabs/RADIO', 
            'radio_model', 
            version=model_version, 
            progress=True, 
            skip_validation=True
        )
        self.model = self.model.to(self.device).eval()

        # Load the external preprocessor if manual normalization is preferred
        self.conditioner = self.model.make_preprocessor_external()

    def preprocess_image(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        x = pil_to_tensor(image).to(dtype=torch.float32, device=self.device)
        x.div_(255.0)  # Normalize to [0, 1]
        x = x.unsqueeze(0)  # Add batch dimension

        # Adjust the resolution to the nearest supported resolution
        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        return F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

    def get_features(self, rgb_image_numpy):
        x = self.preprocess_image(rgb_image_numpy)

        # If using E-RADIO, set the optimal window size
        if 'e-radio' in self.model_version:
            self.model.model.set_optimal_window_size(x.shape[2:])

        # Use mixed precision if enabled
        if self.half_precision:
            with torch.autocast(self.device, dtype=torch.bfloat16):
                cond_x = self.conditioner(x)
                summary, spatial_features = self.model(cond_x, feature_fmt='NCHW')
        else:
            cond_x = self.conditioner(x)
            summary, spatial_features = self.model(cond_x, feature_fmt='NCHW')

        assert spatial_features.ndim == 4  # Should be (B, C, H, W)
        # Remove batch dimension
        spatial_features = spatial_features.squeeze(0)  # Now (C, H, W)
        grid_size = spatial_features.shape[1], spatial_features.shape[2]  # (H, W)
        # Rearrange to (H*W, C)
        spatial_features = spatial_features.permute(1, 2, 0)  # (H, W, C)
        spatial_features = spatial_features.reshape(-1, spatial_features.shape[-1])  # (H*W, C)

        # Release memory
        del x, cond_x
        torch.cuda.empty_cache()

        return spatial_features.detach().cpu().numpy(), grid_size