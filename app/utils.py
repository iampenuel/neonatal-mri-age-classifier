
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

IMG_SIZE = 224

def load_config(config_path: Path):
    with open(config_path, "r") as f:
        return json.load(f)

def make_efficientnet_b0(num_classes: int):
    # IMPORTANT: weights=None because we load your trained checkpoint
    model = torchvision.models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def load_model(checkpoint_path: Path, num_classes: int):
    device = torch.device("cpu")  # Streamlit Community Cloud is CPU
    model = make_efficientnet_b0(num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                       # if image is grayscale -> [1,H,W]
    transforms.Lambda(lambda x: x.repeat(3,1,1)), # -> [3,H,W]
    transforms.Normalize(mean=[0.5]*3, std=[0.25]*3),
])

def preprocess_pil(img_pil: Image.Image) -> torch.Tensor:
    img = img_pil.convert("L")  # force grayscale like training
    x = _preprocess(img).unsqueeze(0)  # [1,3,224,224]
    return x

@torch.no_grad()
def predict_probs(model, x: torch.Tensor) -> np.ndarray:
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx: int):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def effnet_target_layer(model):
    return model.features[-1]

def tensor_to_display_gray(x: torch.Tensor) -> np.ndarray:
    # inverse normalize (approx) for visualization
    img = x[0].detach().cpu()
    img = (img * 0.25) + 0.5
    img = img.clamp(0, 1)
    return img[0].numpy()
