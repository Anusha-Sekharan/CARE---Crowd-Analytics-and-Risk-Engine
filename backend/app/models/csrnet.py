import os
import requests
import torch
import torch.nn as nn

# URL to download pre-trained CSRNet weights from Hugging Face
WEIGHTS_URL = "https://huggingface.co/muasifk/CSRNet/resolve/main/CSRNet.pth"
WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "CSRNet.pth"))

def download_weights(dest_path: str = WEIGHTS_PATH):
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading pre-trained CSRNet weights to {dest_path}...")
        response = requests.get(WEIGHTS_URL, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    else:
        print(f"CSRNet weights found at {dest_path}.")

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_csrnet_model(weights_path: str = WEIGHTS_PATH, device: str = "cpu") -> CSRNet:
    """Helper function to load the pre-trained CSRNet model."""
    download_weights(weights_path)
    model = CSRNet(load_weights=True)
    
    # Load model weights safely
    # If using PyTorch 2.6+, we can keep weights_only=False since it is our own trusted model weight file
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
