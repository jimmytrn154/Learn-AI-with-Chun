# matcher.py
import torch, torchvision
import numpy as np
import torchvision.transforms as T

class FrozenEmbedder:
    def __init__(self, device="cuda", backbone_ckpt=None):
        self.device = device if torch.cuda.is_available() else "cpu"
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        m.eval().to(self.device)
        for p in m.parameters(): p.requires_grad_(False)
        if backbone_ckpt:
            sd = torch.load(backbone_ckpt, map_location=self.device)
            m.load_state_dict(sd, strict=False)
        self.backbone = m
        self.prep = T.Compose([
            T.ToPILImage(),
            T.Resize((224,224), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    @torch.no_grad()
    def encode_np(self, crops, device=None):
        if len(crops) == 0:
            return np.zeros((0,512), dtype=np.float32)
        dev = device or self.device
        batch = torch.stack([self.prep(c) for c in crops]).to(dev, non_blocking=True)
        feats = self.backbone(batch)           # (N,512)
        return feats.detach().float().cpu().numpy()
