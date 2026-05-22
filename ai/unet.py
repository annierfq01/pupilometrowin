"""
unet.py - U-Net Module
======================
U-Net ligera para segmentacion de pupila/iris.
Incluye funciones para entrenar y usar el modelo.

Requiere: torch (pip install torch torchvision)
"""

import cv2
import numpy as np
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _ConvBlock(nn.Module):
        """Bloque convolucional basico para U-Net."""
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        
        def forward(self, x):
            return self.block(x)

    class SimpleUNet(nn.Module):
        """
        U-Net de 3 niveles, entrada 1 canal gris 128x128.
        Salida: mascara binaria (pupila=1, fondo=0).
        ~180K parametros - rapido de entrenar y de inferir.
        """
        def __init__(self):
            super().__init__()
            self.enc1 = _ConvBlock(1,  16)
            self.enc2 = _ConvBlock(16, 32)
            self.enc3 = _ConvBlock(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.up2  = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = _ConvBlock(64, 32)
            self.up1  = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec1 = _ConvBlock(32, 16)
            self.out  = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return torch.sigmoid(self.out(d1))

    class EyeDataset(Dataset):
        """Dataset de imagenes de ojo + mascaras binarias."""
        def __init__(self, img_dir, mask_dir, size=128):
            self.imgs  = sorted([f for f in os.listdir(img_dir)
                                  if f.endswith('.png')])
            self.img_dir  = img_dir
            self.mask_dir = mask_dir
            self.size     = size

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            fname = self.imgs[idx]
            img  = cv2.imread(os.path.join(self.img_dir,  fname))
            mask = cv2.imread(os.path.join(self.mask_dir, fname), 0)

            img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img  = cv2.resize(img,  (self.size, self.size)) / 255.0
            mask = cv2.resize(mask, (self.size, self.size)) / 255.0

            img  = torch.tensor(img,  dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return img, mask


def is_available() -> bool:
    """Verifica si PyTorch esta disponible."""
    return TORCH_AVAILABLE


def create_model():
    """Crea una nueva instancia del modelo U-Net."""
    if not TORCH_AVAILABLE:
        return None
    return SimpleUNet()


def load_model(path: str):
    """Carga un modelo guardado. Devuelve None si falla."""
    if not TORCH_AVAILABLE:
        return None
    try:
        model = SimpleUNet()
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model
    except Exception:
        return None


def segment(model, image: np.ndarray) -> np.ndarray:
    """
    Infiere la mascara de pupila para un frame.
    Devuelve mascara binaria en resolucion original (uint8, 0/255).
    """
    if model is None or not TORCH_AVAILABLE:
        return None
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape
    resized = cv2.resize(gray, (128, 128)) / 255.0
    tensor  = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor).squeeze().numpy()
    mask = (pred > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (w0, h0))


def extract_pupil_from_mask(mask: np.ndarray) -> tuple:
    """
    Extrae centro y radio de la pupila de una mascara binaria.
    Devuelve (center, radius) o (None, None).
    """
    if mask is None:
        return None, None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return (int(cx), int(cy)), int(r)


def train_model(img_dir: str, mask_dir: str,
                epochs: int = 10,
                progress_callback=None,
                save_path: str = 'unet_eye.pth') -> object:
    """
    Entrena el modelo con el dataset en img_dir / mask_dir.
    
    Args:
        img_dir: Directorio de imagenes
        mask_dir: Directorio de mascaras
        epochs: Numero de epochs
        progress_callback: Funcion callback(epoch, total, loss)
        save_path: Ruta para guardar el modelo
        
    Returns:
        Modelo entrenado o None si falla
    """
    if not TORCH_AVAILABLE:
        return None

    dataset = EyeDataset(img_dir, mask_dir)
    if len(dataset) == 0:
        return None

    loader  = DataLoader(dataset, batch_size=8, shuffle=True)
    model   = SimpleUNet()
    opt     = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, masks in loader:
            opt.zero_grad()
            preds = model(imgs)
            loss  = loss_fn(preds, masks)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / max(len(loader), 1)
        if progress_callback:
            progress_callback(epoch + 1, epochs, avg)

    model.eval()
    torch.save(model.state_dict(), save_path)
    return model
