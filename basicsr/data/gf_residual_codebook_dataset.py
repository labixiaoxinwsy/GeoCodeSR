from pathlib import Path

try:
    import rasterio
except ImportError as exc:
    raise ImportError('Please install rasterio to use GFResidualCodebookDataset.') from exc

import numpy as np
import torch
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY


def list_tif_files(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}])


def read_residual_tif(path, bands):
    with rasterio.open(path) as src:
        if src.count < bands:
            raise ValueError(f'{path} has {src.count} bands, but {bands} bands were requested.')
        data = src.read(list(range(1, bands + 1)))
    residual = np.transpose(data, (1, 2, 0)).astype(np.float32)
    if not np.isfinite(residual).all():
        raise ValueError(f'Residual has NaN/Inf values: {path}')
    return residual


def image_to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


@DATASET_REGISTRY.register()
class GFResidualCodebookDataset(data.Dataset):
    """Residual-only GF GeoTIFF dataset for VQ codebook pretraining."""

    def __init__(self, opt):
        super(GFResidualCodebookDataset, self).__init__()
        self.opt = opt
        self.residual_folder = Path(opt['dataroot_residual'])
        self.bands = opt.get('bands', 3)
        self.residual_scale = opt.get('residual_scale', 20.0)
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        self.paths = list_tif_files(self.residual_folder)

        print(f'[GFResidualCodebookDataset] residual files: {len(self.paths)}')
        print(f'[GFResidualCodebookDataset] bands: {self.bands}')
        print(f'[GFResidualCodebookDataset] residual_scale: {self.residual_scale}')
        if self.paths:
            print(f'[GFResidualCodebookDataset] example: {self.paths[0]}')

    def __getitem__(self, index):
        residual_path = self.paths[index]
        residual = read_residual_tif(residual_path, self.bands)
        residual_tensor = image_to_tensor(residual)
        residual_scaled = residual_tensor * float(self.residual_scale)
        return {
            'residual': residual_tensor,
            'residual_scaled': residual_scaled,
            'residual_path': str(residual_path),
        }

    def __len__(self):
        return len(self.paths)
