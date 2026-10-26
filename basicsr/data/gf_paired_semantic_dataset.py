from pathlib import Path

try:
    import rasterio
except ImportError as exc:
    raise ImportError('Please install rasterio to use GFPairedSemanticDataset.') from exc

import numpy as np
import torch
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY


def list_tif_files(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}])


def list_npy_files(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == '.npy'])


def normalize_image(image, normalize_mode='auto', clip=True, source_dtypes=None):
    image = image.astype(np.float32)
    if normalize_mode == 'auto':
        finite_mask = np.isfinite(image)
        if finite_mask.any() and np.nanmax(image[finite_mask]) > 1.0:
            is_uint8 = source_dtypes and all(np.dtype(dtype) == np.dtype('uint8') for dtype in source_dtypes)
            image = image / (255.0 if is_uint8 else 10000.0)
    elif normalize_mode in ['none', None]:
        pass
    else:
        raise ValueError(f'Unsupported normalize_mode: {normalize_mode}')

    if clip:
        image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def read_multiband_tif(path, bands, normalize_mode='auto', clip=True):
    path = Path(path)
    with rasterio.open(path) as src:
        if src.count < bands:
            raise ValueError(f'{path} has {src.count} bands, but {bands} bands were requested.')
        data = src.read(list(range(1, bands + 1)))
        source_dtypes = src.dtypes[:bands]
    image = np.transpose(data, (1, 2, 0))
    return normalize_image(image, normalize_mode=normalize_mode, clip=clip, source_dtypes=source_dtypes)


def image_to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def load_semantic_token(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        obj = obj.item()
    if isinstance(obj, dict):
        if 'token' not in obj:
            raise KeyError(f'Semantic token dict has no "token" key: {path}')
        token = obj['token']
    else:
        token = obj
    token = np.asarray(token, dtype=np.float32)
    if not np.isfinite(token).all():
        raise ValueError(f'NaN/Inf found in semantic token: {path}')
    return torch.from_numpy(np.ascontiguousarray(token)).float()


@DATASET_REGISTRY.register()
class GFPairedSemanticDataset(data.Dataset):
    """Paired multi-band GeoTIFF dataset with offline semantic token cache."""

    def __init__(self, opt):
        super(GFPairedSemanticDataset, self).__init__()
        self.opt = opt
        self.gt_folder = Path(opt['dataroot_gt'])
        self.lq_folder = Path(opt['dataroot_lq'])
        self.residual_folder = Path(opt.get('dataroot_residual', '')) if opt.get('dataroot_residual') else None
        self.semantic_folder = Path(opt['dataroot_semantic'])
        self.scale = opt.get('scale', 4)
        self.bands = opt.get('bands', 3)
        self.use_residual = opt.get('use_residual', False)
        self.normalize_mode = opt.get('normalize_mode', 'auto')
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        self.gt_paths = list_tif_files(self.gt_folder)
        self.lq_paths = list_tif_files(self.lq_folder)
        self.residual_paths = list_tif_files(self.residual_folder) if self.residual_folder is not None else []
        self.semantic_paths = list_npy_files(self.semantic_folder)

        self.lq_map = {p.name: p for p in self.lq_paths}
        self.residual_map = {p.name: p for p in self.residual_paths}
        self.semantic_map = {p.name: p for p in self.semantic_paths}
        self.paths = self._match_paths()

        print(f'[GFPairedSemanticDataset] GT files: {len(self.gt_paths)}')
        print(f'[GFPairedSemanticDataset] LQ files: {len(self.lq_paths)}')
        print(f'[GFPairedSemanticDataset] Residual files: {len(self.residual_paths)}')
        print(f'[GFPairedSemanticDataset] Semantic token files: {len(self.semantic_paths)}')
        print(f'[GFPairedSemanticDataset] Matched samples: {len(self.paths)}')
        print(f'[GFPairedSemanticDataset] use_residual: {self.use_residual}')
        print(f'[GFPairedSemanticDataset] scale: {self.scale}')
        print(f'[GFPairedSemanticDataset] bands: {self.bands}')
        self._print_example_token_shape()

    def _format_lq_name(self, gt_name):
        stem = Path(gt_name).stem
        suffix = Path(gt_name).suffix
        return f'{self.filename_tmpl.format(stem)}{suffix}'

    def _match_paths(self):
        paths = []
        for gt_path in self.gt_paths:
            lq_path = self.lq_map.get(self._format_lq_name(gt_path.name))
            if lq_path is None:
                print(f'[GFPairedSemanticDataset][WARN] Missing LQ for GT: {gt_path}')
                continue

            semantic_path = self.semantic_map.get(f'{gt_path.stem}.npy')
            if semantic_path is None:
                print(f'[GFPairedSemanticDataset][WARN] Missing semantic token for GT: {gt_path}')
                continue

            item = {
                'gt_path': gt_path,
                'lq_path': lq_path,
                'semantic_path': semantic_path,
            }
            if self.use_residual:
                if self.residual_folder is None:
                    print(f'[GFPairedSemanticDataset][WARN] dataroot_residual is not set, skip: {gt_path}')
                    continue
                residual_path = self.residual_map.get(gt_path.name)
                if residual_path is None:
                    print(f'[GFPairedSemanticDataset][WARN] Missing Residual for GT: {gt_path}')
                    continue
                item['residual_path'] = residual_path
            paths.append(item)
        return paths

    def _print_example_token_shape(self):
        if not self.paths:
            print('[GFPairedSemanticDataset] Semantic token example shape: N/A')
            return
        try:
            token = load_semantic_token(self.paths[0]['semantic_path'])
            print(f'[GFPairedSemanticDataset] Semantic token example shape: {tuple(token.shape)}')
        except Exception as exc:
            print(f'[GFPairedSemanticDataset][WARN] Failed to read example semantic token: {exc}')

    def _check_shapes(self, img_gt, img_lq, img_residual, paths):
        if img_gt.shape[2] != self.bands:
            raise ValueError(f'GT channel mismatch: {paths["gt_path"]}, shape={img_gt.shape}, expected bands={self.bands}')
        if img_lq.shape[2] != self.bands:
            raise ValueError(f'LQ channel mismatch: {paths["lq_path"]}, shape={img_lq.shape}, expected bands={self.bands}')

        gt_h, gt_w = img_gt.shape[:2]
        lq_h, lq_w = img_lq.shape[:2]
        if gt_h != lq_h * self.scale or gt_w != lq_w * self.scale:
            raise ValueError(
                f'GT/LQ spatial mismatch: gt={paths["gt_path"]}, lq={paths["lq_path"]}, '
                f'gt_shape={img_gt.shape}, lq_shape={img_lq.shape}, scale={self.scale}')

        if self.use_residual:
            if img_residual is None:
                raise ValueError(f'Residual is missing for {paths["gt_path"]}.')
            if img_residual.shape != img_gt.shape:
                raise ValueError(
                    f'Residual/GT shape mismatch: residual={paths["residual_path"]}, gt={paths["gt_path"]}, '
                    f'residual_shape={img_residual.shape}, gt_shape={img_gt.shape}')

    def __getitem__(self, index):
        paths = self.paths[index]
        img_gt = read_multiband_tif(paths['gt_path'], self.bands, self.normalize_mode, clip=True)
        img_lq = read_multiband_tif(paths['lq_path'], self.bands, self.normalize_mode, clip=True)
        img_residual = None
        if self.use_residual:
            img_residual = read_multiband_tif(paths['residual_path'], self.bands, normalize_mode='none', clip=False)

        self._check_shapes(img_gt, img_lq, img_residual, paths)
        semantic_token = load_semantic_token(paths['semantic_path'])

        data_dict = {
            'lq': image_to_tensor(img_lq),
            'gt': image_to_tensor(img_gt),
            'semantic_token': semantic_token,
            'semantic_path': str(paths['semantic_path']),
            'lq_path': str(paths['lq_path']),
            'gt_path': str(paths['gt_path']),
        }
        if self.use_residual:
            data_dict['residual'] = image_to_tensor(img_residual)
            data_dict['residual_path'] = str(paths['residual_path'])
        return data_dict

    def __len__(self):
        return len(self.paths)
