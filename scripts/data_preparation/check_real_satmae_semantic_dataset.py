import argparse
import importlib.util
import random
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_class(file_path, class_name):
    spec = importlib.util.spec_from_file_location(Path(file_path).stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


GFPairedSemanticDataset = load_class(ROOT_DIR / 'basicsr' / 'data' / 'gf_paired_semantic_dataset.py', 'GFPairedSemanticDataset')


def parse_args():
    parser = argparse.ArgumentParser(description='Check real SatMAE semantic token dataset.')
    parser.add_argument('--dataset_root', default='E:/GeoCodeSR/tuihua_paired_x4')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--bands', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def list_files(directory, suffixes):
    directory = Path(directory)
    if not directory.exists():
        return []
    suffixes = {s.lower() for s in suffixes}
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes])


def load_token(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        obj = obj.item()
    token_type = obj.get('token_type', 'dict') if isinstance(obj, dict) else 'array'
    token = obj['token'] if isinstance(obj, dict) else obj
    token = np.asarray(token, dtype=np.float32)
    return token, token_type


def token_dim(token):
    if token.ndim == 1:
        return token.shape[0]
    if token.ndim in [2, 3]:
        return token.shape[-1]
    if token.ndim == 4:
        return token.shape[0]
    return None


def describe_tokens(token_files, num_samples):
    if not token_files:
        print('[WARN] No token files found.')
        return False
    sample_files = random.sample(token_files, min(num_samples, len(token_files)))
    ok = True
    for path in sample_files:
        try:
            token, token_type = load_token(path)
            finite = np.isfinite(token)
            nan_inf = int(token.size - finite.sum())
            values = token[finite] if finite.any() else np.array([np.nan], dtype=np.float32)
            dim = token_dim(token)
            if dim != 1024:
                print(f'[WARN] token_dim is not 1024: {path.name}, dim={dim}, shape={token.shape}')
            if nan_inf > 0:
                ok = False
            print(
                f'  {path.name}: shape={token.shape}, dtype={token.dtype}, token_type={token_type}, '
                f'token_dim={dim}, min={float(np.nanmin(values)):.6f}, max={float(np.nanmax(values)):.6f}, '
                f'mean={float(np.nanmean(values)):.6f}, std={float(np.nanstd(values)):.6f}, nan_inf={nan_inf}')
        except Exception as exc:
            ok = False
            print(f'  [ERROR] Failed to read token {path}: {exc}')
    return ok


def check_split(dataset_root, split, args):
    split_root = Path(dataset_root) / split
    hr_dir = split_root / 'HR'
    lr_dir = split_root / f'LR_x{args.scale}'
    residual_dir = split_root / f'Residual_x{args.scale}'
    token_dir = split_root / 'SatMAE_tokens_mean'
    hr = list_files(hr_dir, {'.tif', '.tiff'})
    lr = list_files(lr_dir, {'.tif', '.tiff'})
    residual = list_files(residual_dir, {'.tif', '.tiff'})
    tokens = list_files(token_dir, {'.npy'})
    hr_names = {p.stem for p in hr}
    matched = hr_names & {p.stem for p in lr} & {p.stem for p in residual} & {p.stem for p in tokens}

    print(f'\n[{split}]')
    print(f'HR files: {len(hr)}')
    print(f'LR files: {len(lr)}')
    print(f'Residual files: {len(residual)}')
    print(f'SatMAE token files: {len(tokens)}')
    print(f'Matched samples: {len(matched)}')
    ok = len(matched) > 0 and len(hr) == len(lr) == len(residual) == len(tokens) == len(matched)
    if not ok:
        print('[WARN] File counts or matched sample counts are not fully aligned.')
    ok = describe_tokens(tokens, args.num_samples) and ok

    opt = {
        'dataroot_gt': str(hr_dir),
        'dataroot_lq': str(lr_dir),
        'dataroot_residual': str(residual_dir),
        'dataroot_semantic': str(token_dir),
        'scale': args.scale,
        'bands': args.bands,
        'use_residual': True,
        'normalize_mode': 'auto',
    }
    try:
        dataset = GFPairedSemanticDataset(opt)
        print(f'Dataset length: {len(dataset)}')
        for idx in range(min(args.num_samples, len(dataset))):
            sample = dataset[idx]
            print(
                f'  sample {idx}: lq={tuple(sample["lq"].shape)}, gt={tuple(sample["gt"].shape)}, '
                f'residual={tuple(sample["residual"].shape)}, semantic_token={tuple(sample["semantic_token"].shape)}')
    except Exception as exc:
        ok = False
        print(f'[ERROR] Failed to instantiate/read GFPairedSemanticDataset: {exc}')
    return ok


def main():
    args = parse_args()
    random.seed(args.seed)
    all_ok = True
    print(f'Dataset root: {args.dataset_root}')
    for split in ['train', 'val', 'test']:
        all_ok = check_split(args.dataset_root, split, args) and all_ok
    if all_ok:
        print('\nReal SatMAE semantic dataset check passed.')
    else:
        print('\nReal SatMAE semantic dataset check finished with warnings/errors.')


if __name__ == '__main__':
    main()
