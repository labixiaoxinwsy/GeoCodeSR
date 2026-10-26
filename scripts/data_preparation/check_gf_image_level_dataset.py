import argparse
import json
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise ImportError('Please install opencv-python before running this script.') from exc

try:
    import numpy as np
except ImportError as exc:
    raise ImportError('Please install numpy before running this script.') from exc

try:
    import rasterio
except ImportError as exc:
    raise ImportError('Please install rasterio before running this script.') from exc


SPLITS = ['train', 'val', 'test']


def parse_args():
    parser = argparse.ArgumentParser(description='Check GF image-level split LR/HR/Residual dataset.')
    parser.add_argument('--dataset_root', default='E:/GeoCodeSR/tuihua_paired_x4')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--bands', type=int, default=3)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--fft_energy', action='store_true')
    return parser.parse_args()


def list_tifs(path):
    path = Path(path)
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}])


def read_tif(path, bands, clip=True):
    with rasterio.open(path) as src:
        data = src.read(list(range(1, bands + 1)))
    image = np.transpose(data, (1, 2, 0)).astype(np.float32)
    finite = np.isfinite(image)
    if clip and finite.any() and np.nanmax(image[finite]) > 1.0:
        image = image / 10000.0
    return np.clip(image, 0.0, 1.0).astype(np.float32) if clip else image.astype(np.float32)


def source_id_from_patch(filename):
    stem = Path(filename).stem
    marker = '_r'
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem


def ps_freq_ratio(residual):
    ratios = []
    h, w, c = residual.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = float(radius.max()) + 1e-8
    low = radius < 0.15 * max_radius
    mid = (radius >= 0.15 * max_radius) & (radius < 0.4 * max_radius)
    high = radius >= 0.4 * max_radius
    for ch in range(c):
        spec = np.fft.fftshift(np.fft.fft2(residual[:, :, ch]))
        mag = np.abs(spec)
        total = float(mag.sum()) + 1e-12
        ratios.append([float(mag[low].sum() / total), float(mag[mid].sum() / total), float(mag[high].sum() / total)])
    return np.mean(np.asarray(ratios), axis=0).tolist()


def resize_hwc(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def check_split(root, split, args):
    hr_dir = root / split / 'HR'
    lr_dir = root / split / f'LR_x{args.scale}'
    residual_dir = root / split / f'Residual_x{args.scale}'
    hr_paths, lr_paths, residual_paths = list_tifs(hr_dir), list_tifs(lr_dir), list_tifs(residual_dir)
    names = {'hr': {p.name for p in hr_paths}, 'lr': {p.name for p in lr_paths}, 'residual': {p.name for p in residual_paths}}
    matched = sorted(names['hr'] & names['lr'] & names['residual'])
    source_ids = {source_id_from_patch(name) for name in matched}
    errors, sample_reports = [], []
    residual_abs_mean = []
    residual_stats = []
    fft_ratios = []

    if names['hr'] != names['lr'] or names['hr'] != names['residual']:
        errors.append('HR/LR/Residual file names do not fully match.')

    for name in matched[:args.num_samples]:
        hr_path, lr_path, residual_path = hr_dir / name, lr_dir / name, residual_dir / name
        try:
            hr = read_tif(hr_path, args.bands, clip=True)
            lr = read_tif(lr_path, args.bands, clip=True)
            residual = read_tif(residual_path, args.bands, clip=False)
            expected_hr_shape = (args.patch_size, args.patch_size, args.bands)
            expected_lr_shape = (args.patch_size // args.scale, args.patch_size // args.scale, args.bands)
            if hr.shape != expected_hr_shape:
                errors.append(f'{name}: HR shape {hr.shape} != {expected_hr_shape}')
            if lr.shape != expected_lr_shape:
                errors.append(f'{name}: LR shape {lr.shape} != {expected_lr_shape}')
            if residual.shape != expected_hr_shape:
                errors.append(f'{name}: Residual shape {residual.shape} != {expected_hr_shape}')
            lr_up = resize_hwc(lr, hr.shape[1], hr.shape[0])
            mae = float(np.mean(np.abs((hr - lr_up) - residual)))
            if mae > 1e-5:
                errors.append(f'{name}: residual formula MAE too high: {mae:.8f}')
            zero_ratio = float((np.abs(hr).sum(axis=2) <= 0).mean())
            rstats = {
                'min': float(np.min(residual)),
                'max': float(np.max(residual)),
                'mean': float(np.mean(residual)),
                'std': float(np.std(residual)),
                'abs_mean': float(np.mean(np.abs(residual))),
            }
            residual_abs_mean.append(rstats['abs_mean'])
            residual_stats.append(rstats)
            if args.fft_energy:
                fft_ratios.append(ps_freq_ratio(residual))
            sample_reports.append({
                'filename': name,
                'hr_shape': hr.shape,
                'lr_shape': lr.shape,
                'residual_shape': residual.shape,
                'residual_formula_mae': mae,
                'zero_ratio': zero_ratio,
                'residual_stats': rstats,
            })
        except Exception as exc:
            errors.append(f'{name}: read/check failed: {exc}')

    result = {
        'split': split,
        'hr_count': len(hr_paths),
        'lr_count': len(lr_paths),
        'residual_count': len(residual_paths),
        'matched_count': len(matched),
        'file_names_match': names['hr'] == names['lr'] == names['residual'],
        'source_id_count': len(source_ids),
        'sample_count': len(sample_reports),
        'sample_reports': sample_reports,
        'residual_abs_mean_mean': float(np.mean(residual_abs_mean)) if residual_abs_mean else None,
        'residual_stats_mean': {
            key: float(np.mean([s[key] for s in residual_stats])) for key in ['min', 'max', 'mean', 'std', 'abs_mean']
        } if residual_stats else None,
        'fft_low_mid_high_mean': np.mean(np.asarray(fft_ratios), axis=0).tolist() if fft_ratios else None,
        'errors': errors,
        'passed': len(errors) == 0,
    }
    return result, source_ids


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    if not root.exists():
        raise FileNotFoundError(f'Dataset root does not exist: {root}')
    report = {'dataset_root': str(root), 'scale': args.scale, 'bands': args.bands, 'splits': {}, 'source_overlap': {}}
    source_sets = {}
    serious_errors = []
    for split in SPLITS:
        result, source_ids = check_split(root, split, args)
        report['splits'][split] = result
        source_sets[split] = source_ids
        serious_errors.extend([f'{split}: {err}' for err in result['errors']])
        print(f"{split}: HR={result['hr_count']}, LR={result['lr_count']}, Residual={result['residual_count']}, "
              f"matched={result['matched_count']}, sources={result['source_id_count']}, passed={result['passed']}")
        if result['residual_stats_mean']:
            print(f"  residual abs mean={result['residual_stats_mean']['abs_mean']:.6f}, "
                  f"min={result['residual_stats_mean']['min']:.6f}, max={result['residual_stats_mean']['max']:.6f}")

    overlap_errors = []
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        overlap = sorted(source_sets[a] & source_sets[b])
        report['source_overlap'][f'{a}_{b}'] = {
            'left_count': len(source_sets[a]),
            'right_count': len(source_sets[b]),
            'overlap_count': len(overlap),
            'overlap_examples': overlap[:20],
        }
        if overlap:
            overlap_errors.append(f'{a}/{b} overlap: {len(overlap)}')
    if overlap_errors:
        print('ERROR: source leakage detected.')
        serious_errors.extend(overlap_errors)
    else:
        print('No source leakage detected.')

    report['passed'] = len(serious_errors) == 0
    report['errors'] = serious_errors
    manifest_dir = root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifest_dir / 'dataset_check_report.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'Report saved: {out_path}')
    if not report['passed']:
        raise SystemExit('Dataset check failed. See dataset_check_report.json')


if __name__ == '__main__':
    main()
