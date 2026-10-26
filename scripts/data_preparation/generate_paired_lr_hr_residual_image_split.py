import argparse
import csv
import json
import random
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
    from rasterio.windows import Window, transform as window_transform
except ImportError as exc:
    raise ImportError('Please install rasterio before running this script.') from exc


SPLITS = ['train', 'val', 'test']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate HR/LR/Residual patches from paired HR and real LR GeoTIFF images.')
    parser.add_argument('--hr_dir', default='D:/code/tuihua/HR_0.5m', help='Directory containing HR GeoTIFF images.')
    parser.add_argument('--lr_dir', default='D:/code/tuihua/LR_2m', help='Directory containing paired LR GeoTIFF images.')
    parser.add_argument('--output_dir', default='E:/GeoCodeSR/tuihua_paired_x4', help='Output dataset root.')
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale factor.')
    parser.add_argument('--patch_size', type=int, default=256, help='HR patch size.')
    parser.add_argument('--stride', type=int, default=256, help='HR sliding-window stride.')
    parser.add_argument('--bands', type=int, default=3, help='Number of leading bands to read.')
    parser.add_argument('--match_mode', default='name', choices=['name', 'stem'], help='How to match HR/LR pairs.')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum matched image pairs to process.')
    parser.add_argument('--shuffle_images', action='store_true', help='Shuffle image pairs before max_images/split.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Image-level train ratio.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Image-level val ratio.')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Image-level test ratio.')
    parser.add_argument('--max_patches_per_image', type=int, default=None, help='Limit patches per image pair.')
    parser.add_argument('--min_valid_ratio', type=float, default=0.9, help='Minimum HR valid-pixel ratio.')
    parser.add_argument('--nodata_value', type=float, default=0.0, help='Nodata value used by valid-pixel filtering.')
    parser.add_argument(
        '--normalize_mode',
        default='auto',
        choices=['auto', 'uint8', 'reflectance', 'none'],
        help='auto: uint8 -> /255, other values >1 -> /10000; reflectance: /10000; uint8: /255.')
    parser.add_argument('--resume', action='store_true', help='Skip patches whose outputs already exist.')
    parser.add_argument('--manifest', action='store_true', help='Save patch/image manifests.')
    parser.add_argument('--summary_json', action='store_true', help='Save generation summary json.')
    parser.add_argument('--preview', action='store_true', help='Save RGB preview PNG files.')
    return parser.parse_args()


def list_geotiffs(root):
    root = Path(root)
    if not root.exists():
        return []
    excluded_suffixes = {'.aux.xml', '.ovr', '.tfw', '.xml', '.json'}
    paths = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if any(lower_name.endswith(suffix) for suffix in excluded_suffixes):
            continue
        if path.suffix.lower() in {'.tif', '.tiff'}:
            paths.append(path)
    return sorted(paths)


def pair_key(path, match_mode):
    return path.name.lower() if match_mode == 'name' else path.stem.lower()


def build_pairs(hr_dir, lr_dir, match_mode):
    hr_paths = list_geotiffs(hr_dir)
    lr_paths = list_geotiffs(lr_dir)
    lr_map = {}
    duplicate_lr = []
    for lr_path in lr_paths:
        key = pair_key(lr_path, match_mode)
        if key in lr_map:
            duplicate_lr.append(str(lr_path))
            continue
        lr_map[key] = lr_path

    pairs = []
    missing_lr = []
    duplicate_hr_keys = set()
    seen_hr_keys = set()
    for hr_path in hr_paths:
        key = pair_key(hr_path, match_mode)
        if key in seen_hr_keys:
            duplicate_hr_keys.add(key)
            continue
        seen_hr_keys.add(key)
        lr_path = lr_map.get(key)
        if lr_path is None:
            missing_lr.append(str(hr_path))
            continue
        pairs.append({'hr': hr_path, 'lr': lr_path, 'source_id': hr_path.stem})

    matched_lr_keys = {pair_key(pair['lr'], match_mode) for pair in pairs}
    extra_lr = [str(path) for path in lr_paths if pair_key(path, match_mode) not in matched_lr_keys]
    return pairs, hr_paths, lr_paths, missing_lr, extra_lr, duplicate_lr, sorted(duplicate_hr_keys)


def split_pairs(pairs, args):
    pairs = pairs[:]
    if args.shuffle_images:
        random.Random(args.seed).shuffle(pairs)
    if args.max_images is not None:
        pairs = pairs[:args.max_images]
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f'train_ratio + val_ratio + test_ratio must be 1.0, got {ratio_sum}')
    total = len(pairs)
    train_count = int(total * args.train_ratio)
    val_count = int(total * args.val_ratio)
    return {
        'train': pairs[:train_count],
        'val': pairs[train_count:train_count + val_count],
        'test': pairs[train_count + val_count:],
    }


def prepare_dirs(output_dir, scale, preview=False):
    root = Path(output_dir)
    for split in SPLITS:
        for sub in ['HR', f'LR_x{scale}', f'Residual_x{scale}']:
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        if preview:
            for sub in ['HR_preview', f'LR_x{scale}_preview', f'Residual_x{scale}_preview']:
                (root / split / sub).mkdir(parents=True, exist_ok=True)
    (root / 'manifests').mkdir(parents=True, exist_ok=True)


def normalize_to_unit(data, dtypes, mode):
    data = data.astype(np.float32)
    finite = np.isfinite(data)
    if mode == 'none':
        return data
    if mode == 'uint8':
        data = data / 255.0
    elif mode == 'reflectance':
        data = data / 10000.0
    elif finite.any() and np.nanmax(data[finite]) > 1.0:
        dtype_names = {str(dtype).lower() for dtype in dtypes}
        if any(name == 'uint8' for name in dtype_names):
            data = data / 255.0
        else:
            data = data / 10000.0
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def read_geotiff(path, bands, normalize_mode):
    with rasterio.open(path) as src:
        if src.count < bands:
            raise ValueError(f'{path} has {src.count} bands, but --bands={bands} was requested.')
        data = src.read(list(range(1, bands + 1)))
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        dtypes = src.dtypes[:bands]
        width, height = src.width, src.height
    image = np.transpose(data, (1, 2, 0))
    image = normalize_to_unit(image, dtypes, normalize_mode)
    return image, profile, transform, crs, width, height


def resize_hwc(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def make_residual(hr_patch, lr_patch):
    lr_up = resize_hwc(lr_patch, hr_patch.shape[1], hr_patch.shape[0]).astype(np.float32)
    return (hr_patch.astype(np.float32) - lr_up).astype(np.float32)


def valid_patch(hr_patch, lr_patch, min_valid_ratio, nodata_value):
    if not np.isfinite(hr_patch).all() or not np.isfinite(lr_patch).all():
        return False, 0.0, 1.0
    valid_mask = np.abs(hr_patch - float(nodata_value)).sum(axis=2) > 0
    valid_ratio = float(valid_mask.mean())
    zero_ratio = 1.0 - valid_ratio
    return valid_ratio >= min_valid_ratio, valid_ratio, zero_ratio


def update_profile(src_profile, src_transform, window, height, width, bands):
    profile = src_profile.copy()
    profile.update(
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype='float32',
        transform=window_transform(window, src_transform),
        nodata=None)
    return profile


def save_tif(path, image, profile):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.transpose(image.astype(np.float32), (2, 0, 1))
    profile = profile.copy()
    profile.update(count=out.shape[0], dtype='float32', nodata=None)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(out)


def rgb_preview(image):
    if image.shape[2] >= 3:
        rgb = image[:, :, :3]
    else:
        rgb = np.repeat(image[:, :, :1], 3, axis=2)
    rgb = rgb.astype(np.float32)
    lo, hi = np.percentile(rgb, [1, 99])
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def residual_preview(image):
    if image.shape[2] >= 3:
        rgb = image[:, :, :3]
    else:
        rgb = np.repeat(image[:, :, :1], 3, axis=2)
    lo, hi = np.percentile(rgb, [1, 99])
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def save_preview(path, image, residual=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = residual_preview(image) if residual else rgb_preview(image)
    cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def output_paths(output_dir, split, scale, filename):
    root = Path(output_dir)
    return {
        'hr': root / split / 'HR' / filename,
        'lr': root / split / f'LR_x{scale}' / filename,
        'residual': root / split / f'Residual_x{scale}' / filename,
    }


def iter_patch_positions(height, width, patch_size, stride):
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            yield row, col


def warn_alignment(pair, hr_transform, lr_transform, hr_crs, lr_crs, scale):
    warnings = []
    if hr_crs != lr_crs:
        warnings.append('CRS differs')
    hr_px = (abs(float(hr_transform.a)), abs(float(hr_transform.e)))
    lr_px = (abs(float(lr_transform.a)), abs(float(lr_transform.e)))
    if hr_px[0] > 0 and hr_px[1] > 0:
        x_ratio = lr_px[0] / hr_px[0]
        y_ratio = lr_px[1] / hr_px[1]
        if abs(x_ratio - scale) > 1e-3 or abs(y_ratio - scale) > 1e-3:
            warnings.append(f'pixel-size ratio is ({x_ratio:.4f}, {y_ratio:.4f}), expected {scale}')
    if abs(float(hr_transform.c) - float(lr_transform.c)) > 1e-3 or abs(float(hr_transform.f) - float(lr_transform.f)) > 1e-3:
        warnings.append('origin differs')
    if warnings:
        print(f"[WARN] Alignment check for {pair['hr'].name}: {', '.join(warnings)}")


def process_pair(pair, split, args):
    hr, hr_profile, hr_transform, hr_crs, hr_width, hr_height = read_geotiff(
        pair['hr'], args.bands, args.normalize_mode)
    lr, lr_profile, lr_transform, lr_crs, lr_width, lr_height = read_geotiff(
        pair['lr'], args.bands, args.normalize_mode)
    warn_alignment(pair, hr_transform, lr_transform, hr_crs, lr_crs, args.scale)

    effective_height = min(hr_height, lr_height * args.scale)
    effective_width = min(hr_width, lr_width * args.scale)
    if effective_height < args.patch_size or effective_width < args.patch_size:
        print(f"[WARN] Skip {pair['hr'].name}: common area too small, HR={hr.shape}, LR={lr.shape}")
        return [], 0, 0

    valid, skipped, rows = 0, 0, []
    lr_patch_size = args.patch_size // args.scale
    for row, col in iter_patch_positions(effective_height, effective_width, args.patch_size, args.stride):
        lr_row, lr_col = row // args.scale, col // args.scale
        hr_patch = hr[row:row + args.patch_size, col:col + args.patch_size, :]
        lr_patch = lr[lr_row:lr_row + lr_patch_size, lr_col:lr_col + lr_patch_size, :]
        if hr_patch.shape[:2] != (args.patch_size, args.patch_size) or lr_patch.shape[:2] != (lr_patch_size, lr_patch_size):
            skipped += 1
            continue
        ok, valid_ratio, zero_ratio = valid_patch(hr_patch, lr_patch, args.min_valid_ratio, args.nodata_value)
        if not ok:
            skipped += 1
            continue

        filename = f"{pair['hr'].stem}_r{row:04d}_c{col:04d}_paired.tif"
        paths = output_paths(args.output_dir, split, args.scale, filename)
        status = 'saved'
        if args.resume and all(path.exists() for path in paths.values()):
            status = 'skipped_existing'
        else:
            residual = make_residual(hr_patch, lr_patch)
            hr_window = Window(col, row, args.patch_size, args.patch_size)
            lr_window = Window(lr_col, lr_row, lr_patch_size, lr_patch_size)
            save_tif(paths['hr'], hr_patch, update_profile(
                hr_profile, hr_transform, hr_window, args.patch_size, args.patch_size, args.bands))
            save_tif(paths['lr'], lr_patch, update_profile(
                lr_profile, lr_transform, lr_window, lr_patch_size, lr_patch_size, args.bands))
            save_tif(paths['residual'], residual, update_profile(
                hr_profile, hr_transform, hr_window, args.patch_size, args.patch_size, args.bands))
            if args.preview:
                png_name = filename.replace('.tif', '.png')
                save_preview(Path(args.output_dir) / split / 'HR_preview' / png_name, hr_patch)
                save_preview(Path(args.output_dir) / split / f'LR_x{args.scale}_preview' / png_name, lr_patch)
                save_preview(Path(args.output_dir) / split / f'Residual_x{args.scale}_preview' / png_name, residual, residual=True)

        rows.append({
            'split': split,
            'source_id': pair['source_id'],
            'hr_source_image': str(pair['hr']),
            'lr_source_image': str(pair['lr']),
            'filename': filename,
            'hr_path': str(paths['hr']),
            'lr_path': str(paths['lr']),
            'residual_path': str(paths['residual']),
            'hr_row': row,
            'hr_col': col,
            'lr_row': lr_row,
            'lr_col': lr_col,
            'valid_ratio': valid_ratio,
            'zero_ratio': zero_ratio,
            'status': status,
        })
        valid += 1
        if args.max_patches_per_image is not None and valid >= args.max_patches_per_image:
            break
    return rows, valid, skipped


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_image_split_manifest(output_dir, split_map):
    rows = []
    for split, pairs in split_map.items():
        for pair in pairs:
            rows.append({
                'split': split,
                'source_id': pair['source_id'],
                'hr_source_image': str(pair['hr']),
                'lr_source_image': str(pair['lr']),
            })
    write_csv(Path(output_dir) / 'manifests' / 'image_split_manifest.csv', rows,
              ['split', 'source_id', 'hr_source_image', 'lr_source_image'])


def check_source_overlap(output_dir, split_map):
    ids = {split: {pair['source_id'] for pair in pairs} for split, pairs in split_map.items()}
    rows, has_overlap = [], False
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        overlap = sorted(ids[a] & ids[b])
        has_overlap = has_overlap or bool(overlap)
        rows.append({
            'pair': f'{a}-{b}',
            'left_count': len(ids[a]),
            'right_count': len(ids[b]),
            'overlap_count': len(overlap),
            'overlap_examples': ';'.join(overlap[:20]),
        })
    write_csv(Path(output_dir) / 'manifests' / 'source_overlap_report.csv', rows,
              ['pair', 'left_count', 'right_count', 'overlap_count', 'overlap_examples'])
    if has_overlap:
        raise RuntimeError('ERROR: source image overlap detected. See manifests/source_overlap_report.csv')
    print('Image-level split check passed. No source image overlap.')


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.patch_size % args.scale != 0:
        raise ValueError('--patch_size must be divisible by --scale.')
    if args.stride % args.scale != 0:
        raise ValueError('--stride must be divisible by --scale for paired HR/LR cropping.')

    pairs, hr_paths, lr_paths, missing_lr, extra_lr, duplicate_lr, duplicate_hr_keys = build_pairs(
        args.hr_dir, args.lr_dir, args.match_mode)
    split_map = split_pairs(pairs, args)
    prepare_dirs(args.output_dir, args.scale, preview=args.preview)
    check_source_overlap(args.output_dir, split_map)
    if args.manifest:
        write_image_split_manifest(args.output_dir, split_map)

    print(f'HR dir: {Path(args.hr_dir).resolve()}')
    print(f'LR dir: {Path(args.lr_dir).resolve()}')
    print(f'Output dir: {args.output_dir}')
    print(f'HR images found: {len(hr_paths)}')
    print(f'LR images found: {len(lr_paths)}')
    print(f'Matched pairs: {len(pairs)}')
    print(f'Missing LR pairs: {len(missing_lr)}')
    print(f'Extra LR images: {len(extra_lr)}')
    print(f'Duplicate LR keys skipped: {len(duplicate_lr)}')
    print(f'Duplicate HR keys skipped: {len(duplicate_hr_keys)}')
    print(f"Image split counts: train={len(split_map['train'])}, val={len(split_map['val'])}, test={len(split_map['test'])}")
    print(f'scale={args.scale}, patch_size={args.patch_size}, stride={args.stride}, bands={args.bands}')
    print(f'normalize_mode={args.normalize_mode}, match_mode={args.match_mode}')

    if missing_lr:
        print('[WARN] First HR images without LR match:')
        for path in missing_lr[:10]:
            print(f'  {path}')
    if extra_lr:
        print('[WARN] First LR images without HR match:')
        for path in extra_lr[:10]:
            print(f'  {path}')
    if not pairs:
        print('No matched HR/LR pairs were found. Please check --hr_dir, --lr_dir, and --match_mode.')
        return

    patch_rows, image_stats = [], []
    total_valid = total_skipped = 0
    for split in SPLITS:
        for pair in split_map[split]:
            try:
                rows, valid, skipped = process_pair(pair, split, args)
            except Exception as exc:
                print(f"[WARN] Skip pair HR={pair['hr']} LR={pair['lr']}: {exc}")
                rows, valid, skipped = [], 0, 0
            patch_rows.extend(rows)
            total_valid += valid
            total_skipped += skipped
            image_stats.append({
                'split': split,
                'source_id': pair['source_id'],
                'valid_patches': valid,
                'skipped_patches': skipped,
            })
            print(f"{split} {pair['hr'].name}: valid patches={valid}, skipped patches={skipped}")

    if args.manifest:
        fields = [
            'split', 'source_id', 'hr_source_image', 'lr_source_image', 'filename',
            'hr_path', 'lr_path', 'residual_path', 'hr_row', 'hr_col', 'lr_row', 'lr_col',
            'valid_ratio', 'zero_ratio', 'status'
        ]
        write_csv(Path(args.output_dir) / 'manifests' / 'patch_manifest.csv', patch_rows, fields)

    patch_split_counts = {split: sum(1 for row in patch_rows if row['split'] == split) for split in SPLITS}
    if args.summary_json:
        summary = {
            'hr_dir': args.hr_dir,
            'lr_dir': args.lr_dir,
            'output_dir': args.output_dir,
            'hr_images_found': len(hr_paths),
            'lr_images_found': len(lr_paths),
            'matched_pairs': len(pairs),
            'missing_lr_pairs': len(missing_lr),
            'extra_lr_images': len(extra_lr),
            'image_split_counts': {key: len(value) for key, value in split_map.items()},
            'patch_split_counts': patch_split_counts,
            'total_valid_patches': total_valid,
            'total_skipped_patches': total_skipped,
            'scale': args.scale,
            'patch_size': args.patch_size,
            'stride': args.stride,
            'bands': args.bands,
            'normalize_mode': args.normalize_mode,
            'match_mode': args.match_mode,
            'max_patches_per_image': args.max_patches_per_image,
            'min_valid_ratio': args.min_valid_ratio,
            'seed': args.seed,
        }
        with open(Path(args.output_dir) / 'manifests' / 'generation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'Total valid patches: {total_valid}')
    print(f'Total skipped patches: {total_skipped}')
    print(f"Patch split counts: train={patch_split_counts['train']}, val={patch_split_counts['val']}, test={patch_split_counts['test']}")
    print(f'Manifest dir: {Path(args.output_dir) / "manifests"}')
    print('Done.')


if __name__ == '__main__':
    main()
