import argparse
import csv
import hashlib
import importlib
import inspect
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch

try:
    import rasterio
except ImportError as exc:
    raise ImportError('Please install rasterio to read GeoTIFF files.') from exc


DEFAULT_MODULES = 'models_mae,models_mae_temporal,models_vit,models,vit,satmae,mae'
DEFAULT_MODEL_NAMES = (
    'vit_large_patch16,vit_base_patch16,mae_vit_large_patch16,mae_vit_base_patch16,'
    'satmae_vit_large_patch16,satmae_vit_base_patch16,vit_large_patch16_temporal,'
    'mae_vit_large_patch16_dec512d8b,mae_vit_base_patch16_dec512d8b'
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected true/false, got: {value}')


def parse_args():
    parser = argparse.ArgumentParser(description='Extract offline SatMAE semantic tokens from GF GeoTIFF files.')
    parser.add_argument('--input_dir', default='E:/GeoCodeSR/tuihua_paired_x4/train/LR_x4')
    parser.add_argument('--output_dir', default='E:/GeoCodeSR/tuihua_paired_x4/train/SatMAE_tokens_mean')
    parser.add_argument('--source', default='lq', choices=['lq', 'hr', 'bicubic_lq'])
    parser.add_argument('--bands', type=int, default=3)
    parser.add_argument('--rgb_bands', default='1,2,3', help='1-based RGB band ids; default assumes RGB source order.')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--mode', default='dummy', choices=['dummy', 'local_satmae', 'torchscript'])
    parser.add_argument(
        '--satmae_repo',
        default=str(Path(__file__).resolve().parents[2] / 'SatMAE-main'),
        help='SatMAE repository path (defaults to this GeoCodeSR checkout/SatMAE-main).')
    parser.add_argument('--model_module', default=None)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--ckpt', default='pretrained_models/satmae/pretrain_fmow_temporal.pth')
    parser.add_argument('--auto_discover_satmae', action='store_true')
    parser.add_argument('--candidate_modules', default=DEFAULT_MODULES)
    parser.add_argument('--candidate_model_names', default=DEFAULT_MODEL_NAMES)
    parser.add_argument('--strict_load', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--token_reduce', default='mean', choices=['patch', 'cls', 'mean', 'feature_map'])
    parser.add_argument('--normalize', default='imagenet', choices=['none', 'imagenet', 'half'])
    parser.add_argument('--input_adapter', default='rgb', choices=['auto', 'rgb', 'repeat_channel', 'repeat_time', 'conv1x1'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0, help='Reserved for future DataLoader use.')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_manifest', action='store_true')
    return parser.parse_args()


def split_csv(value):
    return [v.strip() for v in value.split(',') if v.strip()]


def list_tifs(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory does not exist: {input_dir}')
    paths = sorted([p for p in input_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}])
    if not paths:
        raise FileNotFoundError(f'No tif/tiff files found in: {input_dir}')
    return paths


def parse_rgb_bands(rgb_bands):
    indices = [int(v.strip()) - 1 for v in rgb_bands.split(',') if v.strip()]
    if len(indices) != 3:
        raise ValueError(f'--rgb_bands must contain three 1-based band ids, got: {rgb_bands}')
    if min(indices) < 0:
        raise ValueError(f'--rgb_bands must be 1-based positive ids, got: {rgb_bands}')
    return indices


def normalize_image(image, source_dtypes=None):
    image = image.astype(np.float32)
    finite_mask = np.isfinite(image)
    if finite_mask.any() and np.nanmax(image[finite_mask]) > 1.0:
        is_uint8 = source_dtypes and all(np.dtype(dtype) == np.dtype('uint8') for dtype in source_dtypes)
        image = image / (255.0 if is_uint8 else 10000.0)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def apply_tensor_normalize(tensor, normalize):
    if normalize == 'none':
        return tensor
    if normalize == 'half':
        return (tensor - 0.5) / 0.5
    if normalize == 'imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std
    raise ValueError(f'Unsupported normalize mode: {normalize}')


def read_rgb_tensor(path, bands, rgb_indices, image_size, normalize):
    with rasterio.open(path) as src:
        if src.count < bands:
            raise ValueError(f'{path} has {src.count} bands, but {bands} were requested.')
        data = src.read(list(range(1, bands + 1)))
        source_dtypes = src.dtypes[:bands]
    image = np.transpose(data, (1, 2, 0))
    image = normalize_image(image, source_dtypes=source_dtypes)
    if max(rgb_indices) >= image.shape[2]:
        raise ValueError(f'RGB band index out of range for {path}: rgb_indices={rgb_indices}, shape={image.shape}')
    rgb = image[:, :, rgb_indices]
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
    return apply_tensor_normalize(tensor, normalize)


def torch_load_checkpoint(path, map_location='cpu'):
    try:
        return torch.load(str(path), map_location=map_location)
    except Exception as exc:
        if 'Weights only load failed' in str(exc) or exc.__class__.__name__ == 'UnpicklingError':
            print('WARNING: torch.load default weights_only=True failed; retrying with weights_only=False for trusted local checkpoint.')
            return torch.load(str(path), map_location=map_location, weights_only=False)
        raise


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ['model', 'state_dict', 'params', 'module']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key], key
    return checkpoint, 'direct'


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ['module.', 'model.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def filter_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    compatible = {}
    skipped_shape = []
    unexpected = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible[key] = value
    missing = [key for key in model_state.keys() if key not in compatible]
    return compatible, missing, unexpected, skipped_shape


def load_checkpoint_to_model(model, ckpt, strict_load):
    checkpoint = torch_load_checkpoint(ckpt, map_location='cpu') if isinstance(ckpt, (str, Path)) else ckpt
    state_dict, loaded_key = extract_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)
    if strict_load:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        skipped_shape = []
        compatible = state_dict
    else:
        compatible, missing, unexpected, skipped_shape = filter_compatible_state_dict(model, state_dict)
        model.load_state_dict(compatible, strict=False)
    return {
        'loaded_key': loaded_key,
        'loaded_tensors': len(compatible),
        'missing': missing,
        'unexpected': unexpected,
        'skipped_shape': skipped_shape,
        'state_dict_len': len(state_dict),
    }


def instantiate_builder(builder):
    try:
        return builder()
    except TypeError as exc:
        signature = str(inspect.signature(builder)) if callable(builder) else 'unknown'
        raise TypeError(f'Could not instantiate builder with no arguments. Signature: {signature}') from exc


def add_repo_to_path(satmae_repo):
    repo = Path(satmae_repo)
    if not repo.exists():
        raise FileNotFoundError(f'SatMAE repo does not exist: {repo}')
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def apply_satmae_compat_patches():
    if not hasattr(np, 'float'):
        np.float = float
    try:
        import timm.models.vision_transformer as vision_transformer
    except Exception as exc:
        print(f'WARNING: failed to import timm for compatibility patching: {exc}')
        return
    block = getattr(vision_transformer, 'Block', None)
    if block is None or getattr(block, '_satmae_qk_scale_compatible', False):
        return

    def block_compat(*args, **kwargs):
        kwargs.pop('qk_scale', None)
        return block(*args, **kwargs)

    block_compat._satmae_qk_scale_compatible = True
    vision_transformer.Block = block_compat
    print('Applied SatMAE compatibility patch: timm Block ignores legacy qk_scale.')


def get_patch_embed_info(model):
    patch_embed = getattr(model, 'patch_embed', None)
    proj = getattr(patch_embed, 'proj', None)
    weight = getattr(proj, 'weight', None)
    if weight is None:
        return None
    shape = tuple(weight.shape)
    in_chans = shape[1] if len(shape) >= 2 else None
    patch_size = shape[2] if len(shape) >= 4 else None
    embed_dim = shape[0] if shape else None
    return {'shape': shape, 'in_chans': in_chans, 'patch_size': patch_size, 'embed_dim': embed_dim}


def discover_satmae_model(candidate_modules, candidate_model_names, ckpt, strict_load):
    best = None
    errors = []
    checkpoint = torch_load_checkpoint(ckpt, map_location='cpu')
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f'{module_name}: import failed: {exc}')
            continue
        for model_name in candidate_model_names:
            if not hasattr(module, model_name):
                continue
            builder = getattr(module, model_name)
            if not callable(builder):
                continue
            try:
                model = instantiate_builder(builder)
                load_info = load_checkpoint_to_model(model, checkpoint, strict_load=False)
                score = load_info['loaded_tensors']
                record = (score, module_name, model_name, model, load_info)
                if best is None or score > best[0]:
                    best = record
            except Exception as exc:
                errors.append(f'{module_name}.{model_name}: {exc}')

    if best is None or best[0] == 0:
        print('SatMAE checkpoint exists, but model definition was not found. Please provide --satmae_repo, --model_module, and --model_name.')
        print('Auto-discovery errors / hints:')
        for line in errors[:50]:
            print(f'  {line}')
        raise RuntimeError('Auto-discovery failed for SatMAE model definition.')

    _, module_name, model_name, model, load_info = best
    print(f'Auto-discovered SatMAE model: module={module_name}, model_name={model_name}')
    print_load_info(load_info)
    return model, module_name, model_name


def print_load_info(load_info):
    print(f'Checkpoint loaded key: {load_info["loaded_key"]}')
    print(f'Loaded compatible tensors: {load_info["loaded_tensors"]}/{load_info["state_dict_len"]}')
    print(f'Missing keys: {len(load_info["missing"])}')
    print(f'Unexpected keys: {len(load_info["unexpected"])}')
    print(f'Shape-mismatch skipped keys: {len(load_info["skipped_shape"])}')
    if load_info['missing']:
        print(f'First missing keys: {load_info["missing"][:30]}')
    if load_info['unexpected']:
        print(f'First unexpected keys: {load_info["unexpected"][:30]}')
    if load_info['skipped_shape']:
        print(f'First shape-mismatch keys: {load_info["skipped_shape"][:30]}')


def build_satmae_model(args, device):
    if args.mode == 'dummy':
        print('WARNING: using dummy semantic tokens, not valid for experiments.')
        return None, None, None

    if args.mode == 'torchscript':
        if not args.ckpt:
            raise ValueError('--ckpt is required when --mode torchscript.')
        model = torch.jit.load(str(args.ckpt), map_location=device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print(f'Loaded TorchScript SatMAE model: {args.ckpt}')
        return model, 'torchscript', Path(args.ckpt).name

    if args.mode != 'local_satmae':
        raise ValueError(f'Unsupported mode: {args.mode}')

    repo = add_repo_to_path(args.satmae_repo)
    apply_satmae_compat_patches()
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f'SatMAE checkpoint does not exist: {args.ckpt}')

    if args.model_module and args.model_name:
        module = importlib.import_module(args.model_module)
        if not hasattr(module, args.model_name):
            raise AttributeError(f'Module {args.model_module} has no model builder named {args.model_name}.')
        model = instantiate_builder(getattr(module, args.model_name))
        load_info = load_checkpoint_to_model(model, args.ckpt, args.strict_load)
        used_module = args.model_module
        used_name = args.model_name
        print(f'Using user-specified SatMAE model: module={used_module}, model_name={used_name}')
        print_load_info(load_info)
    elif args.auto_discover_satmae:
        model, used_module, used_name = discover_satmae_model(
            split_csv(args.candidate_modules),
            split_csv(args.candidate_model_names),
            args.ckpt,
            args.strict_load)
    else:
        raise ValueError(
            'SatMAE checkpoint exists, but model definition was not found. '
            'Please provide --satmae_repo, --model_module, and --model_name, or enable --auto_discover_satmae.')

    patch_info = get_patch_embed_info(model)
    if patch_info is not None:
        print(f'Patch embed: shape={patch_info["shape"]}, in_chans={patch_info["in_chans"]}, patch_size={patch_info["patch_size"]}, embed_dim={patch_info["embed_dim"]}')
        if patch_info['in_chans'] != 3:
            print(f'WARNING: model patch_embed expects {patch_info["in_chans"]} input channels; current preprocessing outputs RGB 3 channels.')
    else:
        print('WARNING: could not inspect model patch_embed input channels.')

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f'SatMAE repo: {repo}')
    return model, used_module, used_name


def pick_tensor_from_output(output):
    if isinstance(output, dict):
        for key in ['x_norm_patchtokens', 'patch_tokens', 'tokens', 'features', 'last_hidden_state', 'latent', 'cls_token']:
            if key in output and torch.is_tensor(output[key]):
                return output[key]
        tensor_keys = [key for key, value in output.items() if torch.is_tensor(value)]
        raise RuntimeError(f'Cannot identify token tensor from model output dict. Tensor keys: {tensor_keys}')
    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value
        raise RuntimeError('Cannot identify token tensor from tuple/list model output.')
    if torch.is_tensor(output):
        return output
    raise RuntimeError(f'Unsupported model output type: {type(output)}')


def call_forward_encoder(model, x):
    try:
        return model.forward_encoder(x, mask_ratio=0.0)
    except TypeError:
        pass
    try:
        return model.forward_encoder(x, 0.0)
    except TypeError:
        pass
    timestamps = torch.zeros((x.shape[0], 3, 3), device=x.device, dtype=torch.long)
    x_time = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)
    return model.forward_encoder(x_time, timestamps, mask_ratio=0.0)


def reduce_tokens(token, token_reduce):
    if token_reduce == 'feature_map':
        if token.ndim == 4:
            return token, 'feature_map'
        if token.ndim == 3:
            patch_tokens = strip_cls_if_present(token)
            grid = int(round(patch_tokens.shape[1] ** 0.5))
            if grid * grid != patch_tokens.shape[1]:
                raise RuntimeError(f'Cannot reshape patch tokens to feature map: shape={tuple(token.shape)}')
            return patch_tokens.transpose(1, 2).reshape(token.shape[0], token.shape[2], grid, grid), 'feature_map'
        raise RuntimeError(f'Cannot make feature_map from token shape: {tuple(token.shape)}')

    if token.ndim == 4:
        if token_reduce != 'feature_map':
            return token, 'feature_map'
    if token.ndim == 2:
        return token, 'cls'
    if token.ndim != 3:
        raise RuntimeError(f'Unsupported token tensor shape: {tuple(token.shape)}')

    patch_tokens = strip_cls_if_present(token)
    if token_reduce == 'cls':
        return token[:, 0, :] if token.shape[1] != patch_tokens.shape[1] else token.mean(dim=1), 'cls'
    if token_reduce == 'mean':
        return patch_tokens.mean(dim=1), 'mean'
    if token_reduce == 'patch':
        return patch_tokens, 'patch'
    raise ValueError(f'Unsupported token_reduce: {token_reduce}')


def strip_cls_if_present(token):
    n_tokens = token.shape[1]
    maybe_patch = n_tokens - 1
    grid = int(round(maybe_patch ** 0.5))
    if maybe_patch > 0 and grid * grid == maybe_patch:
        return token[:, 1:, :]
    if maybe_patch > 0 and maybe_patch % 3 == 0:
        temporal_grid = int(round((maybe_patch // 3) ** 0.5))
        if temporal_grid * temporal_grid * 3 == maybe_patch:
            return token[:, 1:, :]
    return token


@torch.no_grad()
def extract_tokens_from_model(model, x, token_reduce):
    if hasattr(model, 'forward_features'):
        output = model.forward_features(x)
    elif hasattr(model, 'forward_encoder'):
        output = call_forward_encoder(model, x)
    else:
        output = model(x)
    token = pick_tensor_from_output(output)
    token, token_type = reduce_tokens(token, token_reduce)
    return token.detach().cpu().float().numpy(), token_type


def deterministic_dummy_token(path, seed, shape=(196, 768)):
    digest = hashlib.sha256((str(path.name) + str(seed)).encode('utf-8')).digest()
    local_seed = int.from_bytes(digest[:8], byteorder='little', signed=False) % (2 ** 32)
    rng = np.random.default_rng(local_seed)
    return rng.standard_normal(shape).astype(np.float32)


def save_token(path, token, token_type, source_path, args, actual_module, actual_name):
    output = {
        'token': token.astype(np.float32),
        'token_type': token_type,
        'source_path': str(source_path),
        'source': args.source,
        'model_mode': args.mode,
        'model_name': actual_name or args.model_name,
        'model_module': actual_module or args.model_module,
        'satmae_repo': args.satmae_repo,
        'ckpt': args.ckpt,
        'image_size': args.image_size,
        'normalize': args.normalize,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, output, allow_pickle=True)


def batch_iter(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    input_paths = list_tifs(args.input_dir)
    if args.max_samples is not None:
        input_paths = input_paths[:args.max_samples]
    output_dir = Path(args.output_dir)
    rgb_indices = parse_rgb_bands(args.rgb_bands)
    model, actual_module, actual_name = build_satmae_model(args, device)

    print(f'Input images: {len(input_paths)}')
    print(f'Output dir: {output_dir}')
    print(f'Mode: {args.mode}')
    print(f'Device: {device}')
    print(f'RGB bands: {args.rgb_bands}')
    print(f'Image size: {args.image_size}')
    print(f'Normalize: {args.normalize}')
    print(f'Token reduce: {args.token_reduce}')
    print(f'Input adapter: {args.input_adapter}')

    manifest_rows = []
    saved = 0
    skipped = 0
    failed = 0

    if args.mode == 'dummy':
        for path in input_paths:
            out_path = output_dir / f'{path.stem}.npy'
            if args.resume and out_path.exists():
                skipped += 1
                continue
            token = deterministic_dummy_token(path, args.seed)
            save_token(out_path, token, 'dummy', path, args, actual_module, actual_name)
            manifest_rows.append([str(path), str(out_path), 'dummy', str(token.shape)])
            saved += 1
            if saved % 100 == 0:
                print(f'Saved dummy tokens: {saved}')
    else:
        pending = []
        for path in input_paths:
            out_path = output_dir / f'{path.stem}.npy'
            if args.resume and out_path.exists():
                skipped += 1
                continue
            pending.append((path, out_path))

        for batch in batch_iter(pending, args.batch_size):
            valid_items = []
            tensors = []
            for path, out_path in batch:
                try:
                    tensors.append(read_rgb_tensor(path, args.bands, rgb_indices, args.image_size, args.normalize))
                    valid_items.append((path, out_path))
                except Exception as exc:
                    failed += 1
                    print(f'[WARN] Failed to read {path}: {exc}')
            if not tensors:
                continue
            x = torch.stack(tensors, dim=0).to(device)
            try:
                tokens, token_type = extract_tokens_from_model(model, x, args.token_reduce)
            except Exception as exc:
                raise RuntimeError(f'Failed to extract SatMAE tokens: {exc}') from exc

            if tokens.shape[0] != len(valid_items):
                raise RuntimeError(f'Batch output first dimension mismatch: token_shape={tokens.shape}, batch={len(valid_items)}')
            for i, (path, out_path) in enumerate(valid_items):
                token = tokens[i]
                save_token(out_path, token, token_type, path, args, actual_module, actual_name)
                manifest_rows.append([str(path), str(out_path), token_type, str(token.shape)])
                saved += 1
            print(f'Saved tokens: {saved}/{len(pending)}')

    if args.save_manifest:
        manifest_path = output_dir / 'tokens_manifest.csv'
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source_path', 'token_path', 'token_type', 'token_shape'])
            writer.writerows(manifest_rows)
        print(f'Manifest saved: {manifest_path}')

    print(f'Done. saved={saved}, skipped={skipped}, failed={failed}, output_dir={output_dir}')


if __name__ == '__main__':
    main()
