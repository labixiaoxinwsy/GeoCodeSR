# GeoCodeSR

GeoCodeSR is a three-channel remote-sensing super-resolution project rebuilt from selected QuanTexSR components.
The current pipeline uses paired 0.5 m HR and 2 m LR RGB GeoTIFF images at scale x4.

## Project layout

- `basicsr/`: BasicSR framework and GeoCodeSR datasets/networks.
- `options/`: original QuanTexSR options and current GeoCodeSR RGB options.
- `scripts/data_preparation/`: paired dataset generation and validation.
- `scripts/features/`: SatMAE mean-token extraction.
- `scripts/train/`: residual VQ and unified GeoCodeSR training.
- `scripts/inference/`: residual VQ and unified GeoCodeSR inference.
- `SatMAE-main/`: local SatMAE model definitions.

Datasets and generated figure/experiment data are not included. Model weights are
kept under `pretrained_models/` and `experiments/`; Git LFS is configured for
the included `.pth` files. The 3.7 GB SatMAE checkpoint is not included in this
repository because it exceeds GitHub's 2 GiB per-file limit. Download the official
[fMoW-Temporal ViT-Large pretraining checkpoint](https://zenodo.org/record/7369797/files/pretrain_fmow_temporal.pth)
and save it as `pretrained_models/satmae/pretrain_fmow_temporal.pth` before
extracting semantic tokens. This file is ignored by Git and will remain local.
The training options currently use the local dataset path
`E:/GeoCodeSR/tuihua_paired_x4`; update the paths in the selected option file for
your machine before training. BasicSR's legacy dataset file-list manifests are
archived outside the repository; restore them only if using the upstream dataset
options that require those lists.

## RGB training order

1. Generate and check the paired LR/HR/residual dataset.
2. Extract SatMAE mean tokens for train, val, and test splits.
3. Train the three-channel residual VQ codebook.
4. Reproduce the original QuanTexSR baseline.
5. Train codebook, dynamic-kernel, or full frequency-expert configurations.

```powershell
python scripts/train/train_residual_vq_codebook.py -opt options/train_residual_vq_codebook_tuihua_rgb.yml
python scripts/train/train_semantic_frequency_expert_sr.py -opt options/train_semantic_frequency_expert_sr_satmae_tuihua_rgb.yml
```

Full GeoCodeSR retains the pretrained RRDB Content Model and Multi-scale Encoder. It replaces the original texture
Quantizer, SFT fusion, fixed-upsample Decoder, and reconstruction head with one semantic residual-code quantization path,
three code-response-driven dynamic pixel reassembly stages, Decoder-embedded routed frequency experts, and a new head.
There is no complete original QuanTexSR bypass inside the Full model.
The dynamic kernels are spatial and use both code confidence and normalized code entropy. Frequency-expert routing
predicts normalized `[B, 3, H, W]` low/mid/high weights from decoder, code, semantic, confidence, and entropy maps,
so the Full model performs region-adaptive reconstruction at every Decoder scale.
See `options/README.md` for the complete option index. The RGB pipeline does not use an NIR band, so NDVI loss and metrics are disabled.

## Environment and data

Install Git LFS before cloning or pushing this repository so the included model
checkpoints are handled correctly. Install the pinned dependencies from
`requirements.txt` in a compatible Python environment. The PyTorch/CUDA pins in
that file reflect the environment used to verify this project and may need to be
adjusted for a different platform or GPU. Dataset files and generated experiment
outputs remain local.

## Attribution

The project retains code from QuanTexSR, BasicSR, and SatMAE. See `LICENSE` and
the upstream license files for licensing information. Review checkpoint
redistribution terms before publishing the model weights.
QuanTexSR references that remain in source names, baseline options, checkpoints, and the license identify the upstream
baseline and are not the name of this project.
