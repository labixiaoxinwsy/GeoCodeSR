# Training Options

## GeoCodeSR RGB pipeline

The paired RGB dataset root is `E:/GeoCodeSR/tuihua_paired_x4`. The current task uses
three input/output channels, real paired 2 m LR and 0.5 m HR images, and scale x4.

Training options:

1. `train_residual_vq_codebook_tuihua_rgb.yml`: Stage 1 residual VQ pretraining.
2. `train_geocodesr_baseline_tuihua_rgb.yml`: rebuilt encoder-decoder baseline.
3. `train_semantic_codebook_sr_satmae_tuihua_rgb.yml`: residual-code retrieval with static reassembly.
4. `train_semantic_dynamic_kernel_sr_satmae_tuihua_rgb.yml`: retrieval plus dynamic pixel reassembly.
5. `train_semantic_frequency_expert_sr_satmae_tuihua_rgb.yml`: full single-path GeoCodeSR.

The original QuanTexSR options are retained only for reproducing the external original-model baseline.
Full GeoCodeSR does not contain a complete QuanTexSR bypass.

## Stage 1: paired data and residual VQ pretraining

Stage 1 builds leakage-free paired patches and learns the high-frequency residual codebook.
SatMAE tokens are not consumed by the residual VQ model in this stage.

```mermaid
flowchart TD
    HR0["Paired HR GeoTIFFs<br/>0.5 m, RGB"] --> MATCH["Match HR and LR by filename"]
    LR0["Paired real LR GeoTIFFs<br/>2 m, RGB"] --> MATCH
    MATCH --> SPLIT["Image-level split<br/>train 288 / val 36 / test 36"]
    SPLIT --> CROP["Aligned sliding crop<br/>HR 256 x 256, stride 256<br/>LR 64 x 64, scale x4"]
    CROP --> FILTER["Valid-pixel and finite-value checks"]
    FILTER --> HR["HR patches<br/>34848 / 4356 / 4356"]
    FILTER --> LR["Real LR_x4 patches<br/>34848 / 4356 / 4356"]
    HR --> RES["Residual = HR - BicubicUp LR"]
    LR --> RES
    RES --> RESTIF["Residual_x4 float32 GeoTIFFs"]
    RESTIF --> AUDIT["Pair, shape, value and<br/>source-leakage audit"]

    AUDIT --> DATASET["GFResidualCodebookDataset<br/>3 x 256 x 256"]
    DATASET --> SCALE["Residual scaling x20"]
    SCALE --> ENC["ResidualEncoder<br/>two x2 downsampling stages"]
    ENC --> Z["Continuous latent z<br/>64 x 64 x 64"]
    Z --> VQ["Residual VectorQuantizer<br/>512 codes x 64 dimensions"]
    VQ --> IDX["Code indices<br/>64 x 64"]
    VQ --> ZQ["Quantized latent z_q"]
    ZQ --> DEC["ResidualDecoder<br/>two x2 upsampling stages"]
    DEC --> RECON["Reconstructed scaled residual<br/>3 x 256 x 256"]
    RECON --> LOSS["L1 + 0.05 FFT + VQ loss"]
    SCALE --> LOSS
    LOSS --> OPT["AdamW, lr 2e-4<br/>30000 iterations"]
    OPT --> VAL["Validation<br/>L1, FFT, perplexity, usage"]
    VAL --> CKPT["net_residual_vq_latest.pth"]
    CKPT --> STAGE2["Stage 2 core quantization path"]
```

Stage 1 files:

- `scripts/data_preparation/generate_paired_lr_hr_residual_image_split.py`
- `scripts/data_preparation/check_gf_image_level_dataset.py`
- `basicsr/data/gf_residual_codebook_dataset.py`
- `basicsr/archs/gf_residual_vqvae_arch.py`
- `scripts/train/train_residual_vq_codebook.py`
- `options/train_residual_vq_codebook_tuihua_rgb.yml`

## Stage 2: single-path backbone-level GeoCodeSR rebuild

Stage 2 retains only two pretrained feature extractors from QuanTexSR:

- RRDB Content Model.
- Multi-scale Encoder.

The following original QuanTexSR modules are directly replaced and do not exist in the Full model:

- Native texture `quantize_group`, `before_quant_group`, and `after_quant_group`.
- Original `SimSFT` fusion blocks.
- Original `DecoderBlock` modules and their fixed `Upsample` operations.
- Original reconstruction `out_conv`.
- Complete original QuanTexSR bypass output.

### 2.1 Offline SatMAE tokens and paired batch

SatMAE is frozen and runs offline. GeoCodeSR reads one cached 1024-D mean token per LR patch.

```mermaid
flowchart TD
    LR["LR_x4 RGB<br/>3 x 64 x 64"] --> PRE["Resize 224 x 224<br/>ImageNet normalization"]
    PRE --> TEMP["Temporal compatibility adapter<br/>repeat to 3 timestamps when required"]
    TEMP --> SAT["Frozen SatMAE ViT-Large patch16<br/>pretrain_fmow_temporal.pth"]
    SAT --> TOKENS["Patch tokens without CLS"]
    TOKENS --> MEAN["Mean token pooling"]
    MEAN --> CACHE["1024-D float32 .npy"]

    LR2["LR patch<br/>3 x 64 x 64"] --> MATCH["GFPairedSemanticDataset<br/>filename matching and checks"]
    HR["HR patch<br/>3 x 256 x 256"] --> MATCH
    RES["GT residual<br/>3 x 256 x 256"] --> MATCH
    CACHE --> MATCH
    MATCH --> BATCH["Batch: lq, gt, residual,<br/>semantic_token"]
```

### 2.2 New core quantization path

The Stage 1 codebook replaces the original texture quantizer as the only quantization path.
The encoder and codebook loaded from Stage 1 are frozen; the semantic code predictor is trainable.

```mermaid
flowchart TD
    LQ["LR<br/>3 x 64 x 64"] --> RRDB["Retained RRDB Content Model"]
    RRDB --> CONTENTHR["Content image<br/>3 x 256 x 256"]
    CONTENTHR --> DOWN["Resize to LR grid"]
    DOWN --> CONTENTLR["Content LR<br/>3 x 64 x 64"]
    CONTENTLR --> STEM["Trainable content stem<br/>64 x 64 x 64"]

    TOKEN["SatMAE token<br/>1024-D"] --> SEM["Linear projection<br/>gamma 64-D, beta 64-D"]
    STEM --> MOD["Semantic modulation<br/>f * (1 + gamma) + beta"]
    SEM --> MOD
    MOD --> PRED["Residual-code predictor"]
    PRED --> LOGITS["code_logits<br/>512 x 64 x 64"]
    LOGITS --> PROB["Softmax, temperature 1.0"]

    STAGE1["Stage 1 checkpoint"] --> FENC["Frozen ResidualEncoder"]
    STAGE1 --> FVQ["Frozen codebook<br/>512 x 64-D"]
    PROB --> SOFT["Training soft code lookup"]
    FVQ --> SOFT
    LOGITS --> HARDIDX["Evaluation argmax"]
    HARDIDX --> HARD["Hard code lookup"]
    FVQ --> HARD
    SOFT --> CODE["code_feat<br/>64 x 64 x 64"]
    HARD --> CODE
    PROB --> CONF["Confidence map<br/>maximum code probability"]
    PROB --> UNC["Normalized uncertainty<br/>entropy divided by log 512"]

    RESGT["GT residual<br/>training mode only"] --> SCALE["Multiply by 20"]
    SCALE --> FENC
    FENC --> TARGETZ["Target latent"]
    TARGETZ --> FVQ
    FVQ --> TARGETIDX["target_indices<br/>64 x 64"]
    LOGITS --> CE["Target-code cross entropy"]
    TARGETIDX --> CE
```

`ResidualEncoder` is a target-label generator only. It runs under `torch.no_grad()` only when
the model is in training mode and `residual_gt` is supplied. Validation and inference never call
the target encoder and reconstruct SR solely from LR plus the cached semantic token.

Scale alignment and channel projection are explicit and stage-specific:

| Scale | Main feature | `code_feat` alignment | Semantic alignment | Confidence/uncertainty alignment |
|---|---|---|---|---|
| 32 bottleneck | 256 x 32 x 32 | 64 to 256 channels, bilinear 64 to 32 | 1024 to 256, spatial expansion | confidence and normalized entropy independently resized 64 to 32 |
| 64 Decoder | 256 x 64 x 64 | 64 to 256 channels, native 64 grid | 1024 to 256, spatial expansion | confidence and normalized entropy remain on the 64 grid |
| 128 Decoder | 128 x 128 x 128 | 64 to 128 channels, bilinear 64 to 128 | 1024 to 128, spatial expansion | confidence and normalized entropy independently resized 64 to 128 |
| 256 Decoder | 64 x 256 x 256 | 64 to 64 channels, bilinear 64 to 256 | 1024 to 64, spatial expansion | confidence and normalized entropy independently resized 64 to 256 |

The four independent adapters are `condition_align_32`, `condition_align_64`,
`condition_align_128`, and `condition_align_256`. For each dynamic reassembly predictor at
input scales 32, 64, and 128, code, semantic, confidence, and normalized entropy are projected
independently and fused as a zero-initialized modulation of the CARAFE compressed feature.
No condition relies on implicit spatial broadcasting.

At the bottleneck and the three Decoder output scales, `GatedCodeSkipFusion` replaces SimSFT.
It predicts independent skip and code gates from decoder, skip, aligned code, aligned semantic,
confidence, and entropy features, then reports each gate mean and response mean for auditing.

### 2.3 Code-response-driven dynamic pixel reassembly

Each decoder stage uses an x2 reassembly module following the official CARAFE pack structure.
There is no fixed nearest/bilinear
upsampling in the Full Decoder main path. Internal nearest expansion only aligns unfolded source
patches with the output grid; the output value is determined by the predicted normalized kernel.

```mermaid
flowchart LR
    FIN["Input decoder feature<br/>C_in x H x W"] --> COMP["CARAFE channel compressor<br/>1 x 1, 64 channels"]
    FIN --> UNFOLD["3 x 3 unfold<br/>C_in x 9 x H x W"]
    UNFOLD --> ALIGN["Align source patches<br/>to 2H x 2W grid"]

    CODE["Normalized code_feat"] --> CCOND["Code condition"]
    TOKEN["Normalized SatMAE feature"] --> SCOND["Semantic condition"]
    CONF["Code confidence"] --> CPROJ["Independent confidence projection"]
    UNC["Normalized code entropy"] --> EPROJ["Independent entropy projection"]
    CPROJ --> CMOD["Zero-initialized condition fusion"]
    EPROJ --> CMOD
    CCOND --> CMOD
    SCOND --> CMOD
    COMP --> ADD["Compressed feature + condition modulation"]
    CMOD --> ADD
    ADD --> KLOGITS["CARAFE content encoder<br/>36 channels = 9 x 2 x 2"]
    KLOGITS --> PS["PixelShuffle x2"]
    PS --> SOFTMAX["Softmax over 9 positions"]
    SOFTMAX --> KERNEL["Per-pixel normalized<br/>3 x 3 kernels at 2H x 2W"]
    ALIGN --> SUM["Kernel-weighted sum"]
    KERNEL --> SUM
    SUM --> REFINE["Conv + two QuanTexSR<br/>GroupNorm residual blocks"]
    REFINE --> FOUT["Reassembled feature<br/>C_out x 2H x 2W"]
```

### 2.4 Decoder-embedded semantic-codebook frequency experts

Low/mid/high experts are applied inside every Decoder stage, not after image reconstruction.
Each scale uses a spatial router. It outputs `[B, 3, H, W]` weights and normalizes the three
experts independently at every spatial position. The innovation is therefore region-adaptive
semantic-codebook joint routing rather than image-level correction.
The feature, code, and semantic inputs are normalized before routing. Experts use the
GroupNorm pre-activation residual design from the QuanTexSR Decoder and zero-initialized output
projections. Router logits are regularized with the standard router z-loss; no entropy floor,
logit clamp, or expert-output clipping is used.

```mermaid
flowchart TD
    F["Decoder feature"] --> LOW["Low band<br/>Gaussian sigma 2.0"]
    F --> MID["Mid band<br/>Gaussian sigma 1.0 minus low"]
    F --> HIGH["High band<br/>feature minus Gaussian sigma 1.0"]
    LOW --> E1["Low-frequency expert"]
    MID --> E2["Mid-frequency expert"]
    HIGH --> E3["High-frequency expert"]

    TOKEN["Aligned semantic feature map"] --> ROUTER["Per-scale spatial router"]
    CODE["Aligned code feature map"] --> ROUTER
    CONF["Aligned confidence map"] --> ROUTER
    UNC["Aligned normalized entropy map"] --> ROUTER
    F --> ROUTER
    ROUTER --> WEIGHT["Spatial softmax weights [B,3,H,W]<br/>low, mid, high"]
    E1 --> MIX["Weighted expert sum"]
    E2 --> MIX
    E3 --> MIX
    WEIGHT --> MIX
    MIX --> DELTA["Feature frequency delta"]
    DELTA --> OUT["f_out = f + lambda_freq * delta"]
    F --> OUT
```

### 2.5 Full single reconstruction path

```mermaid
flowchart TD
    LQ["LR<br/>3 x 64 x 64"] --> RRDB["Retained RRDB Content Model<br/>x4 content image"]
    RRDB --> CHR["detached content_hr<br/>3 x 256 x 256"]
    CHR --> MSE["Retained Multi-scale Encoder"]
    MSE --> E32["enc_32<br/>256 x 32 x 32"]
    MSE --> E64["enc_64<br/>256 x 64 x 64"]
    MSE --> E128["enc_128<br/>128 x 128 x 128"]

    CODE["Semantic residual code_feat"] --> BFUSE["Bottleneck code fusion"]
    TOKEN["SatMAE token"] --> BFUSE
    E32 --> BFUSE
    BFUSE --> F32["code-conditioned feature<br/>256 x 32 x 32"]

    F32 --> DPR64["Dynamic pixel reassembly<br/>32 to 64, 256 channels"]
    CODE --> DPR64
    TOKEN --> DPR64
    DPR64 --> FUSE64["Fuse enc_64 and code context"]
    E64 --> FUSE64
    FUSE64 --> FREQ64["Embedded frequency experts<br/>at 64 x 64"]

    FREQ64 --> DPR128["Dynamic pixel reassembly<br/>64 to 128, 128 channels"]
    CODE --> DPR128
    TOKEN --> DPR128
    DPR128 --> FUSE128["Fuse enc_128 and code context"]
    E128 --> FUSE128
    FUSE128 --> FREQ128["Embedded frequency experts<br/>at 128 x 128"]

    FREQ128 --> DPR256["Dynamic pixel reassembly<br/>128 to 256, 64 channels"]
    CODE --> DPR256
    TOKEN --> DPR256
    DPR256 --> FUSE256["Fuse content_hr and code context"]
    CHR --> FUSE256
    FUSE256 --> FREQ256["Embedded frequency experts<br/>at 256 x 256"]
    FREQ256 --> HEAD["New reconstruction head<br/>ResBlock and Conv 64 to 3"]
    HEAD --> SR["Single output sr_final<br/>3 x 256 x 256"]
```

There is no `sr_base` or `sr_enhanced` bypass output. Full GeoCodeSR has one reconstruction result.

### 2.6 Losses, optimizer, and validation

```mermaid
flowchart TD
    SR["sr_final"] --> LREC["L1 reconstruction<br/>weight 1.0"]
    GT["GT HR"] --> LREC
    SR --> LFFT["FFT magnitude L1<br/>weight 0.03"]
    GT --> LFFT
    SR --> LSAM["SAM proxy<br/>weight 0.01"]
    GT --> LSAM
    LOGITS["code_logits"] --> LCODE["Target-code CE<br/>weight 1.0"]
    TARGET["target_indices"] --> LCODE

    SR --> SFB["Decompose final SR into<br/>low/mid/high bands"]
    GT --> FLOW["GT low<br/>Gaussian sigma 2.0"]
    GT --> FMID["GT mid<br/>Gaussian sigma 1.0 minus sigma 2.0"]
    GT --> FHIGH["GT high<br/>GT minus Gaussian sigma 1.0"]
    SFB --> LFREQ["Three final-image band L1 losses<br/>combined weight 0.05"]
    FLOW --> LFREQ
    FMID --> LFREQ
    FHIGH --> LFREQ
    LOAD["Hard assignments and soft importance<br/>at each Decoder scale"] --> LBAL["Switch load-balance loss<br/>weight 0.001"]
    RLOGIT["Router logits"] --> RZ["Router z-loss<br/>weight 0.001"]

    LREC --> TOTAL["Weighted total loss"]
    LFFT --> TOTAL
    LSAM --> TOTAL
    LCODE --> TOTAL
    LFREQ --> TOTAL
    LBAL --> TOTAL
    RZ --> TOTAL
    TOTAL --> AMP["AMP backward"]
    AMP --> ADAM["Adam, betas 0.9 and 0.99"]
    ADAM --> LOWLR["Retained Encoder<br/>lr 5e-6"]
    ADAM --> MAINLR["New quantization and Decoder<br/>lr 1e-4"]
    ADAM --> VAL["Validation every 1000 iter<br/>PSNR, MAE, SAM, code usage,<br/>route and kernel entropy"]
    ADAM --> SAVE["Checkpoint every 2000 iter"]
```

Full-model states:

| Module | Source | State |
|---|---|---|
| RRDB Content Model | Original QuanTexSR checkpoint | Frozen; output detached as in official LQ path |
| Multi-scale Encoder | Original QuanTexSR checkpoint | Trainable, low LR |
| Original texture Quantizer/SFT/Decoder/head | Replaced | Not instantiated |
| ResidualEncoder | Stage 1 checkpoint | Frozen, training target path only |
| 512 x 64 residual codebook | Stage 1 checkpoint | Frozen, prediction lookup and training target lookup |
| Semantic residual-code predictor | New | Trainable |
| Three dynamic pixel reassembly stages | New | Trainable |
| Three code/skip fusion stages plus bottleneck fusion | New | Trainable |
| Three routed frequency-expert groups | New | Trainable |
| New reconstruction head | New | Trainable |
| SatMAE | Temporal checkpoint | Frozen, offline only |

Stage 2 files:

- `scripts/features/extract_satmae_tokens.py`
- `basicsr/data/gf_paired_semantic_dataset.py`
- `basicsr/archs/gf_semantic_frequency_expert_sr_arch.py`
- `scripts/train/train_semantic_frequency_expert_sr.py`
- `scripts/inference/infer_semantic_frequency_expert_sr.py`
- `options/train_semantic_frequency_expert_sr_satmae_tuihua_rgb.yml`

## Progressive ablations

All GeoCodeSR ablations use the rebuilt single path. The original QuanTexSR is evaluated separately.

| Experiment | Residual code retrieval | Dynamic reassembly | Embedded frequency experts |
|---|---:|---:|---:|
| A Rebuilt baseline | false | false, fixed-center reassembly | false |
| B Core quantization | true | false, fixed-center reassembly | false |
| C Dynamic Decoder | true | true | false |
| D Full GeoCodeSR | true | true | true |

When dynamic reassembly is disabled for an ablation, the same reassembly operator uses a fixed
center kernel. This ablation behavior does not add the removed original Decoder back into the model.

## Original QuanTexSR reference options

- `train_QuanTexSR_HQ_pretrain_stage.yml`
- `train_QuanTexSR_LQ_stage.yml`
- `train_QuanTexSR_semantic_cluster_stage.yml`
