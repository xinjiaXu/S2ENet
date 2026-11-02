# S2ENet Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/S2ENet`  
Dataset root: `/home/xxj/Data`

## Clone and update

First time:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test
git clone --branch reproduce https://github.com/xinjiaXu/S2ENet.git S2ENet
```

Later updates:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/S2ENet
git switch reproduce
git pull --ff-only
```

## Environment and pretrained backbone

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/S2ENet
bash setup_server_env.sh
conda activate s2enet-rs
bash prepare_server_data.sh /home/xxj/Data
```

The isolated environment uses Python 3.10 and PyTorch 2.7.1/CUDA 12.8 for the RTX 5090. Setup caches the ResNet18 ImageNet checkpoint used by the model constructor. This checkpoint is stored in the user's Torch cache, not in the repository.

## Dataset layout

The adapter reads the existing image and label folders directly. If `train.txt`, `val.txt`, or `test.txt` exists either in the split's `list/` folder or beside the split, it is used; otherwise filenames are read from the label folder.

| Key | Dataset root | Image A | Image B | Label |
| --- | --- | --- | --- | --- |
| `levircd` | `/home/xxj/Data/LEVIR_CD` | `A` | `B` | `label` |
| `levircdplus` | `/home/xxj/Data/LEVIR_CD_PLUS/cut_data` | `A` | `B` | `label` |
| `cdd` | `/home/xxj/Data/CDD` | `A` | `B` | `OUT` |
| `whucd` | `/home/xxj/Data/WHUCD` | `A` | `B` | `label` |
| `sysucd` | `/home/xxj/Data/SYSU_CD` | `time1` | `time2` | `label` |

Supported image formats are those readable by OpenCV; TIFF labels named `.tif` are mapped to same-stem `.png` labels to match the existing dataset adapter convention.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

Defaults follow the source CLI: 40,000 max steps, batch size 32, 256x256 inputs, lr `5e-4`, and polynomial decay. Training validates on `val`, saves the best-F1 checkpoint and evaluates `test` at the end. A later run resumes from the dataset's `checkpoint.pth.tar` when present, following the source default. Use `GPU_ID=1` for GPU 1; runtime overrides include `NUM_WORKERS=4`, `BATCH_SIZE=32`, and `MAX_STEPS=40000`.

## Test

```bash
bash run_test.sh levircd
bash run_test.sh levircdplus
bash run_test.sh cdd
bash run_test.sh whucd
bash run_test.sh sysucd
```

Metrics and prediction artifacts are written under `S2ENet_work_dirs/<key>/test/`.

## Parameters and FLOPs

After activating `s2enet-rs`, install the separate profiling dependency and profile one paired 256x256 input:

```bash
python -m pip install -r requirements_profile.txt
CUDA_VISIBLE_DEVICES=0 python profile_model.py --device cuda:0 \
  --output S2ENet_work_dirs/profile_256.json
```

The script disables only pretrained-weight loading during profiling; architecture and tensor dimensions remain unchanged. It reports total/trainable parameters and fvcore operations. Unsupported wavelet operators are listed; FLOPs are partial if any are reported.
