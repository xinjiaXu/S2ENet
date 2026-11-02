#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_id="${GPU_ID:-0}"
workers="${NUM_WORKERS:-4}"
batch_size="${BATCH_SIZE:-32}"
max_steps="${MAX_STEPS:-40000}"
lr="${LR:-5e-4}"
input_size="${INPUT_SIZE:-256}"

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd}\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) name="levircd"; root="$data_root/LEVIR_CD"; image_a=A; image_b=B; label=label ;;
  levircdplus) name="levircdplus"; root="$data_root/LEVIR_CD_PLUS/cut_data"; image_a=A; image_b=B; label=label ;;
  cdd) name="cdd"; root="$data_root/CDD"; image_a=A; image_b=B; label=OUT ;;
  whucd) name="whucd"; root="$data_root/WHUCD"; image_a=A; image_b=B; label=label ;;
  sysucd) name="sysucd"; root="$data_root/SYSU_CD"; image_a=time1; image_b=time2; label=label ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

for split in train val test; do
  for folder in "$image_a" "$image_b" "$label"; do
    [[ -d "$root/$split/$folder" ]] || { printf 'Missing data directory: %s\n' "$root/$split/$folder" >&2; exit 1; }
  done
done

cd "$repo_root"
CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
  --dataset-root "$root" --dataset-name "$name" \
  --image-dir-a "$image_a" --image-dir-b "$image_b" --label-dir "$label" \
  --output-dir "$repo_root/S2ENet_work_dirs" --onGPU true \
  --num_workers "$workers" --batch_size "$batch_size" \
  --max_steps "$max_steps" --lr "$lr" --inWidth "$input_size" --inHeight "$input_size"
