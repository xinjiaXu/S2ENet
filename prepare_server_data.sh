#!/usr/bin/env bash
set -euo pipefail

data_root="${1:-/home/xxj/Data}"

check_dataset() {
  local name="$1" root="$2" image_a="$3" image_b="$4" label="$5"
  for split in train val test; do
    for folder in "$image_a" "$image_b" "$label"; do
      [[ -d "$root/$split/$folder" ]] || { printf 'Missing %s: %s\n' "$name" "$root/$split/$folder" >&2; exit 1; }
    done
  done
  printf 'Found %s at %s\n' "$name" "$root"
}

check_dataset LEVIR-CD "$data_root/LEVIR_CD" A B label
check_dataset LEVIR-CD-PLUS "$data_root/LEVIR_CD_PLUS/cut_data" A B label
check_dataset CDD "$data_root/CDD" A B OUT
check_dataset WHU-CD "$data_root/WHUCD" A B label
check_dataset SYSU-CD "$data_root/SYSU_CD" time1 time2 label
