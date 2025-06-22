#!/bin/bash

set -e

cd /home/ayu/datasets
cd fleurs-r/data

find . -type f -path "*/audio/*.tar.gz" -print0 |
while IFS= read -r -d '' tgz; do
  aud_dir=$(dirname "$tgz")                 # .../audio
  split=$(basename "$tgz" .tar.gz)          # train / dev / test
  out_dir="$aud_dir/$split"                 # .../audio/train など
  mkdir -p "$out_dir"
  echo "⇢ extracting $tgz → $out_dir"
  tar -xzf "$tgz" -C "$out_dir"             # 展開
done
