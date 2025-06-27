#!/bin/bash

set -e  # エラー時に即停止

BASE_URL="https://www.openslr.org/resources/141"
FILES=(
  "doc.tar.gz"
  "dev_clean.tar.gz"
  "dev_other.tar.gz"
  "test_clean.tar.gz"
  "test_other.tar.gz"
  "train_clean_100.tar.gz"
  "train_clean_360.tar.gz"
  "train_other_500.tar.gz"
  "libritts_r_failed_speech_restoration_examples.tar.gz"
  "md5sum.txt"
)

# 保存先ディレクトリを作成
cd /home/ayu/datasets
mkdir -p libritts_r
cd libritts_r

# ダウンロード
for FILE in "${FILES[@]}"; do
  echo "Downloading $FILE..."
  curl -L -O "$BASE_URL/$FILE"
done

# チェックサム検証
echo "Verifying checksums..."
if command -v md5sum &> /dev/null; then
  md5sum -c md5sum.txt
elif command -v md5 &> /dev/null; then
  while read -r CHECKSUM FILE; do
    echo "$CHECKSUM  $FILE" | md5 -r -c -
  done < md5sum.txt
else
  echo "Error: md5sum Not found."
  exit 1
fi

echo "All files downloaded and verified successfully."
