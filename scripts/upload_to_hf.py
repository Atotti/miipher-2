#!/usr/bin/env python3
"""
Hugging Face Hubにmiipher-2モデルをアップロードするスクリプト
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
import argparse

def setup_model_repo(model_dir: Path, repo_id: str):
    """モデルリポジトリをセットアップしてアップロード"""
    
    # Hugging Face APIを初期化
    api = HfApi()
    
    print(f"Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print("✅ Repository created/verified")
    except Exception as e:
        print(f"Repository creation: {e}")
    
    # モデルフォルダをアップロード
    print(f"Uploading model files from {model_dir}")
    try:
        upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Miipher-2 complete model (Adapter + Vocoder)"
        )
        print("✅ Model uploaded successfully!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Upload Miipher-2 model to Hugging Face Hub")
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default="models/miipher2",
        help="Path to model directory"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1",
        help="Hugging Face repository ID"
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Check if required files exist before upload"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # モデルディレクトリの存在確認
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    # 必要なファイルの確認
    required_files = [
        "config.json",
        "README.md", 
        "checkpoint_199k_fixed.pt",
        "epoch=77-step=137108.ckpt"
    ]
    
    print("📋 Checking required files...")
    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {file} ({size_mb:.1f}MB)")
        else:
            print(f"  ❌ {file} (missing)")
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        if not args.check_files:
            print("Use --check-files to only check without uploading")
            return
    
    if args.check_files:
        print("📋 File check completed")
        return
    
    # Hugging Face tokenの確認
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN environment variable not set")
        print("Please run: export HF_TOKEN=your_token_here")
        return
    
    print(f"🚀 Starting upload to {args.repo_id}")
    success = setup_model_repo(model_dir, args.repo_id)
    
    if success:
        print(f"🎉 Model successfully uploaded to: https://huggingface.co/{args.repo_id}")
    else:
        print("❌ Upload failed")

if __name__ == "__main__":
    main()