import itertools
import json
import pathlib
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from miipher_2.accelerate_utils import build_accelerator, log_metrics, print_main, setup_random_seeds
from miipher_2.data.webdataset_loader import VocoderDataset
from miipher_2.hifigan.meldataset import mel_spectrogram
from miipher_2.hifigan.models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from miipher_2.hifigan.prenet import MHubertToMel
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.utils.audio import save


def collate_tensors(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    バッチ内の音声テンソルの長さをパディングして揃える関数
    """
    noisy_tensors, clean_tensors = zip(*batch, strict=False)

    # (1, T) -> (T) のように次元を1つ削除
    noisy_tensors = [x.squeeze(0) for x in noisy_tensors]
    clean_tensors = [x.squeeze(0) for x in clean_tensors]

    # pad_sequenceで長さを揃え、(B, T_max) のテンソルに変換
    noisy_padded = pad_sequence(noisy_tensors, batch_first=True)
    clean_padded = pad_sequence(clean_tensors, batch_first=True)

    return noisy_padded, clean_padded


# 公式Generatorが要求するAttrDict形式のためのヘルパークラス
class AttrDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


@torch.no_grad()
def validate(
    h: AttrDict,
    cleaner: torch.nn.Module,
    prenet: torch.nn.Module,
    generator: torch.nn.Module,
    mpd: torch.nn.Module,
    msd: torch.nn.Module,
    val_dl: DataLoader,
    accelerator: Any,
    cfg: DictConfig,
) -> dict[str, float]:
    """HiFi-GAN検証関数（Accelerate対応版）"""
    cleaner.eval()
    prenet.eval()
    generator.eval()
    mpd.eval()
    msd.eval()

    total_losses = {
        "generator_total": 0.0,
        "discriminator_total": 0.0,
        "mel_l1": 0.0,
        "feature_matching": 0.0,
        "generator_adv": 0.0,
    }
    total_count = 0

    limit_batches = cfg.get("validation_batches", None)

    for i, (noisy_16k, clean_22k) in enumerate(val_dl):
        if limit_batches is not None and i >= limit_batches:
            break

        with accelerator.autocast():
            if clean_22k.dim() == 2:
                clean_22k = clean_22k.unsqueeze(1)

            # Feature extraction and generation
            feat = cleaner(noisy_16k)
            y_g_hat_prenet = prenet(feat)
            y_g_hat = generator(y_g_hat_prenet)

            # Align length
            min_len = min(clean_22k.size(2), y_g_hat.size(2))
            clean_22k = clean_22k[:, :, :min_len]
            y_g_hat = y_g_hat[:, :, :min_len]

            # Mel L1 loss
            y_mel = mel_spectrogram(
                clean_22k.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
            )
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
            )
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            # GAN Loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(clean_22k, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(clean_22k, y_g_hat)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f

        # Accelerateのgatherを使用してマルチGPUでの損失を集約
        all_losses = accelerator.gather(torch.stack([
            loss_gen_all, loss_disc_all, loss_mel,
            loss_fm_s + loss_fm_f, loss_gen_s + loss_gen_f
        ]))
        batch_sizes = accelerator.gather(torch.tensor(noisy_16k.size(0), device=accelerator.device))

        total_losses["generator_total"] += all_losses[0].sum().item()
        total_losses["discriminator_total"] += all_losses[1].sum().item()
        total_losses["mel_l1"] += all_losses[2].sum().item()
        total_losses["feature_matching"] += all_losses[3].sum().item()
        total_losses["generator_adv"] += all_losses[4].sum().item()
        total_count += batch_sizes.sum().item()

    avg_losses = {f"val_loss/{key}": val / total_count for key, val in total_losses.items()}

    # Set models back to train mode, except for the frozen cleaner
    prenet.train()
    generator.train()
    mpd.train()
    msd.train()
    # Note: cleaner stays in eval mode since it's frozen and used for feature extraction

    return avg_losses


def train_hifigan(cfg: DictConfig) -> None:
    """
    Accelerateベースの並列HiFi-GAN学習
    """
    # Acceleratorとロガーの初期化
    accelerator, logger = build_accelerator(cfg)

    # 乱数シードの設定
    setup_random_seeds(accelerator, cfg.get("seed", 42))

    # データローダーの準備
    train_ds = VocoderDataset(
        cfg.dataset.path_pattern,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.loader.num_workers  # マルチワーカー対応
    )
    val_ds = VocoderDataset(
        cfg.dataset.val_path_pattern,
        shuffle=False,
        num_workers=cfg.loader.num_workers  # マルチワーカー対応
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=collate_tensors,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        collate_fn=collate_tensors,
        drop_last=False,
    )

    # 特徴量クリーナー (Adapter適用済みHuBERT)の準備
    cleaner = FeatureCleaner(cfg.model).eval()
    adapter_checkpoint = torch.load(cfg.adapter_ckpt, map_location="cpu", weights_only=False)
    cleaner.load_state_dict(adapter_checkpoint["model_state_dict"])
    for param in cleaner.parameters():
        param.requires_grad = False

    # HiFi-GANモデルの構築
    with (pathlib.Path(cfg.pretrained_gen).parent / "config.json").open() as f:
        h_dict = json.load(f)
    h = AttrDict(h_dict)

    # miipher-2のprenetと公式Generatorを接続
    hubert_dim = cleaner.extractor.hubert.config.hidden_size
    prenet = MHubertToMel(hubert_dim)

    # 次に公式Generator
    generator = Generator(h)

    # Discriminators
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()

    # --- 最適化アルゴリズム ---
    # prenetとgeneratorのパラメータを一緒に最適化
    optim_g = optim.AdamW(
        itertools.chain(prenet.parameters(), generator.parameters()), lr=cfg.lr, betas=tuple(cfg.betas)
    )
    optim_d = optim.AdamW(itertools.chain(mpd.parameters(), msd.parameters()), lr=cfg.lr, betas=tuple(cfg.betas))

    # Accelerator.prepareで一括準備
    cleaner, prenet, generator, mpd, msd, optim_g, optim_d, train_dl, val_dl = accelerator.prepare(
        cleaner, prenet, generator, mpd, msd, optim_g, optim_d, train_dl, val_dl
    )

    # --- 事前学習済み重みとチェックポイントの読み込み ---
    start_step = 0
    checkpoint_dir = pathlib.Path(cfg.save_dir)
    if (checkpoint_dir / "pytorch_model.bin").exists():
        print_main(accelerator, f"Loading checkpoint from {checkpoint_dir}")
        accelerator.load_state(checkpoint_dir)
        step_file = checkpoint_dir / "step.txt"
        if step_file.exists():
            start_step = int(step_file.read_text()) + 1
            print_main(accelerator, f"Resuming from step {start_step}")
    else:
        # 新規学習時はステージ1で事前学習したモデルを読み込む
        if cfg.get("pretrained_gen"):
            pretrain_ckpt = torch.load(cfg.pretrained_gen, map_location="cpu", weights_only=False)
            # Accelerateでunwrapしてから状態を読み込み
            accelerator.unwrap_model(prenet).load_state_dict(pretrain_ckpt["prenet"])
            accelerator.unwrap_model(generator).load_state_dict(pretrain_ckpt["generator"])
            print_main(accelerator, f"Loaded pre-trained vocoder from: {cfg.pretrained_gen}")

    # データローダーのイテレーター作成
    train_iter = iter(train_dl)

    print_main(accelerator, "Starting HiFi-GAN training with Accelerate")
    print_main(accelerator, f"Mixed precision: {accelerator.mixed_precision}")
    print_main(accelerator, f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")

    # 学習ループ
    for step in range(start_step, cfg.steps):
        # 検証の実行
        if step > 0 and step % cfg.validation_interval == 0:
            val_losses = validate(
                h, cleaner, prenet, generator, mpd, msd, val_dl, accelerator, cfg
            )
            log_metrics(accelerator, val_losses, step)
            print_main(
                accelerator,
                f"[HiFi-GAN] Step:{step:>7} | "
                f"Val Gen Loss: {val_losses['val_loss/generator_total']:.4f}, "
                f"Val Disc Loss: {val_losses['val_loss/discriminator_total']:.4f}"
            )

        # データの取得
        try:
            noisy_16k, clean_22k = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            noisy_16k, clean_22k = next(train_iter)

        # Generator学習
        with accelerator.accumulate(generator):
            with accelerator.autocast():
                # (B, 1, T)の形式に
                if clean_22k.dim() == 2:
                    clean_22k = clean_22k.unsqueeze(1)

                # HuBERT特徴量抽出
                with torch.no_grad():
                    feat = cleaner(noisy_16k)

                # 音声生成
                y_g_hat = prenet(feat)
                y_g_hat = generator(y_g_hat)

                # Mel-Spectrogram Loss
                min_len = min(clean_22k.size(2), y_g_hat.size(2))
                clean_22k_aligned = clean_22k[:, :, :min_len]
                y_g_hat_aligned = y_g_hat[:, :, :min_len]

                # メルスペクトログラム損失
                y_mel = mel_spectrogram(
                    clean_22k_aligned.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                    h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
                )
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat_aligned.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                    h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
                )
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                # GAN Loss
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(clean_22k_aligned, y_g_hat_aligned)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(clean_22k_aligned, y_g_hat_aligned)

                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, _ = generator_loss(y_df_hat_g)
                loss_gen_s, _ = generator_loss(y_ds_hat_g)

                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            # Generator逆伝播
            accelerator.backward(loss_gen_all)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(itertools.chain(prenet.parameters(), generator.parameters()), max_norm=cfg.get("max_grad_norm", 1.0))

            optim_g.step()
            optim_g.zero_grad()

        # Discriminator学習
        with accelerator.accumulate(mpd):
            with accelerator.autocast():
                # Discriminatorの損失（detach済みのGeneratorの出力を使用）
                y_df_hat_r, y_df_hat_g, _, _ = mpd(clean_22k_aligned, y_g_hat_aligned.detach())
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(clean_22k_aligned, y_g_hat_aligned.detach())

                loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
                loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
                loss_disc_all = loss_disc_s + loss_disc_f

            # Discriminator逆伝播
            accelerator.backward(loss_disc_all)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(itertools.chain(mpd.parameters(), msd.parameters()), max_norm=cfg.get("max_grad_norm", 1.0))

            optim_d.step()
            optim_d.zero_grad()

        # ログ出力
        if step % cfg.log_interval == 0:
            metrics = {
                "train/generator_loss": loss_gen_all.item(),
                "train/discriminator_loss": loss_disc_all.item(),
                "train/mel_l1": loss_mel.item(),
                "train/feature_matching": (loss_fm_s + loss_fm_f).item(),
                "train/generator_adv": (loss_gen_s + loss_gen_f).item(),
                "lr": optim_g.param_groups[0]['lr'],
            }
            log_metrics(accelerator, metrics, step)
            print_main(
                accelerator,
                f"[HiFi-GAN] Step:{step:>7} | "
                f"Gen Loss={loss_gen_all.item():.4f} | "
                f"Disc Loss={loss_disc_all.item():.4f} | "
                f"Mel L1={loss_mel.item():.4f}",
            )

        # サンプル音声の生成と保存
        if step > 0 and step % cfg.get("audio_log_interval", 5000) == 0 and accelerator.is_main_process:
            with torch.no_grad():
                with accelerator.autocast():
                    sample_feat = cleaner(noisy_16k[:1])
                    sample_prenet = prenet(sample_feat)
                    sample_wav = generator(sample_prenet)[0].cpu()

                audio_path = pathlib.Path(cfg.save_dir) / f"sample_hifigan_{step}.wav"
                save(audio_path, sample_wav, sr=h.sampling_rate)

                if cfg.wandb.get("log_audio", False):
                    import wandb
                    log_metrics(accelerator, {
                        "audio/generated_sample": wandb.Audio(str(audio_path), sample_rate=h.sampling_rate)
                    }, step)

        # チェックポイントの保存
        if step > 0 and step % cfg.checkpoint.save_interval == 0:
            print_main(accelerator, f"Saving checkpoint at step {step}")
            accelerator.save_state(output_dir=cfg.save_dir, safe_serialization=False)
            if accelerator.is_main_process:
                (pathlib.Path(cfg.save_dir) / "step.txt").write_text(str(step))

    # 学習終了時の処理
    print_main(accelerator, "HiFi-GAN training completed")
    if accelerator.is_main_process:
        accelerator.save_state(output_dir=cfg.save_dir, safe_serialization=False)
        accelerator.end_training()
