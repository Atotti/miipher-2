import itertools
import json
import pathlib

import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from miipher_2.accelerate_utils import build_accelerator, log_metrics, print_main
from miipher_2.data.webdataset_loader import CleanVocoderDataset  # CleanVocoderDataset をインポート
from miipher_2.extractors.hubert import HubertExtractor  # HubertExtractorを直接使う
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
from miipher_2.utils.audio import save


# collate_fn と AttrDict は train_hifigan.py と同じ
def collate_tensors(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    clean_16k_tensors, clean_22k_tensors = zip(*batch, strict=False)
    clean_16k_tensors = [x.squeeze(0) for x in clean_16k_tensors]
    clean_22k_tensors = [x.squeeze(0) for x in clean_22k_tensors]
    clean_16k_padded = pad_sequence(clean_16k_tensors, batch_first=True)
    clean_22k_padded = pad_sequence(clean_22k_tensors, batch_first=True)
    return clean_16k_padded, clean_22k_padded


class AttrDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


@torch.no_grad()
def validate(
    h: AttrDict,
    hubert_extractor: torch.nn.Module,
    prenet: torch.nn.Module,
    generator: torch.nn.Module,
    mpd: torch.nn.Module,
    msd: torch.nn.Module,
    val_dl: DataLoader,
    accelerator,
    cfg: DictConfig,
) -> dict:
    """HiFi-GAN Pre-training Validation with Accelerate"""
    hubert_extractor.eval()
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
    limit_batches = cfg.get("validation_batches")

    for i, (clean_16k, clean_22k) in enumerate(val_dl):
        if limit_batches is not None and i >= limit_batches:
            break

        with accelerator.autocast():
            if clean_22k.dim() == 2:
                clean_22k = clean_22k.unsqueeze(1)

            # Feature extraction and generation from CLEAN audio
            feat = hubert_extractor(clean_16k)
            y_g_hat_prenet = prenet(feat)
            y_g_hat = generator(y_g_hat_prenet)

            # Align length
            min_len = min(clean_22k.size(2), y_g_hat.size(2))
            clean_22k_aligned = clean_22k[:, :, :min_len]
            y_g_hat_aligned = y_g_hat[:, :, :min_len]

            # Mel L1 loss
            y_mel = mel_spectrogram(
                clean_22k_aligned.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
            )
            y_g_hat_mel = mel_spectrogram(
                y_g_hat_aligned.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
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

            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f

        # Gather losses from all processes
        loss_gen_all_gathered = accelerator.gather(loss_gen_all.unsqueeze(0))
        loss_disc_all_gathered = accelerator.gather(loss_disc_all.unsqueeze(0))
        loss_mel_gathered = accelerator.gather(loss_mel.unsqueeze(0))
        loss_fm_gathered = accelerator.gather((loss_fm_s + loss_fm_f).unsqueeze(0))
        loss_adv_gathered = accelerator.gather((loss_gen_s + loss_gen_f).unsqueeze(0))
        batch_size_gathered = accelerator.gather(torch.tensor(clean_16k.size(0), device=accelerator.device))

        # Accumulate losses
        if accelerator.is_main_process:
            batch_size = batch_size_gathered.sum().item()
            total_losses["generator_total"] += loss_gen_all_gathered.mean().item() * batch_size
            total_losses["discriminator_total"] += loss_disc_all_gathered.mean().item() * batch_size
            total_losses["mel_l1"] += loss_mel_gathered.mean().item() * batch_size
            total_losses["feature_matching"] += loss_fm_gathered.mean().item() * batch_size
            total_losses["generator_adv"] += loss_adv_gathered.mean().item() * batch_size
            total_count += batch_size

    # Calculate average losses
    if accelerator.is_main_process and total_count > 0:
        avg_losses = {f"val_loss/{key}": val / total_count for key, val in total_losses.items()}
    else:
        avg_losses = {}

    # Set models back to train mode, except for the frozen extractor
    prenet.train()
    generator.train()
    mpd.train()
    msd.train()

    return avg_losses


def pre_train_vocoder(cfg: DictConfig) -> None:  # noqa: PLR0912
    """HiFi-GAN Pre-training with Accelerate"""
    # Acceleratorの初期化
    accelerator = build_accelerator(cfg)

    print_main(accelerator, "Initializing HiFi-GAN pre-training with Accelerate")
    print_main(accelerator, f"Mixed precision: {accelerator.mixed_precision}")

    # データセットとデータローダー
    train_dataset = CleanVocoderDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle)
    val_dataset = CleanVocoderDataset(cfg.dataset.val_path_pattern, shuffle=False)

    train_dl = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=True,
        collate_fn=collate_tensors,
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=False,
        collate_fn=collate_tensors,
    )

    # --- モデルの構築 ---
    # 1. FeatureCleanerではなくHubertExtractorを直接使用
    hubert_extractor = HubertExtractor(
        model_name=cfg.model.hubert_model_name,
        layer=cfg.model.hubert_layer,
    ).eval()
    for param in hubert_extractor.parameters():
        param.requires_grad = False

    # 2. HiFi-GANモデル
    config_path = pathlib.Path(cfg.pretrained_gen).parent / "config.json"
    with config_path.open() as f:
        h_dict = json.load(f)
    h = AttrDict(h_dict)

    hubert_dim = hubert_extractor.hubert.config.hidden_size
    prenet = MHubertToMel(hubert_dim)
    generator = Generator(h)

    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()

    # オプティマイザー
    optim_g = optim.AdamW(
        itertools.chain(prenet.parameters(), generator.parameters()), lr=cfg.lr, betas=tuple(cfg.betas)
    )
    optim_d = optim.AdamW(itertools.chain(mpd.parameters(), msd.parameters()), lr=cfg.lr, betas=tuple(cfg.betas))

    # Accelerator.prepareで一括準備
    hubert_extractor, prenet, generator, mpd, msd, optim_g, optim_d, train_dl, val_dl = accelerator.prepare(
        hubert_extractor, prenet, generator, mpd, msd, optim_g, optim_d, train_dl, val_dl
    )

    # チェックポイントの読み込み
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
        # 事前学習済み重みの読み込み
        if cfg.get("pretrained_gen"):
            state_dict_g = torch.load(cfg.pretrained_gen, map_location="cpu")
            accelerator.unwrap_model(generator).load_state_dict(state_dict_g["generator"])
            print_main(accelerator, f"Loaded pre-trained Generator from: {cfg.pretrained_gen}")

    # データローダーのイテレーター作成
    train_iter = iter(train_dl)

    print_main(accelerator, "Starting HiFi-GAN pre-training")
    print_main(accelerator, f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")

    # 学習ループ
    for step in range(start_step, cfg.steps):
        # 検証の実行
        if step > 0 and step % cfg.validation_interval == 0:
            val_losses = validate(
                h, hubert_extractor, prenet, generator, mpd, msd, val_dl, accelerator, cfg
            )
            log_metrics(accelerator, val_losses, step)
            if val_losses:  # val_lossesが空でない場合のみログ出力
                print_main(
                    accelerator,
                    f"[Pre-train] Step:{step:>7} | "
                    f"Val Gen Loss: {val_losses['val_loss/generator_total']:.4f}, "
                    f"Val Disc Loss: {val_losses['val_loss/discriminator_total']:.4f}"
                )

        # データの取得
        try:
            clean_16k, clean_22k = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            clean_16k, clean_22k = next(train_iter)

        # Generator学習
        with accelerator.accumulate(generator):
            with accelerator.autocast():
                if clean_22k.dim() == 2:
                    clean_22k = clean_22k.unsqueeze(1)

                # HuBERT特徴量抽出（クリーン音声から）
                with torch.no_grad():
                    feat = hubert_extractor(clean_16k)

                y_g_hat_prenet = prenet(feat)
                y_g_hat = generator(y_g_hat_prenet)

                # 長さ合わせ
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

                # GAN損失
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
                "pretrain/train/generator_loss": loss_gen_all.item(),
                "pretrain/train/discriminator_loss": loss_disc_all.item(),
                "pretrain/train/mel_l1": loss_mel.item(),
                "pretrain/train/feature_matching": (loss_fm_s + loss_fm_f).item(),
                "pretrain/train/generator_adv": (loss_gen_s + loss_gen_f).item(),
                "pretrain/lr": optim_g.param_groups[0]['lr'],
            }
            log_metrics(accelerator, metrics, step)
            print_main(
                accelerator,
                f"[Pre-train] Step:{step:>7} | "
                f"Gen Loss={loss_gen_all.item():.4f} | "
                f"Disc Loss={loss_disc_all.item():.4f} | "
                f"Mel L1={loss_mel.item():.4f}",
            )

        # サンプル音声の生成と保存
        if step > 0 and step % cfg.get("audio_log_interval", 5000) == 0 and accelerator.is_main_process:
            with torch.no_grad():
                with accelerator.autocast():
                    sample_feat = hubert_extractor(clean_16k[:1])
                    sample_prenet = prenet(sample_feat)
                    sample_wav = generator(sample_prenet)[0].cpu()

                audio_path = pathlib.Path(cfg.save_dir) / f"sample_pretrain_{step}.wav"
                save(audio_path, sample_wav, sr=h.sampling_rate)

                if cfg.wandb.get("log_audio", False):
                    import wandb
                    log_metrics(accelerator, {
                        "pretrain/audio/generated_sample": wandb.Audio(str(audio_path), sample_rate=h.sampling_rate)
                    }, step)

        # チェックポイントの保存
        if step > 0 and step % cfg.checkpoint.save_interval == 0:
            print_main(accelerator, f"Saving checkpoint at step {step}")
            accelerator.save_state(output_dir=cfg.save_dir, safe_serialization=False)
            if accelerator.is_main_process:
                (pathlib.Path(cfg.save_dir) / "step.txt").write_text(str(step))

    # 学習終了時の処理
    print_main(accelerator, "HiFi-GAN pre-training completed")
    if accelerator.is_main_process:
        accelerator.save_state(output_dir=cfg.save_dir, safe_serialization=False)
        accelerator.end_training()
