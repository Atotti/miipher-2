import itertools
import json
import os
import pathlib

import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import wandb
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
from miipher_2.hifigan.prenet import Miipher2PreNet
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.utils.audio import save


def collate_tensors(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    noisy_tensors, clean_tensors = zip(*batch, strict=False)
    noisy_tensors = [x.squeeze(0) for x in noisy_tensors]
    clean_tensors = [x.squeeze(0) for x in clean_tensors]
    noisy_padded = pad_sequence(noisy_tensors, batch_first=True, padding_value=0.0)
    clean_padded = pad_sequence(clean_tensors, batch_first=True, padding_value=0.0)
    return noisy_padded, clean_padded


class AttrDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


@torch.no_grad()
def validate(
    h: AttrDict,
    accelerator: Accelerator,
    cleaner: torch.nn.Module,
    prenet: torch.nn.Module,
    generator: torch.nn.Module,
    mpd: torch.nn.Module,
    msd: torch.nn.Module,
    val_dl: DataLoader,
    limit_batches: int | None = None,
) -> dict:
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

    for i, (noisy_16k, clean_22k) in enumerate(val_dl):
        if limit_batches is not None and i >= limit_batches:
            break
        if clean_22k.dim() == 2:
            clean_22k = clean_22k.unsqueeze(1)

        with torch.no_grad():
            feat = cleaner(noisy_16k)
        y_g_hat_prenet = prenet(feat)
        y_g_hat = generator(y_g_hat_prenet)

        min_len = min(clean_22k.size(2), y_g_hat.size(2))
        clean_22k, y_g_hat = clean_22k[:, :, :min_len], y_g_hat[:, :, :min_len]

        y_mel = mel_spectrogram(
            clean_22k.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
        )
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
        )
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

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

        # 集計
        losses_g = accelerator.gather_for_metrics({"generator_total": loss_gen_all})
        losses_d = accelerator.gather_for_metrics({"discriminator_total": loss_disc_all})

        total_losses["generator_total"] += losses_g["generator_total"].sum().item()
        total_losses["discriminator_total"] += losses_d["discriminator_total"].sum().item()
        total_count += noisy_16k.size(0) * accelerator.num_processes

    avg_losses = {f"val_loss/{key}": val / total_count for key, val in total_losses.items()}

    cleaner.train()
    prenet.train()
    generator.train()
    mpd.train()
    msd.train()

    return avg_losses


def train_hifigan(cfg: DictConfig) -> None:
    checkpointing_dir = pathlib.Path(cfg.save_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs],
    )

    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs={"entity": cfg.wandb.entity, "name": cfg.wandb.name, "tags": cfg.wandb.tags},
    )

    dl = DataLoader(
        VocoderDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=True,
        collate_fn=collate_tensors,
    )
    val_dl = DataLoader(
        VocoderDataset(cfg.dataset.val_path_pattern, shuffle=False),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=False,
        collate_fn=collate_tensors,
    )

    # モデル構築
    cleaner = FeatureCleaner(cfg.model).eval()
    adapter_checkpoint = torch.load(cfg.adapter_ckpt, map_location="cpu", weights_only=False)
    cleaner.load_state_dict(adapter_checkpoint["model_state_dict"])
    for param in cleaner.parameters():
        param.requires_grad = False

    with (pathlib.Path(cfg.pretrained_gen).parent / "config.json").open() as f:
        h = AttrDict(json.load(f))
    h.sampling_rate = 22050
    h.fmax_for_loss = 8000.0

    hubert_dim = cleaner.extractor.hubert.config.hidden_size
    prenet = Miipher2PreNet(in_dim=hubert_dim)
    generator = Generator(h)
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()

    optim_g = optim.AdamW(
        itertools.chain(prenet.parameters(), generator.parameters()), lr=cfg.lr, betas=tuple(cfg.betas)
    )
    optim_d = optim.AdamW(itertools.chain(mpd.parameters(), msd.parameters()), lr=cfg.lr, betas=tuple(cfg.betas))

    # accelerator.prepare
    cleaner, prenet, generator, mpd, msd, optim_g, optim_d, dl, val_dl = accelerator.prepare(
        cleaner, prenet, generator, mpd, msd, optim_g, optim_d, dl, val_dl
    )

    # チェックポイントからの再開または事前学習済み重みのロード
    resume_from_checkpoint = cfg.checkpoint.get("resume_from")
    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            resume_from_checkpoint = max(checkpointing_dir.glob("checkpoint_*"), key=os.path.getmtime)
        accelerator.print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint)
        path = pathlib.Path(resume_from_checkpoint)
        training_basename = path.stem.split("_")[-1]
        start_step = int(training_basename.replace("k", "")) * 1000 + 1
    else:
        # 新規学習時は事前学習Vocoderをロード
        pretrain_ckpt = torch.load(cfg.pretrained_gen, map_location="cpu")
        accelerator.unwrap_model(prenet).load_state_dict(pretrain_ckpt["prenet"])
        accelerator.unwrap_model(generator).load_state_dict(pretrain_ckpt["generator"])
        accelerator.print(f"Loaded pre-trained vocoder from: {cfg.pretrained_gen}")
        start_step = 0

    # 学習ループ
    dl_iter = iter(dl)
    for step in range(start_step, cfg.steps):
        if (
            accelerator.is_main_process
            and hasattr(cfg, "validation_interval")
            and step > 0
            and step % cfg.validation_interval == 0
        ):
            val_losses = validate(
                h,
                accelerator,
                cleaner,
                prenet,
                generator,
                mpd,
                msd,
                val_dl,
                limit_batches=cfg.get("validation_batches"),
            )
            accelerator.log(val_losses, step=step)
            accelerator.print(
                f"[Step {step:>7d}/{cfg.steps}] "
                f"Val Gen Loss: {val_losses['val_loss/generator_total']:.4f}, "
                f"Val Disc Loss: {val_losses['val_loss/discriminator_total']:.4f}"
            )

        # データ取得
        try:
            noisy_16k, clean_22k = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            noisy_16k, clean_22k = next(dl_iter)

        if clean_22k.dim() == 2:
            clean_22k = clean_22k.unsqueeze(1)

        # Generator学習
        with accelerator.accumulate(prenet, generator):
            optim_g.zero_grad()
            with torch.no_grad():
                feat = cleaner(noisy_16k)
            y_g_hat_prenet = prenet(feat)
            y_g_hat = generator(y_g_hat_prenet)

            min_len = min(clean_22k.size(2), y_g_hat.size(2))
            clean_22k_seg, y_g_hat_seg = clean_22k[:, :, :min_len], y_g_hat[:, :, :min_len]

            y_mel = mel_spectrogram(
                clean_22k_seg.squeeze(1),
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.fmax_for_loss,
            )
            y_g_hat_mel = mel_spectrogram(
                y_g_hat_seg.squeeze(1),
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.fmax_for_loss,
            )
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(clean_22k_seg, y_g_hat_seg)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(clean_22k_seg, y_g_hat_seg)
            loss_fm_f, loss_fm_s = feature_loss(fmap_f_r, fmap_f_g), feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            accelerator.backward(loss_gen_all)
            optim_g.step()

        # Discriminator学習
        with accelerator.accumulate(mpd, msd):
            optim_d.zero_grad()
            y_g_hat_detached = y_g_hat_seg.detach()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(clean_22k_seg, y_g_hat_detached)
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(clean_22k_seg, y_g_hat_detached)
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f
            accelerator.backward(loss_disc_all)
            optim_d.step()

        # ログとチェックポイント
        if accelerator.is_main_process and (step % cfg.log_interval == 0):
            log_data = {
                "loss/generator_total": loss_gen_all.item(),
                "loss/discriminator_total": loss_disc_all.item(),
                "learning_rate": optim_g.param_groups[0]["lr"],
            }
            accelerator.log(log_data, step=step)
            accelerator.print(
                f"[Step {step:>7d}/{cfg.steps}] Gen Loss: {loss_gen_all.item():.4f}, Disc Loss: {loss_disc_all.item():.4f}"
            )

        if step > 0 and step % cfg.checkpoint.save_interval == 0:
            output_dir = checkpointing_dir / f"checkpoint_{step // 1000}k"
            accelerator.save_state(output_dir)

            # メインプロセスで音声サンプルをログ
            if accelerator.is_main_process and cfg.wandb.enabled and cfg.wandb.log_audio:
                sample_path = checkpointing_dir / f"sample_{step}.wav"
                save(sample_path, y_g_hat[0:1].squeeze().cpu(), sr=h.sampling_rate)
                accelerator.log(
                    {"audio/generated_sample": wandb.Audio(str(sample_path), sample_rate=h.sampling_rate)}, step=step
                )

    # 最終モデル保存
    if accelerator.is_main_process:
        final_model_path = checkpointing_dir / "hifigan_final.pt"
        unwrapped_prenet = accelerator.unwrap_model(prenet)
        unwrapped_generator = accelerator.unwrap_model(generator)
        torch.save(
            {"prenet": unwrapped_prenet.state_dict(), "generator": unwrapped_generator.state_dict()},
            final_model_path,
        )
        accelerator.print(f"Final model saved to {final_model_path}")

    accelerator.end_training()
