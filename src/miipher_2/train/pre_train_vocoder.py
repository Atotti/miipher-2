import itertools
import json
import pathlib

import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import get_scheduler

import wandb
from miipher_2.data.webdataset_loader import CleanVocoderDataset
from miipher_2.extractors.hubert import HubertExtractor
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
from miipher_2.utils.audio import save
from miipher_2.utils.checkpoint import (
    get_resume_checkpoint_path,
    load_checkpoint,
    restore_random_states,
    save_checkpoint,
    setup_wandb_resume,
)
from miipher_2.utils.ema import EMA


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
    ema_g: EMA,
    mpd: torch.nn.Module,
    msd: torch.nn.Module,
    val_dl: DataLoader,
    cfg: DictConfig,
    limit_batches: int | None = None,
) -> dict:
    """HiFi-GAN Pre-training Validation with EMA"""
    hubert_extractor.eval()
    prenet.eval()
    generator.eval()
    mpd.eval()
    msd.eval()

    # Apply EMA weights for validation
    ema_g.apply_shadow()

    total_losses = {
        "generator_total": 0.0,
        "discriminator_total": 0.0,
        "mel_l1": 0.0,
        "feature_matching": 0.0,
        "generator_adv": 0.0,
    }
    total_count = 0

    for i, (clean_16k, clean_22k) in enumerate(val_dl):
        if limit_batches is not None and i >= limit_batches:
            break
        clean_16k, clean_22k = clean_16k.cuda(), clean_22k.cuda()
        if clean_22k.dim() == 2:
            clean_22k = clean_22k.unsqueeze(1)

        feat = hubert_extractor(clean_16k)
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
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * cfg.lambda_mel

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(clean_22k, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(clean_22k, y_g_hat)

        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)

        loss_adv = (loss_gen_s + loss_gen_f) * cfg.lambda_adv
        loss_fm = (loss_fm_f + loss_fm_s) * cfg.lambda_fm
        loss_gen_all = loss_adv + loss_fm + loss_mel

        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_disc_all = loss_disc_s + loss_disc_f

        batch_size = clean_16k.size(0)
        total_losses["generator_total"] += loss_gen_all.item() * batch_size
        total_losses["discriminator_total"] += loss_disc_all.item() * batch_size
        total_losses["mel_l1"] += loss_mel.item() * batch_size
        total_losses["feature_matching"] += loss_fm.item() * batch_size
        total_losses["generator_adv"] += loss_adv.item() * batch_size
        total_count += batch_size

    avg_losses = {f"pretrain/val_loss/{key}": val / total_count for key, val in total_losses.items()}

    # Restore original weights
    ema_g.restore()

    prenet.train()
    generator.train()
    mpd.train()
    msd.train()

    return avg_losses


def pre_train_vocoder(cfg: DictConfig) -> None:  # noqa: PLR0912
    resume_checkpoint_path = get_resume_checkpoint_path(cfg)
    resumed_checkpoint = load_checkpoint(str(resume_checkpoint_path)) if resume_checkpoint_path else None
    if resumed_checkpoint:
        restore_random_states(resumed_checkpoint)
        print(f"[INFO] Resuming from step {resumed_checkpoint['steps']}")

    # WandB初期化
    if resumed_checkpoint and "wandb_run_id" in resumed_checkpoint:
        wandb_id = resumed_checkpoint["wandb_run_id"]
        setup_wandb_resume(cfg, resumed_checkpoint)
    else:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb_id = wandb.run.id

    dl = DataLoader(
        CleanVocoderDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=True,
        collate_fn=collate_tensors,
    )
    val_dl = DataLoader(
        CleanVocoderDataset(cfg.dataset.val_path_pattern, shuffle=False),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=False,
        collate_fn=collate_tensors,
    )

    hubert_extractor = (
        HubertExtractor(model_name=cfg.model.hubert_model_name, layer=cfg.model.hubert_layer).cuda().eval()
    )
    for param in hubert_extractor.parameters():
        param.requires_grad = False

    config_path = pathlib.Path(cfg.pretrained_gen).parent / "config.json"
    with config_path.open() as f:
        h = AttrDict(json.load(f))

    hubert_dim = hubert_extractor.hubert.config.hidden_size
    generator_modules = torch.nn.ModuleDict(
        {"prenet": Miipher2PreNet(in_dim=hubert_dim), "generator": Generator(h)}
    ).cuda()

    mpd = MultiPeriodDiscriminator().cuda()
    msd = MultiScaleDiscriminator().cuda()

    optim_g = optim.AdamW(generator_modules.parameters(), lr=cfg.optim.lr, betas=tuple(cfg.optim.betas))
    optim_d = optim.AdamW(
        itertools.chain(mpd.parameters(), msd.parameters()), lr=cfg.optim.lr, betas=tuple(cfg.optim.betas)
    )

    scheduler_g = get_scheduler(
        name=cfg.optim.scheduler.name,
        optimizer=optim_g,
        num_warmup_steps=cfg.optim.scheduler.warmup_steps,
        num_training_steps=cfg.steps,
    )
    scheduler_d = get_scheduler(
        name=cfg.optim.scheduler.name,
        optimizer=optim_d,
        num_warmup_steps=cfg.optim.scheduler.warmup_steps,
        num_training_steps=cfg.steps,
    )
    scaler = GradScaler(enabled=True)

    ema_g = EMA(generator_modules, decay=cfg.optim.ema_decay)

    start_step = 0
    if resumed_checkpoint:
        generator_modules.load_state_dict(resumed_checkpoint["generator_modules"])
        mpd.load_state_dict(resumed_checkpoint["mpd"])
        msd.load_state_dict(resumed_checkpoint["msd"])
        optim_g.load_state_dict(resumed_checkpoint["optim_g"])
        optim_d.load_state_dict(resumed_checkpoint["optim_d"])
        if "scheduler_g" in resumed_checkpoint:
            scheduler_g.load_state_dict(resumed_checkpoint["scheduler_g"])
        if "scheduler_d" in resumed_checkpoint:
            scheduler_d.load_state_dict(resumed_checkpoint["scheduler_d"])
        if "ema_g" in resumed_checkpoint:
            ema_g.load_state_dict(resumed_checkpoint["ema_g"])
        else:
            ema_g.register()
        start_step = resumed_checkpoint["steps"] + 1
    else:
        state_dict_g = torch.load(cfg.pretrained_gen, map_location="cpu")
        generator_modules.generator.load_state_dict(state_dict_g["generator"])
        ema_g.register()
        print(f"[INFO] Loaded pre-trained Generator from: {cfg.pretrained_gen}")

    dl_iter = iter(dl)
    for step in range(start_step, cfg.steps):
        if hasattr(cfg, "validation_interval") and step > 0 and step % cfg.validation_interval == 0:
            val_losses = validate(
                h,
                hubert_extractor,
                generator_modules.prenet,
                generator_modules.generator,
                ema_g,
                mpd,
                msd,
                val_dl,
                cfg,
                limit_batches=cfg.get("validation_batches"),
            )
            if cfg.wandb.enabled:
                wandb.log(val_losses, step=step)
            print(
                f"[Pre-train Step {step:>7d}/{cfg.steps}] Val Gen Loss: {val_losses['pretrain/val_loss/generator_total']:.4f}, Val Disc Loss: {val_losses['pretrain/val_loss/discriminator_total']:.4f}"
            )

        try:
            clean_16k, clean_22k = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            clean_16k, clean_22k = next(dl_iter)

        clean_16k, clean_22k = clean_16k.cuda(), clean_22k.cuda()
        if clean_22k.dim() == 2:
            clean_22k = clean_22k.unsqueeze(1)

        with torch.no_grad():
            feat = hubert_extractor(clean_16k)
        with autocast(enabled=True):
            y_g_hat = generator_modules.generator(generator_modules.prenet(feat))

        min_len = min(clean_22k.size(2), y_g_hat.size(2))
        clean_22k_prepared, y_g_hat_prepared = clean_22k[:, :, :min_len], y_g_hat[:, :, :min_len]

        # --- Discriminatorの学習 ---
        optim_d.zero_grad()
        with autocast(enabled=True):
            clean_22k_prepared.requires_grad = True
            y_g_hat_detached = y_g_hat_prepared.detach()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(clean_22k_prepared, y_g_hat_detached)
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(clean_22k_prepared, y_g_hat_detached)
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            grad_real_mpd = torch.autograd.grad(
                outputs=sum(torch.sum(o) for o in y_df_hat_r), inputs=clean_22k_prepared, create_graph=True
            )[0]
            grad_real_msd = torch.autograd.grad(
                outputs=sum(torch.sum(o) for o in y_ds_hat_r), inputs=clean_22k_prepared, create_graph=True
            )[0]
            r1_penalty = (
                0.5
                * cfg.optim.r1_lambda
                * (
                    grad_real_mpd.pow(2).view(grad_real_mpd.size(0), -1).sum(1).mean()
                    + grad_real_msd.pow(2).view(grad_real_msd.size(0), -1).sum(1).mean()
                )
            )

            loss_disc_all = loss_disc_s + loss_disc_f + r1_penalty

        scaler.scale(loss_disc_all).backward()
        if cfg.optim.max_grad_norm:
            scaler.unscale_(optim_d)
            torch.nn.utils.clip_grad_norm_(itertools.chain(mpd.parameters(), msd.parameters()), cfg.optim.max_grad_norm)
        scaler.step(optim_d)

        # --- Generatorの学習 ---
        optim_g.zero_grad()
        with autocast(enabled=True):
            y_mel = mel_spectrogram(
                clean_22k_prepared.squeeze(1),
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.fmax_for_loss,
            )
            y_g_hat_mel = mel_spectrogram(
                y_g_hat_prepared.squeeze(1),
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.fmax_for_loss,
            )
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * cfg.lambda_mel

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(clean_22k_prepared, y_g_hat_prepared)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(clean_22k_prepared, y_g_hat_prepared)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            loss_adv = (loss_gen_s + loss_gen_f) * cfg.lambda_adv
            loss_fm = (loss_fm_f + loss_fm_s) * cfg.lambda_fm
            loss_gen_all = loss_adv + loss_fm + loss_mel

        scaler.scale(loss_gen_all).backward()
        if cfg.optim.max_grad_norm:
            scaler.unscale_(optim_g)
            torch.nn.utils.clip_grad_norm_(generator_modules.parameters(), cfg.optim.max_grad_norm)
        scaler.step(optim_g)
        ema_g.update()

        scaler.update()
        scheduler_g.step()
        scheduler_d.step()

        if (step % cfg.log_interval) == 0:
            log_data = {
                "pretrain/loss/generator_total": loss_gen_all.item(),
                "pretrain/loss/generator_adv": loss_adv.item(),
                "pretrain/loss/feature_matching": loss_fm.item(),
                "pretrain/loss/mel_l1": loss_mel.item(),
                "pretrain/loss/discriminator_total": loss_disc_all.item(),
                "pretrain/loss/r1_penalty": r1_penalty.item(),
                "pretrain/learning_rate": scheduler_g.get_last_lr()[0],
            }
            print(
                f"[Pre-train Step {step:>7d}/{cfg.steps}] Gen_Loss: {log_data['pretrain/loss/generator_total']:.4f}, Disc_Loss: {log_data['pretrain/loss/discriminator_total']:.4f}"
            )
            if cfg.wandb.enabled:
                wandb.log(log_data, step=step)

        if hasattr(cfg, "checkpoint") and step > 0 and step % cfg.checkpoint.save_interval == 0:
            checkpoint_dir = pathlib.Path(cfg.save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                checkpoint_dir=str(checkpoint_dir),
                step=step,
                model_state={},
                optimizer_state={},
                additional_states={
                    "generator_modules": generator_modules.state_dict(),
                    "mpd": mpd.state_dict(),
                    "msd": msd.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "scheduler_g": scheduler_g.state_dict(),
                    "scheduler_d": scheduler_d.state_dict(),
                    "ema_g": ema_g.state_dict(),
                    "steps": step,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "wandb_run_id": wandb_id,
                },
                cfg=cfg,
                keep_last_n=cfg.checkpoint.keep_last_n,
            )

            if cfg.wandb.enabled and cfg.wandb.log_audio:
                with torch.no_grad():
                    ema_g.apply_shadow()
                    y_g_hat_ema = generator_modules.generator(generator_modules.prenet(feat))
                    ema_g.restore()
                sample_path = checkpoint_dir / f"sample_ema_{step}.wav"
                save(sample_path, y_g_hat_ema[0:1].squeeze(0).cpu(), sr=h.sampling_rate)
                wandb.log(
                    {"pretrain/audio/generated_sample_ema": wandb.Audio(str(sample_path), sample_rate=h.sampling_rate)},
                    step=step,
                )

    if cfg.wandb.enabled:
        wandb.finish()
