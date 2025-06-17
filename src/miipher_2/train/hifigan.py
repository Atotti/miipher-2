import itertools
import pathlib

import torch
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader

import wandb
from miipher_2.data.webdataset_loader import VocoderDataset
from miipher_2.hifigan.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from miipher_2.hifigan.generator import Generator
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.utils.audio import save
from miipher_2.utils.checkpoint import (
    get_resume_checkpoint_path,
    load_checkpoint,
    restore_random_states,
    save_checkpoint,
    setup_wandb_resume,
)


# ---------------- util ----------------
def _mag(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 0].square() + x[..., 1].square()).sqrt()


def stft_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ma = _mag(torch.stft(a, 1024, return_complex=False))
    mb = _mag(torch.stft(b, 1024, return_complex=False))
    return (ma - mb).abs().mean()


# ---------------- build ----------------
def _build_gen(cfg: DictConfig, hubert_dim: int) -> Generator:  # hubert_dimを受け取る
    return Generator(
        hubert_dim=hubert_dim,
        upsample_rates=cfg.upsample_rates,
        upsample_kernel_sizes=cfg.upsample_kernel_sizes,
    ).cuda()


@torch.no_grad()
def _load_pretrained(gen: Generator, path: pathlib.Path) -> None:
    state = torch.load(path, map_location="cpu")
    if "generator" in state:
        state = state["generator"]
    if "state_dict" in state:
        state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    gen.load_state_dict(state, strict=False)
    print(f"[INFO] pre-trained G loaded: {path}")


def train_hifigan(cfg: DictConfig) -> None:  # noqa: PLR0912
    # ---------------- チェックポイント確認と再開 ----------------
    resume_checkpoint_path = get_resume_checkpoint_path(cfg)
    resumed_checkpoint = None

    if resume_checkpoint_path:
        resumed_checkpoint = load_checkpoint(resume_checkpoint_path)
        restore_random_states(resumed_checkpoint)
        print(f"[INFO] Resuming from step {resumed_checkpoint['step']}")

    # ---------------- Wandb初期化 ----------------
    setup_wandb_resume(cfg, resumed_checkpoint)

    # --- Data
    dl = DataLoader(
        VocoderDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    )
    dl_iter = iter(dl)

    # --- Models
    cleaner = FeatureCleaner(cfg.model).cuda().eval()
    cleaner.load_state_dict(torch.load(cfg.adapter_ckpt, map_location="cpu"))
    hubert_dim = cleaner.extractor.hubert.config.hidden_size

    gen = _build_gen(cfg, hubert_dim=hubert_dim)
    if not resumed_checkpoint:
        # 新規学習時のみ事前学習済みモデルを読み込み
        _load_pretrained(gen, pathlib.Path(cfg.pretrained_gen))

    mpd, msd = MultiPeriodDiscriminator().cuda(), MultiScaleDiscriminator().cuda()

    # --- Optim
    opt_g = optim.AdamW(gen.parameters(), lr=cfg.lr, betas=tuple(cfg.betas))
    opt_d = optim.AdamW(itertools.chain(mpd.parameters(), msd.parameters()), lr=cfg.lr, betas=tuple(cfg.betas))
    scaler = torch.cuda.amp.GradScaler()

    # ---------------- チェックポイントから状態復元 ----------------
    start_step = 0

    if resumed_checkpoint:
        gen.load_state_dict(resumed_checkpoint["model_state_dict"])
        opt_g.load_state_dict(resumed_checkpoint["optimizer_state_dict"])

        # 追加の状態があれば復元
        if "mpd_state_dict" in resumed_checkpoint:
            mpd.load_state_dict(resumed_checkpoint["mpd_state_dict"])
        if "msd_state_dict" in resumed_checkpoint:
            msd.load_state_dict(resumed_checkpoint["msd_state_dict"])
        if "opt_d_state_dict" in resumed_checkpoint:
            opt_d.load_state_dict(resumed_checkpoint["opt_d_state_dict"])
        if "scaler_state_dict" in resumed_checkpoint:
            scaler.load_state_dict(resumed_checkpoint["scaler_state_dict"])

        start_step = resumed_checkpoint["step"]
        print("[INFO] Restored model and optimizer states")

    # --- Loop
    for step in range(start_step, cfg.steps):
        try:
            noisy_16k, clean_22k = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            noisy_16k, clean_22k = next(dl_iter)

        noisy_16k, clean_22k = noisy_16k.cuda(), clean_22k.cuda()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            feat = cleaner(noisy_16k)
            fake_22k = gen(feat)
            l_stft = stft_loss(fake_22k, clean_22k)

        # ---- D
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            d_real = [d(clean_22k) for d in (mpd, msd)]
            d_fake = [d(fake_22k.detach()) for d in (mpd, msd)]
            l_d = sum(
                (r[0] - 1).square().mean() + f[0].square().mean()
                for r, f in zip(itertools.chain(*d_real), itertools.chain(*d_fake), strict=False)
            )

        scaler.scale(l_d).backward()
        scaler.step(opt_d)
        scaler.update()
        opt_d.zero_grad(set_to_none=True)

        # ---- G
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            g_adv = sum((f[0] - 1).square().mean() for f in itertools.chain(mpd(fake_22k), msd(fake_22k)))
            l_g = cfg.lambda_stft * l_stft + cfg.lambda_mpd * g_adv / 2 + cfg.lambda_msd * g_adv / 2

        scaler.scale(l_g).backward()
        scaler.step(opt_g)
        scaler.update()
        opt_g.zero_grad(set_to_none=True)

        # ---- log / save
        if (step % cfg.log_interval) == 0:
            log_data = {
                "step": step,
                "loss/generator": l_g.item(),
                "loss/discriminator": l_d.item(),
                "loss/stft": l_stft.item(),
                "loss/adversarial": g_adv.item(),
                "learning_rate": opt_g.param_groups[0]["lr"],
            }

            print(f"[G] step {step:>6}/{cfg.steps}  L_STFT:{l_stft:.3f}  L_adv:{g_adv:.3f}")

            if cfg.wandb.enabled:
                wandb.log(log_data, step=step)

        # ---------------- チェックポイント保存 ----------------
        if hasattr(cfg, "checkpoint") and step > 0 and step % cfg.checkpoint.save_interval == 0:
            # 音声サンプル保存
            sd = pathlib.Path(cfg.save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            sample_path = sd / f"sample_{step}.wav"
            save(sample_path, fake_22k[0:1].cpu(), sr=22050)

            # チェックポイント保存
            additional_states = {
                "mpd_state_dict": mpd.state_dict(),
                "msd_state_dict": msd.state_dict(),
                "opt_d_state_dict": opt_d.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }

            checkpoint_path = save_checkpoint(
                checkpoint_dir=cfg.save_dir,
                step=step,
                model_state=gen.state_dict(),
                optimizer_state=opt_g.state_dict(),
                additional_states=additional_states,
                cfg=cfg,
                keep_last_n=cfg.checkpoint.keep_last_n,
            )

            # Log model checkpoint and audio sample as wandb artifacts
            if cfg.wandb.enabled:
                # Log model checkpoint
                if cfg.wandb.log_model:
                    model_artifact = wandb.Artifact(f"hifigan_checkpoint_{step}", type="model")
                    model_artifact.add_file(checkpoint_path)
                    wandb.log_artifact(model_artifact)

                # Log audio sample
                if cfg.wandb.log_audio:
                    audio_artifact = wandb.Artifact(f"audio_sample_{step}", type="audio")
                    audio_artifact.add_file(str(sample_path))
                    wandb.log_artifact(audio_artifact)

                    # Also log audio directly to wandb
                    wandb.log({"audio/generated_sample": wandb.Audio(str(sample_path), sample_rate=22050)}, step=step)

    if cfg.wandb.enabled:
        wandb.finish()
