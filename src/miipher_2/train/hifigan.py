import pathlib

import torch
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader

from miipher_2.data.webdataset_loader import WavPairDataset
from miipher_2.hifigan.discriminator import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from miipher_2.hifigan.generator import Generator
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.utils.audio import save


# ------------------------------------------------------------
def _mag(stft_out: torch.Tensor) -> torch.Tensor:
    """複素 STFT (… ,2) → magnitude"""
    return (stft_out[..., 0].square() + stft_out[..., 1].square()).sqrt()


def stft_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sx = _mag(torch.stft(x, 1024, return_complex=False))
    sy = _mag(torch.stft(y, 1024, return_complex=False))
    return (sx - sy).abs().mean()


# ------------------------------------------------------------
def _build_generator(cfg: DictConfig) -> Generator:
    """YAML で与えた upsample ハイパーパラメータを反映して Generator を構築"""
    return Generator(
        upsample_rates=cfg.upsample_rates,
        upsample_kernel_sizes=cfg.upsample_kernel_sizes,
    ).cuda()


# ------------------------------------------------------------
@torch.no_grad()
def _load_pretrained_weights(gen: Generator, weight_path: pathlib.Path) -> None:
    """事前学習済み HiFi‑GAN Generator 重みをロード"""
    if not weight_path.exists():
        msg = f"pretrained_gen not found: {weight_path}"
        raise FileNotFoundError(msg)
    state = torch.load(weight_path, map_location="cpu")
    # 汎用 HiFi‑GAN pth は {"generator": …} or {"state_dict": …} 等が混在するためハンドリング
    if "generator" in state:
        state = state["generator"]
    elif "state_dict" in state:
        state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    gen.load_state_dict(state, strict=False)
    print(f"[INFO] Pre‑trained Generator weights loaded from {weight_path}")


# ------------------------------------------------------------
def train_hifigan(cfg: DictConfig) -> None:  # ← cmd から直接呼ばれるエントリ
    # ---------- Data ----------
    dl = DataLoader(
        WavPairDataset(cfg.dataset.path_pattern),
        batch_size=cfg.batch_size,
        num_workers=8,
        pin_memory=True,
    )

    # ---------- Models ----------
    feat_cleaner = FeatureCleaner().cuda().eval()
    feat_cleaner.load_state_dict(torch.load(cfg.adapter_ckpt, map_location="cpu"))
    gen = _build_generator(cfg)
    _load_pretrained_weights(gen, pathlib.Path(cfg.pretrained_gen))

    mpd = MultiPeriodDiscriminator().cuda()
    msd = MultiScaleDiscriminator().cuda()

    # ---------- Opt ----------
    opt_g = optim.AdamW(gen.parameters(), lr=cfg.lr)
    opt_d = optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), lr=cfg.lr)

    scaler = torch.cuda.amp.GradScaler()
    step = 0
    dl_iter = iter(dl)

    while step < cfg.steps:
        try:
            noisy, clean = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            noisy, clean = next(dl_iter)

        noisy, clean = noisy.cuda(), clean.cuda()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            feat = feat_cleaner(noisy)  # always noisy → cleaned feature
            fake = gen(feat)
            loss_stft = stft_loss(fake, clean)

            # ----- Discriminator update -----
            real_d_mpd, fake_d_mpd = mpd(clean), mpd(fake.detach())
            d_mpd = sum(
                (r[0] - 1).square().mean() + f[0].square().mean() for r, f in zip(real_d_mpd, fake_d_mpd, strict=False)
            )

            real_d_msd, fake_d_msd = msd(clean), msd(fake.detach())
            d_msd = sum(
                (r[0] - 1).square().mean() + f[0].square().mean() for r, f in zip(real_d_msd, fake_d_msd, strict=False)
            )

            d_loss = d_mpd + d_msd

        scaler.scale(d_loss).backward()
        scaler.step(opt_d)
        scaler.update()
        opt_d.zero_grad(set_to_none=True)

        # ----- Generator update -----
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            fake_d_mpd = mpd(fake)
            fake_d_msd = msd(fake)
            adv_mpd = sum((f[0] - 1).square().mean() for f in fake_d_mpd)
            adv_msd = sum((f[0] - 1).square().mean() for f in fake_d_msd)

            g_loss = cfg.lambda_stft * loss_stft + cfg.lambda_mpd * adv_mpd + cfg.lambda_msd * adv_msd

        scaler.scale(g_loss).backward()
        scaler.step(opt_g)
        scaler.update()
        opt_g.zero_grad(set_to_none=True)

        # ----- log / save -----
        if step % 1000 == 0:
            print(
                f"[{step:>6}/{cfg.steps}] "
                f"STFT:{loss_stft.item():.3f}  "
                f"adv_mpd:{adv_mpd.item():.3f}  adv_msd:{adv_msd.item():.3f}"
            )

        if (step + 1) % 50000 == 0:
            sd = pathlib.Path(cfg.save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step + 1,
                    "gen": gen.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "mpd": mpd.state_dict(),
                    "msd": msd.state_dict(),
                },
                sd / f"g_{(step + 1) // 1000}k.pth",
            )
            save(sd / f"sample_{step + 1}.wav", fake[0:1].cpu())

        step += 1
