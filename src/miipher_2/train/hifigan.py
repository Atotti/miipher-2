import itertools
import pathlib

import torch
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader

from miipher_2.data.webdataset_loader import WavPairDataset
from miipher_2.hifigan.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from miipher_2.hifigan.generator import Generator
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.utils.audio import save


# ---------------- util ----------------
def _mag(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 0].square() + x[..., 1].square()).sqrt()


def stft_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ma = _mag(torch.stft(a, 1024, return_complex=False))
    mb = _mag(torch.stft(b, 1024, return_complex=False))
    return (ma - mb).abs().mean()


# ---------------- build ----------------
def _build_gen(cfg: DictConfig) -> Generator:
    return Generator(
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
    print(f"[INFO] pre‑trained G loaded: {path}")


# ---------------- main ----------------
def train_hifigan(cfg: DictConfig) -> None:
    # --- Data
    dl = DataLoader(
        WavPairDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    )
    dl_iter = iter(dl)

    # --- Models
    cleaner = FeatureCleaner().cuda().eval()
    cleaner.load_state_dict(torch.load(cfg.adapter_ckpt, map_location="cpu"))
    gen = _build_gen(cfg)
    _load_pretrained(gen, pathlib.Path(cfg.pretrained_gen))
    mpd, msd = MultiPeriodDiscriminator().cuda(), MultiScaleDiscriminator().cuda()

    # --- Optim
    opt_g = optim.AdamW(gen.parameters(), lr=cfg.lr, betas=tuple(cfg.betas))
    opt_d = optim.AdamW(itertools.chain(mpd.parameters(), msd.parameters()), lr=cfg.lr, betas=tuple(cfg.betas))
    scaler = torch.cuda.amp.GradScaler()

    # --- Loop
    for step in range(cfg.steps):
        try:
            noisy, clean = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            noisy, clean = next(dl_iter)

        noisy, clean = noisy.cuda(), clean.cuda()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            feat = cleaner(noisy)
            fake = gen(feat)
            l_stft = stft_loss(fake, clean)

        # ---- D
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            d_real = [d(clean) for d in (mpd, msd)]
            d_fake = [d(fake.detach()) for d in (mpd, msd)]
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
            g_adv = sum((f[0] - 1).square().mean() for f in itertools.chain(mpd(fake), msd(fake)))
            l_g = cfg.lambda_stft * l_stft + cfg.lambda_mpd * g_adv / 2 + cfg.lambda_msd * g_adv / 2

        scaler.scale(l_g).backward()
        scaler.step(opt_g)
        scaler.update()
        opt_g.zero_grad(set_to_none=True)

        # ---- log / save
        if (step % cfg.log_interval) == 0:
            print(f"[G] step {step:>6}/{cfg.steps}  L_STFT:{l_stft:.3f}  L_adv:{g_adv:.3f}")

        if (step + 1) % cfg.checkpoint_interval == 0:
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
