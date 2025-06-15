"""
HiFi‑GAN Generator / Discriminators の pretrain & adversarial fine‑tune
"""

import argparse
import pathlib

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from miipher_2.data.dataloader import CleanNoisyDataset
from miipher_2.hifigan.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from miipher_2.hifigan.generator import Generator
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.utils.audio import save


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_list", type=pathlib.Path, required=True)
    ap.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch_sec", type=float, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--adapter_ckpt", type=pathlib.Path)
    ap.add_argument("--resume", type=pathlib.Path)
    ap.add_argument("--save", type=pathlib.Path, default=pathlib.Path("exp/hifigan"))
    ap.add_argument("--lambda_stft", type=float, default=1.0)
    ap.add_argument("--lambda_mpd", type=float, default=2.5)
    ap.add_argument("--lambda_msd", type=float, default=2.5)
    ap.add_argument("--r1", type=float, default=10.0)
    return ap.parse_args()


def _mag(stft_out: torch.Tensor):
    return (stft_out[..., 0] ** 2 + stft_out[..., 1] ** 2).sqrt()


def stft_loss(x, y):
    sx = _mag(torch.stft(x, 1024, return_complex=False))
    sy = _mag(torch.stft(y, 1024, return_complex=False))
    return torch.mean(torch.abs(sx - sy))


def main() -> None:
    args = parse()
    wavs = [p.strip() for p in args.wav_list.read_text().splitlines()]
    dl = DataLoader(CleanNoisyDataset([pathlib.Path(p) for p in wavs]), batch_size=4, shuffle=True, num_workers=8)
    feat_cleaner = FeatureCleaner().cuda().eval()
    if args.adapter_ckpt:
        feat_cleaner.load_state_dict(torch.load(args.adapter_ckpt))
    gen = Generator().cuda()
    mpd, msd = MultiPeriodDiscriminator().cuda(), MultiScaleDiscriminator().cuda()
    opt_g = optim.AdamW(gen.parameters(), lr=args.lr)
    opt_d = optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), lr=args.lr)
    start_step = 0
    if args.resume:
        state = torch.load(args.resume)
        gen.load_state_dict(state["gen"])
        opt_g.load_state_dict(state["opt_g"])
        mpd.load_state_dict(state["mpd"])
        msd.load_state_dict(state["msd"])
        start_step = state["step"]
    scaler = torch.cuda.amp.GradScaler()
    for step in range(start_step, args.steps):
        noisy, clean = next(iter(dl))
        noisy, clean = noisy.cuda(), clean.cuda()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            feat = feat_cleaner(noisy if args.stage == "finetune" else clean)
            fake = gen(feat)
            loss_stft = stft_loss(fake, clean)
        if args.stage == "pretrain":
            loss = loss_stft
            scaler.scale(loss).backward()
            scaler.step(opt_g)
            scaler.update()
            opt_g.zero_grad()
        else:  # adversarial
            # --- D update ---
            real_d, fake_d = mpd(clean), mpd(fake.detach())
            D_mpd = sum([torch.mean((r - 1) ** 2) + torch.mean(f**2) for r, f in zip(real_d, fake_d, strict=False)])
            real_d, fake_d = msd(clean), msd(fake.detach())
            D_msd = sum([torch.mean((r - 1) ** 2) + torch.mean(f**2) for r, f in zip(real_d, fake_d, strict=False)])
            D_loss = D_mpd + D_msd
            scaler.scale(D_loss).backward()
            scaler.step(opt_d)
            scaler.update()
            opt_d.zero_grad()
            # --- G update ---
            fake_d_mpd, fake_d_msd = mpd(fake), msd(fake)
            adv_mpd = sum([torch.mean((f - 1) ** 2) for f in fake_d_mpd])
            adv_msd = sum([torch.mean((f - 1) ** 2) for f in fake_d_msd])
            G_loss = args.lambda_stft * loss_stft + args.lambda_mpd * adv_mpd + args.lambda_msd * adv_msd
            scaler.scale(G_loss).backward()
            scaler.step(opt_g)
            scaler.update()
            opt_g.zero_grad()
        if step % 1000 == 0:
            print(f"{step}/{args.steps} stft:{loss_stft.item():.3f}")
        if (step + 1) % 50000 == 0:
            args.save.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step + 1,
                    "gen": gen.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "mpd": mpd.state_dict(),
                    "msd": msd.state_dict(),
                },
                args.save / f"g_{step + 1 // 1000}k.pth",
            )
            save(args.save / f"sample_{step + 1}.wav", fake[0:1].cpu())


if __name__ == "__main__":
    main()
