import itertools
import json
import pathlib

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import wandb
from miipher_2.data.webdataset_loader import VocoderDataset
from miipher_2.hifigan.meldataset import mel_spectrogram

# 公式HiFi-GANのモデルと損失関数をインポート
from miipher_2.hifigan.models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.utils.audio import save
from miipher_2.utils.checkpoint import (
    get_resume_checkpoint_path,
    load_checkpoint,
    restore_random_states,
    save_checkpoint,
    setup_wandb_resume,
)


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


def train_hifigan(cfg: DictConfig) -> None:  # noqa: PLR0912
    # ---------------- チェックポイント確認と再開 ----------------
    resume_checkpoint_path = get_resume_checkpoint_path(cfg)
    resumed_checkpoint = None
    if resume_checkpoint_path:
        resumed_checkpoint = load_checkpoint(resume_checkpoint_path)
        restore_random_states(resumed_checkpoint)
        print(f"[INFO] Resuming from step {resumed_checkpoint['steps']}")

    # ---------------- Wandb初期化 ----------------
    # wandbのIDはチェックポイントから引き継ぐ
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

    # --- データローダー ---
    # VocoderDatasetは(noisy_16k, clean_22k)のタプルを返す
    dl = DataLoader(
        VocoderDataset(cfg.dataset.path_pattern, shuffle=cfg.dataset.shuffle),
        batch_size=cfg.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=True,
        collate_fn=collate_tensors,
    )

    # --- モデルの構築 ---
    # 1. 特徴量クリーナー (Adapter適用済みHuBERT)
    # これは学習せず、特徴量抽出にのみ使用する
    cleaner = FeatureCleaner(cfg.model).cuda().eval()
    adapter_checkpoint = torch.load(cfg.adapter_ckpt, map_location="cpu", weights_only=False)
    cleaner.load_state_dict(adapter_checkpoint["model_state_dict"])
    for param in cleaner.parameters():
        param.requires_grad = False

    # 2. 公式HiFi-GANモデル
    # Generatorは公式の設定ファイル(config.json)を元にAttrDictを作成して渡す
    with open(pathlib.Path(cfg.pretrained_gen).parent / "config.json") as f:
        h_dict = json.load(f)
    h = AttrDict(h_dict)

    # miipher-2のprenetと公式Generatorを接続
    # まずは、HuBERT特徴量をメルスペクトログラム風(80次元)に変換するprenet
    from miipher_2.hifigan.prenet import MHubertToMel

    hubert_dim = cleaner.extractor.hubert.config.hidden_size
    prenet = MHubertToMel(hubert_dim).cuda()

    # 次に公式Generator
    generator = Generator(h).cuda()

    # Discriminators
    mpd = MultiPeriodDiscriminator().cuda()
    msd = MultiScaleDiscriminator().cuda()

    # --- 最適化アルゴリズム ---
    # prenetとgeneratorのパラメータを一緒に最適化
    optim_g = optim.AdamW(
        itertools.chain(prenet.parameters(), generator.parameters()), lr=cfg.lr, betas=tuple(cfg.betas)
    )
    optim_d = optim.AdamW(itertools.chain(mpd.parameters(), msd.parameters()), lr=cfg.lr, betas=tuple(cfg.betas))

    # --- 事前学習済み重みとチェックポイントの読み込み ---
    start_step = 0
    if resumed_checkpoint:
        # チェックポイントから再開
        prenet.load_state_dict(resumed_checkpoint["prenet"])
        generator.load_state_dict(resumed_checkpoint["generator"])
        mpd.load_state_dict(resumed_checkpoint["mpd"])
        msd.load_state_dict(resumed_checkpoint["msd"])
        optim_g.load_state_dict(resumed_checkpoint["optim_g"])
        optim_d.load_state_dict(resumed_checkpoint["optim_d"])
        start_step = resumed_checkpoint["steps"] + 1
        print("[INFO] Restored model and optimizer states from miipher-2 checkpoint.")
    else:
        # 新規学習時は公式の事前学習済み重みを読み込む
        state_dict_g = torch.load(cfg.pretrained_gen, map_location="cpu")
        generator.load_state_dict(state_dict_g["generator"])
        print(f"[INFO] Loaded pre-trained Generator from: {cfg.pretrained_gen}")

    # --- 学習ループ ---
    dl_iter = iter(dl)
    for step in range(start_step, cfg.steps):
        try:
            noisy_16k, clean_22k = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            noisy_16k, clean_22k = next(dl_iter)

        noisy_16k, clean_22k = noisy_16k.cuda(), clean_22k.cuda()
        # (B, 1, T)の形式に
        if clean_22k.dim() == 2:
            clean_22k = clean_22k.unsqueeze(1)

        # --- Generatorの学習 ---
        optim_g.zero_grad()

        # HuBERT特徴量抽出
        with torch.no_grad():
            feat = cleaner(noisy_16k)

        # 音声生成
        y_g_hat = prenet(feat)
        y_g_hat = generator(y_g_hat)

        # Mel-Spectrogram Loss
        min_len = min(clean_22k.size(2), y_g_hat.size(2))
        clean_22k = clean_22k[:, :, :min_len]
        y_g_hat = y_g_hat[:, :, :min_len]
        # 公式実装に合わせて教師データからメルスペクトログラムを計算
        y_mel = mel_spectrogram(
            clean_22k.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
        )
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
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

        loss_gen_all.backward()
        optim_g.step()

        # --- Discriminatorの学習 ---
        optim_d.zero_grad()

        y_g_hat_detached = y_g_hat.detach()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(clean_22k, y_g_hat_detached)
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(clean_22k, y_g_hat_detached)
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f
        loss_disc_all.backward()
        optim_d.step()

        # ---- ログ出力とチェックポイント保存 ----
        if (step % cfg.log_interval) == 0:
            log_data = {
                "step": step,
                "loss/generator_total": loss_gen_all.item(),
                "loss/generator_adv": (loss_gen_s + loss_gen_f).item(),
                "loss/feature_matching": (loss_fm_s + loss_fm_f).item(),
                "loss/mel_l1": loss_mel.item(),
                "loss/discriminator_total": loss_disc_all.item(),
                "learning_rate": optim_g.param_groups[0]["lr"],
            }
            print(
                f"[Step {step:>7d}/{cfg.steps}] Gen_Loss: {loss_gen_all.item():.4f}, Disc_Loss: {loss_disc_all.item():.4f}"
            )
            if cfg.wandb.enabled:
                wandb.log(log_data, step=step)

        if hasattr(cfg, "checkpoint") and step > 0 and step % cfg.checkpoint.save_interval == 0:
            # チェックポイント保存
            checkpoint_dir = pathlib.Path(cfg.save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = save_checkpoint(
                checkpoint_dir=str(checkpoint_dir),
                step=step,
                model_state=None,  # ここでは使わない
                optimizer_state=None,  # ここでは使わない
                additional_states={
                    "prenet": prenet.state_dict(),
                    "generator": generator.state_dict(),
                    "mpd": mpd.state_dict(),
                    "msd": msd.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "steps": step,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "wandb_run_id": wandb_id,
                },
                cfg=cfg,
                keep_last_n=cfg.checkpoint.keep_last_n,
            )

            # 音声サンプルをログ
            if cfg.wandb.enabled and cfg.wandb.log_audio:
                sample_path = checkpoint_dir / f"sample_{step}.wav"
                save(sample_path, y_g_hat[0:1].squeeze(0).cpu().detach(), sr=h.sampling_rate)
                wandb.log(
                    {"audio/generated_sample": wandb.Audio(str(sample_path), sample_rate=h.sampling_rate)}, step=step
                )

    if cfg.wandb.enabled:
        wandb.finish()
