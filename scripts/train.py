#print("Starting train.py")
import os
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from birdset_dataloader_v2 import BirdSetDataLoader
from build_model import build_model


# -----------------------------
# CONFIG
# -----------------------------
torch.set_float32_matmul_precision("high")

DATASET_PATH = "/scratch/Projects/CFP-04/CFP04-CF-029/birdset"
BATCH_SIZE = 64
NUM_WORKERS = 16
MAX_EPOCHS = 1

CHECKPOINT_DIR = Path(
    "/scratch/Projects/CFP-04/CFP04-CF-029/checkpoints/jepa_audio/long_jepa_run"
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Set this to resume training from a checkpoint, or leave as None for fresh training
RESUME_CHECKPOINT = None
#RESUME_CHECKPOINT = "/scratch/Projects/CFP-04/CFP04-CF-029/checkpoints/jepa_audio/jepa_run/last-v3.ckpt"


def main():
    # ---- Data ----
    data = BirdSetDataLoader(
        dataset_path=DATASET_PATH,
        subset="XCL",
        split="train",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        use_mixup=True,
        use_geo_mixup=True,
        mixup_prob=0.5,
        label_vocab_path="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_label_vocab.json",
        spec_norm_path="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_spec_stats_true_log.json",
        apply_spec_norm=True,
    )
    
    train_loader = data.get_loader()
    num_classes = data.dataset.num_classes

    # ---- Patch count ----
    patch_size = 16
    num_patches = (128 // patch_size) * (512 // patch_size)

    # ---- Model ----
    model = build_model(num_classes, num_patches)

    # ---- MANUALLY LOAD CHECKPOINT (bypassing Lightning) ----
    if RESUME_CHECKPOINT:
        print(f"Manually loading checkpoint from: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location="cpu", weights_only=False)

        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print("Model weights loaded successfully")

    # ---- Checkpoint Callback ----
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="epoch{epoch:02d}-trainloss{train_loss:.3f}",
        monitor="train_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        save_weights_only=False,
        save_on_train_epoch_end=True,
    )

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32-true",
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        logger=CSVLogger("logs", name="jepa_audio"),
    )

    print("Starting Training")
    trainer.fit(model, train_loader, ckpt_path=None)


if __name__ == "__main__":
    main()