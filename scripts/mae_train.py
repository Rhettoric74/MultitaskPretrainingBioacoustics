import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

from heapq import heappush, heapreplace
from pathlib import Path

from birdset_waveform_dataloader import BirdSetDataLoader
from build_mae_model import build_model
from mae_eval import evaluate as evaluate_pow


# -----------------------------
# CONFIG
# -----------------------------
torch.set_float32_matmul_precision("high")

DATASET_PATH = "/scratch/Projects/CFP-04/CFP04-CF-029/birdset"
BATCH_SIZE = 384
NUM_WORKERS = 32
MAX_EPOCHS = 50

CHECKPOINT_DIR = Path(
    "/scratch/Projects/CFP-04/CFP04-CF-029/checkpoints/mae_audio/drop_path_01_height_first_vit_b_4_by_8_proto_cls_dcgd_pos_gamma_1_neg_gamma_2_random_mixup_db_scale"
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESUME_CHECKPOINT = None
#RESUME_CHECKPOINT = "/scratch/Projects/CFP-04/CFP04-CF-029/checkpoints/mae_audio/vit_l_4_by_8_proto_cls_dcgd_pos_gamma_1_neg_gamma_2_random_mixup_db_scale/last.ckpt"
LABEL_VOCAB_PATH = "/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_label_vocab.json"

# -----------------------------
# Helpers
# -----------------------------
def move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    return batch


# -----------------------------
# Callback: evaluate POW and save best checkpoints manually
# -----------------------------
class PowCheckpointCallback(Callback):
    """Evaluate POW at the end of every epoch and save last + top-k by AUROC."""

    def __init__(self, pow_loader, dirpath: str, top_k: int = 3):
        super().__init__()
        self.pow_loader = pow_loader
        self.dirpath = Path(dirpath)
        self.top_k = int(top_k)
        self.best = []  # min-heap of (score, path)
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        device = next(pl_module.parameters()).device
        metrics = evaluate_pow(pl_module, self.pow_loader, device)
        score = float(metrics["AUROC"])
        epoch = int(trainer.current_epoch)

        # Log to CSV/progress bar.
        pl_module.log("pow_val_auroc", score, prog_bar=True, on_step=False, on_epoch=True)
        pl_module.log("pow_val_cmap", float(metrics["cmAP"]), prog_bar=False, on_step=False, on_epoch=True)
        pl_module.log("pow_val_top1", float(metrics["top1_acc"]), prog_bar=False, on_step=False, on_epoch=True)

        print(
            f"[POW VAL] epoch={epoch} | cmAP={metrics['cmAP']:.6f} | "
            f"AUROC={metrics['AUROC']:.6f} | top1={metrics['top1_acc']:.6f}"
        )

        # Always save last checkpoint.
        last_path = self.dirpath / "last.ckpt"
        trainer.save_checkpoint(str(last_path))

        # Save the top-k AUROC checkpoints.
        ckpt_name = f"epoch{epoch:02d}-powauroc{score:.4f}.ckpt"
        ckpt_path = self.dirpath / ckpt_name

        if len(self.best) < self.top_k:
            trainer.save_checkpoint(str(ckpt_path))
            heappush(self.best, (score, str(ckpt_path)))
        else:
            worst_score, worst_path = self.best[0]
            if score > worst_score:
                trainer.save_checkpoint(str(ckpt_path))
                _, old_path = heapreplace(self.best, (score, str(ckpt_path)))
                try:
                    Path(old_path).unlink(missing_ok=True)
                except Exception:
                    pass


# -----------------------------
# Main
# -----------------------------
def main():
    # ---- Data ----
    train_data = BirdSetDataLoader(
        dataset_path=DATASET_PATH,
        subset="XCL",
        split="train",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        use_mixup=True,
        use_geo_mixup=True,
        mixup_prob=1.0,
        mix_alpha=91.3,
        mix_beta=100.0,
        mix_omega=1.0,
        mix_n=2,
        label_vocab_path=LABEL_VOCAB_PATH,
    )
    train_loader = train_data.get_loader()
    num_classes = train_data.dataset.num_classes

    # POW split used for model selection.
    pow_val_data = BirdSetDataLoader(
        dataset_path=DATASET_PATH,
        subset="POW",
        split="test_5s",
        batch_size=512,      # or even larger if it fits
        num_workers=0,       # start here
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
        label_vocab_path=LABEL_VOCAB_PATH,
    )
    pow_val_loader = pow_val_data.get_loader()
    
    # ---- Model ----
    model = build_model(
        num_classes=num_classes,
        classifier_type="proto",
        use_cls_token=False,
        objective_mode="joint",
        optimizer_type="dcgd",
        mask_ratio=0.75,
        lambda_recon=1.0,
        lambda_cls=1.0,
        learning_rate=2e-4,
        weight_decay=0.0,
    )
    """
    model = build_model(
        num_classes=num_classes,
        classifier_type="proto",
        classifier_kwargs={
            "num_prototypes_per_class": 5,
            "activation": "cosine",
            "temperature": 0.1,
            "learnable_readout": True,
        },
        objective_mode="joint",
        optimizer_type="dcgd",
        mask_ratio=0.75,
        lambda_recon=1.0,
        lambda_cls=0.01,
        learning_rate=3e-4,
        weight_decay=0.05,
    )
    """
    
    

    # ---- Resume ----
    if RESUME_CHECKPOINT:
        print(f"Loading checkpoint from: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")

    # ---- Callbacks ----
    pow_ckpt_callback = PowCheckpointCallback(pow_val_loader, dirpath=str(CHECKPOINT_DIR), top_k=3)

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32-true",
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        callbacks=[pow_ckpt_callback],
        logger=CSVLogger("logs", name="mae_audio_pow"),
    )

    print("Starting Training")
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()



