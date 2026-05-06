import torch

from birdset_dataloader import BirdSetDataLoader
from build_model import build_model


DATASET_PATH = "/scratch/Projects/CFP-04/CFP04-CF-029/birdset"


def main():

    data = BirdSetDataLoader(dataset_path=DATASET_PATH, use_geo_mixup = False)
    loader = data.get_loader()

    batch = next(iter(loader))

    print("Batch shapes:")
    print("spectrograms:", batch["spectrograms"].shape)
    print("labels:", batch["labels"].shape)

    num_classes = data.dataset.num_classes
    num_patches = (128 // 16) * (512 // 16)

    model = build_model(num_classes, num_patches)

    model.eval()

    with torch.no_grad():
        out = model.model_step(batch)

    print("Forward pass OK")
    print("Loss:", out[0].item())
    print("Jepa loss:", out[1].item())
    print("Classification loss:", out[2].item())


if __name__ == "__main__":
    main()