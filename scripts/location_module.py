import torch
import torch.nn as nn
import torch.nn.functional as F
from sinr.models import ResidualFCNet, LinNet
from sinr.setup import get_default_params_train
from sinr.utils import CoordEncoder

sinr_params = get_default_params_train()
DEFAULT_PATH = "/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/sinr/experiments/demo/sinr_model.pt"
sinr_classes = 47375

class LocationModule(nn.Module):
    def __init__(
        self,
        encoder_type="ResidualFCNet",
        embed_dim=sinr_params["num_filts"],
        num_classes=9734,
        location_protos_per_class=1,
        encoder_path=DEFAULT_PATH,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.coord_encoder = CoordEncoder('sin_cos')

        if self.encoder_type == "ResidualFCNet":
            self.location_encoder = ResidualFCNet(
                4,
                sinr_classes,
                sinr_params["num_filts"],
                sinr_params["depth"],
            )
        elif self.encoder_type == "LinNet":
            self.location_encoder = LinNet(
                4,
                sinr_classes,
            )
        else:
            raise NotImplementedError("Invalid model specified.")

        if encoder_path is not None:
            print("Loading location encoder")
            encoder_params = torch.load(encoder_path, map_location="cpu")["state_dict"]
            self.location_encoder.load_state_dict(encoder_params, strict=True)

        for p in self.location_encoder.parameters():
            p.requires_grad = False

        self.location_encoder.eval()
        self.num_classes = num_classes

        prototypes = torch.randn(
            num_classes,
            location_protos_per_class,
            embed_dim,
        )
        prototypes = F.normalize(prototypes, dim=-1)
        self.prototypes = nn.Parameter(prototypes)

        self.prototype_weights = nn.Parameter(
            torch.full(
                (num_classes, location_protos_per_class),
                1.0 / location_protos_per_class,
                dtype=torch.float32,
            )
        )

    def forward(self, coordinates):
        # [B]
        has_coordinates = torch.isfinite(coordinates).all(dim=1)
    
        B = coordinates.shape[0]
    
        # Initialize all priors to zero
        prior_logits = torch.zeros(
            B,
            self.num_classes,
            device=coordinates.device,
            dtype=torch.float32,
        )
    
        # Skip encoder if nobody has coordinates
        if not has_coordinates.any():
            return prior_logits
    
        # Only process valid coordinates
        valid_coords = coordinates[has_coordinates]
        print(valid_coords[0])
    
        coord_enc = self.coord_encoder.encode(valid_coords)
        loc_embed = self.location_encoder(coord_enc, return_feats=True)
        loc_embed = F.normalize(loc_embed, dim=-1)
    
        norm_protos = F.normalize(self.prototypes, dim=-1)
    
        similarities = torch.einsum("ckd,bd->bck", norm_protos, loc_embed)
    
        valid_prior_logits = torch.einsum(
            "bck,ck->bc",
            similarities,
            self.prototype_weights,
        )
    
        # Scatter back into full batch
        prior_logits[has_coordinates] = valid_prior_logits
        #pow_indices = [0, 75, 351, 1118, 1707, 2453, 3231, 3277, 3297, 3325, 3346, 4792, 5806, 5820, 6054, 6303, 6371, 6373, 6376, 6377, 6386, 6396, 6402, 6406, 6407, 6414, 6418, 6879, 6933, 6951, 6956, 7103, 7109, 7111, 7116, 7171, 7179, 7206, 7761, 7774, 7807, 7931, 8016, 8041, 8048, 8095, 9550, 9573, 9675]
        #print("Sampled logits:", prior_logits[0][pow_indices].cpu()) 
    
        return prior_logits
        
if __name__ == '__main__':
    coords = torch.tensor([[-93.1, 45.1], [float("nan"), float("nan")]])
    location_module = LocationModule()
    print(location_module(coords)[:,0])
    
        
        