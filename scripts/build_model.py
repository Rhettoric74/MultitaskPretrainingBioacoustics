import torch.nn as nn
import torch.optim as optim
import torch

from encoder import JEPATimmViT
from predictor import JEPAPredictor
from classifier import BirdClassifierHead
from jepa_module import BioacousticJEPAModule
from losses import *
from timm.loss import AsymmetricLossMultiLabel
from dcgd import DCGD



def build_model(num_classes, num_patches):
    # ---- Encoder ----
    encoder = JEPATimmViT(
        model_name="vit_base_patch16_224",
        pretrained=False,
    )
    """
    for param in encoder.parameters():
        param.requires_grad = False
    """
    target_encoder = JEPATimmViT(
        model_name="vit_base_patch16_224",
        pretrained=False,
    )

    embed_dim = encoder.embed_dim

    # ---- Predictor ----
    predictor = JEPAPredictor(
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=embed_dim // 2,
        depth=6,
        num_heads=12,
    )

    # ---- Classifier ----
    classifier = BirdClassifierHead(
        embed_dim=embed_dim,
        num_classes=num_classes
    )

    # ---- Losses ----
    #weight = torch.full((num_classes,), 20.0)
    #criterion_jepa = NormalizedMSELoss()
    #criterion_jepa = NormalizedMSEWithVarianceLoss()
    criterion_jepa = CosineSimWithVarianceCovarianceLoss()
    #criterion_cls = FocalLoss(gamma=2.0)
    criterion_cls = AsymmetricLossMultiLabel(
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip=0.05
    )

    model = BioacousticJEPAModule(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        classifier=classifier,
        criterion_jepa=criterion_jepa,
        criterion_cls=criterion_cls,
        lambda_cls=1.0,
        optimizer_type="adam",  # "dcgd" or "adam"
        learning_rate=3e-4,
        #learning_rate=1e-3,
        weight_decay=1e-4,
        ema_momentum = 0.996,
        classify_with_preds=True,
        objective_mode="jepa",
        
    )

    return model