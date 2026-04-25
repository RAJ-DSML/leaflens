import torch
import torch.nn as nn
import open_clip

class CLIPClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "ViT-B-32", pretrained: str = "openai", dropout: float = 0.3):
        super().__init__()

        # Load pretrained CLIP vision encoder
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        # Freeze CLIP backbone — only train the head
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Get CLIP embedding dimension
        embed_dim = self.clip_model.visual.output_dim

        # Trainable classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            features = features.float()
        return self.classifier(features)

    def unfreeze_last_n_blocks(self, n: int = 2):
        """Gradually unfreeze last n transformer blocks for fine-tuning."""
        blocks = self.clip_model.visual.transformer.resblocks
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"Unfroze last {n} transformer blocks")


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = CLIPClassifier(num_classes=15).to(device)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Test forward pass
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out = model(dummy)
    print(f"Output shape    : {out.shape}")