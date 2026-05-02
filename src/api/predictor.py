import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import open_clip
from PIL import Image
import io

from src.model.clip_model import CLIPClassifier

class LeafPredictor:
    def __init__(self, model_path: str = "models/best_model.pt"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        config = checkpoint["config"]

        # Load model
        self.model = CLIPClassifier(
            num_classes=len(self.classes),
            model_name=config["model_name"],
            pretrained=config["pretrained"],
            dropout=config["dropout"]
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # CLIP preprocessing
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            config["model_name"], pretrained=config["pretrained"]
        )

        print(f"Model loaded on {self.device} | Classes: {len(self.classes)}")

    def predict(self, image_bytes: bytes, top_k: int = 5) -> dict:
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        # Top-k results
        top_probs, top_indices = probs.topk(top_k)
        results = [
            {
                "species": self.classes[idx.item()].replace("_", " "),
                "confidence": round(prob.item() * 100, 2)
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        return {
            "prediction": results[0]["species"],
            "confidence": results[0]["confidence"],
            "top_k": results
        }


# Singleton — loaded once at startup
_predictor = None

def get_predictor() -> LeafPredictor:
    global _predictor
    if _predictor is None:
        _predictor = LeafPredictor()
    return _predictor