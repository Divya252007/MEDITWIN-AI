from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "models" / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_encoder():
    checkpoint = torch.load(
        CHECKPOINTS / "image_encoder.pth",
        map_location=DEVICE,
        weights_only=False,
    )

    model = ImageEncoder(
        embedding_dim=checkpoint.get("embedding_dim", 64),
        pretrained=False,
    )

    saved_state = checkpoint["model_state_dict"]

    # The checkpoint was saved from CXRClassifier,
    # which contains ImageEncoder under the "encoder." prefix.
    encoder_state = {}

    for key, value in saved_state.items():
        if key.startswith("encoder."):
            encoder_state[key[len("encoder."):]] = value

    model.load_state_dict(encoder_state, strict=True)
    model.to(DEVICE).eval()
    return model


def load_text_encoder():
    checkpoint = torch.load(
        CHECKPOINTS / "text_encoder.pth",
        map_location=DEVICE,
        weights_only=False,
    )
    model_name = checkpoint.get(
        "model_name", "dmis-lab/biobert-base-cased-v1.2"
    )
    model = TextEncoder(model_name=model_name, embedding_dim=64)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def extract_image_embedding(model, image_file):
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    with Image.open(image_file) as image:
        image_tensor = image_transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        embedding = model(image_tensor.to(DEVICE))

    return embedding.cpu()


def extract_text_embedding(model, tokenizer, clinical_text):
    if not clinical_text.strip():
        raise ValueError("Please provide clinical text.")

    tokens = tokenizer(
        clinical_text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

    with torch.inference_mode():
        embedding = model(
            tokens["input_ids"].to(DEVICE),
            tokens["attention_mask"].to(DEVICE),
        )

    return embedding.cpu()