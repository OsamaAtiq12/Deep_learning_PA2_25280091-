"""
Standalone inference script for Quick, Draw! leaderboard submission.
Loads the best Champion model weights and generates submission.txt.
Usage: python inference.py
Requires: processed_data/quickdraw_test.npz, best_champion.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

# --------------- Config (must match training) ---------------
INPUT_SIZE = 28 * 28
NUM_CLASSES = 15
DATA_DIR = "processed_data"
TEST_FILE = os.path.join(DATA_DIR, "quickdraw_test.npz")
BEST_WEIGHTS_PATH = "best_champion.pt"
SUBMISSION_FILE = "submission.txt"
BATCH_SIZE = 128


class ChampionMLP(nn.Module):
    """Must match the architecture used in training (Part C)."""
    def __init__(self, input_dim=INPUT_SIZE, num_classes=NUM_CLASSES, hidden_width=512, num_hidden_layers=4, dropout=0.35):
        super().__init__()
        layers = []
        in_f = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_f, hidden_width))
            layers.append(nn.BatchNorm1d(hidden_width))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_f = hidden_width
        self.backbone = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_width, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.backbone(x)
        return self.fc_out(x)


def load_test_data(file_path):
    """Load test images and normalize to [0, 1]."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test file not found: {file_path}")
    data = np.load(file_path)
    x = data["test_images"]
    x = torch.from_numpy(x).float() / 255.0
    return x


def get_predictions(model, images, device, batch_size=BATCH_SIZE):
    model.eval()
    model.to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy().tolist())
    return preds


def main():
    parser = argparse.ArgumentParser(description="Quick Draw inference")
    parser.add_argument("--weights", type=str, default=BEST_WEIGHTS_PATH, help="Path to model weights")
    parser.add_argument("--test", type=str, default=TEST_FILE, help="Path to test .npz")
    parser.add_argument("--out", type=str, default=SUBMISSION_FILE, help="Output submission file")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}. Train the Champion model first (Part C in notebook).")

    print(f"Loading weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)

    model = ChampionMLP(hidden_width=512, num_hidden_layers=5, dropout=0.38)
    model.load_state_dict(state_dict)

    print(f"Loading test data from {args.test}...")
    test_images = load_test_data(args.test)
    print(f"  {len(test_images)} images")

    print("Running inference...")
    predictions = get_predictions(model, test_images, device, batch_size=args.batch_size)

    submission_string = ",".join(map(str, predictions))
    with open(args.out, "w") as f:
        f.write(submission_string)
    print(f"Predictions saved to '{args.out}'.")
    print("Copy & paste the file contents to the submission portal.")


if __name__ == "__main__":
    main()
