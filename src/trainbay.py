import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src.data import get_loaders, get_pems_bay_dataset, get_metr_la_dataset
from src.torch_geometric_temporal.train_test_split import temporal_signal_split
from src.model.dcrnn import DCRNN

# import batch from torch_geometric
from torch_geometric.data import Batch

# =====================================
# =====================================
# Train a DCRNN model on the Pems-Bay dataset
# =====================================
# =====================================


# =====================================
# Hyperparameters
# =====================================
# Model
node_features = 1
out_channels = 32
K = 3

# Data
proportion_original_dataset = 1  # Use 1% of the original dataset to debug

# Training
num_workers = 16
batch_size = 64
resume_training = True
tau_sampling = 20  # should be 3000 for full training

# Paths
logs_path = "runs/logs_dcrnn_BAY/"
checkpoint_path = "runs/model_checkpoint_dcrnn_no_skip_BAY.pth"


# =====================================
# Data
# =====================================
if __name__ == "__main__":
    dataset = get_pems_bay_dataset()

    train_loader, val_loader, test_loader = get_loaders(
        dataset,
        val_ratio=0.1,
        test_ratio=0.2,
        proportion_original_dataset=proportion_original_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # =====================================
    # Model
    # =====================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DCRNN(node_features=node_features, out_channels=out_channels, K=K).to(
        device
    )

    # =====================================
    # Training
    # =====================================

    start_epoch = 0
    iterations = 0
    best_val_loss = float("inf")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_function = torch.nn.MSELoss()
    writer = SummaryWriter(logs_path)

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        iterations = checkpoint["iterations"]
        best_val_loss = checkpoint["best_val_loss"]
        print(
            f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}"
        )

    model.train()
    for epoch in range(start_epoch, 100):
        running_loss = 0.0
        for s, data in enumerate(train_loader):
            x, edge_index, edge_weight, y = data
            x = x.to(device)  # Remove batch dimension and move to device
            edge_index = edge_index[0].to(device)
            edge_weight = edge_weight[0].to(device)
            y = y.to(device)

            optimizer.zero_grad()
            h = None  # Initialize hidden state

            epsilon = tau_sampling / (
                tau_sampling + torch.exp(torch.tensor(iterations / tau_sampling))
            )

            y_hat = model(
                x,
                edge_index,
                edge_weight,
                h,
                training_target=y,
                target_sample_prob=epsilon,
            )

            loss = loss_function(y_hat, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            iterations += 1

            if (s + 1) % (len(train_loader) // 100 + 1) == 0:
                print(
                    f"\r Epoch {epoch+1}, Training Snapshot {np.round((s+1) * 100 / len(train_loader), 4)} %, Loss:  {running_loss/(s+1)}",
                    end="",
                )

        running_loss /= len(train_loader)
        writer.add_scalar("Training Loss", running_loss, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s, data in enumerate(val_loader):
                x, edge_index, edge_weight, y = data
                x = x.to(device)  # Remove batch dimension and move to device
                edge_index = edge_index[0].to(device)
                edge_weight = edge_weight[0].to(device)
                y = y.to(device)

                h = None
                y_hat = model(x, edge_index, edge_weight, h)
                loss = loss_function(y_hat, y)
                val_loss += loss.item()

        print()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        writer.add_scalar("Validation Loss", val_loss, epoch)

        print(
            f"\r Epoch {epoch+1}, Training Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "iterations": iterations,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved new best model with validation loss {best_val_loss:.4f}")

        model.train()
