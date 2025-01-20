import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from src.data import get_loaders, get_metr_la_dataset
from src.model.stgcn import STGCN

# =====================================
# Hyperparameters
# =====================================
node_features = 1
out_channels = 1
time_steps = 12
K = 3
num_workers = 16
batch_size = 64
learning_rate = 0.01
epochs = 100
logs_path = "runs/logs_stgcn/"
checkpoint_path = "runs/model_checkpoint_stgcn.pth"

# =====================================
# Data
# =====================================
if __name__ == "__main__":
    dataset = get_metr_la_dataset()
    train_loader, val_loader, test_loader = get_loaders(
        dataset,
        val_ratio=0.1,
        test_ratio=0.2,
        proportion_original_dataset=1,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # =====================================
    # Model
    # =====================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = dataset[0].x.size(0)
    model = STGCN(
        in_channels=node_features,
        out_channels=out_channels,
        num_nodes=num_nodes,
        time_steps=time_steps,
        K=K,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    loss_function = torch.nn.MSELoss()
    writer = SummaryWriter(logs_path)

    # =====================================
    # Training Loop
    # =====================================
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x, edge_index, edge_weight, y = batch
            x, edge_index, edge_weight, y = x.to(device), edge_index.to(device), edge_weight.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(x, edge_index, edge_weight)
            loss = loss_function(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(train_loader)
        writer.add_scalar("Training Loss", running_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, edge_index, edge_weight, y = batch
                x, edge_index, edge_weight, y = x.to(device), edge_index.to(device), edge_weight.to(device), y.to(device)

                y_hat = model(x, edge_index, edge_weight)
                loss = loss_function(y_hat, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar("Validation Loss", val_loss, epoch)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}, Training Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model with validation loss {best_val_loss:.4f}")
