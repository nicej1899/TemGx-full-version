import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src.data import get_loaders, get_pems_bay_dataset, get_metr_la_dataset
from src.torch_geometric_temporal.train_test_split import temporal_signal_split
from src.model.dcrnn import DCRNN

# import batch from torch_geometric
from torch_geometric.data import Batch


class Metrics:
    def __init__(
        self, dataset_type, raw_data_dir=os.path.join(os.getcwd(), "data"), device="cpu"
    ):
        self.device = device
        self.compute_std(dataset_type, raw_data_dir)
        self.reset()

    def update(self, y, y_hat):
        # y and y_hat are torch tensors of shape (batch_size, num_nodes, num_features, num_timesteps)
        y = self.destandardize(y)
        y_hat = self.destandardize(y_hat)
        for i in range(y.shape[3]):
            # get mask from y that is different from 0
            mask = y[:, :, :, i] > 1
            y_i = y[:, :, :, i][mask]
            y_hat_i = y_hat[:, :, :, i][mask]
            self.num_samples[:, i] += mask.sum()
            self.non_averaged_mse[:, i] += torch.sum((y_i - y_hat_i) ** 2)
            self.non_averaged_mae[:, i] += torch.sum(torch.abs(y_i - y_hat_i))
            self.non_averaged_mape[:, i] += torch.sum(torch.abs((y_i - y_hat_i) / y_i))

    def destandardize(self, y):
        return y * self.stds + self.means

    def compute(self):
        metrics = {
            "MSE": self.non_averaged_mse.cpu() / self.num_samples,
            "RMSE": np.sqrt(self.non_averaged_mse.cpu() / self.num_samples),
            "MAE": self.non_averaged_mae.cpu() / self.num_samples,
            "MAPE": self.non_averaged_mape.cpu() / self.num_samples,
        }
        return metrics

    def compute_std(self, dataset_type, raw_data_dir):
        if dataset_type == "la":
            X = np.load(os.path.join(raw_data_dir, "node_values.npy")).transpose(
                (1, 2, 0)
            )
        elif dataset_type == "bay":
            X = np.load(os.path.join(raw_data_dir, "pems_node_values.npy")).transpose(
                (1, 2, 0)
            )
        X = X.astype(np.float32)

        # should only have two values
        self.means = torch.tensor(np.mean(X, axis=(0, 2))[None, None, :, None]).to(
            self.device
        )
        self.stds = torch.tensor(np.std(X, axis=(0, 2))[None, None, :, None]).to(
            self.device
        )

    def reset(self):
        self.num_samples = torch.zeros(2, 12).to(self.device)
        self.non_averaged_mse = torch.zeros(2, 12).to(self.device)
        self.non_averaged_mae = torch.zeros(2, 12).to(self.device)
        self.non_averaged_mape = torch.zeros(2, 12).to(self.device)


class Rescaler:
    def __init__(
        self,
        dataset_type_train,
        dataset_type_validate,
        raw_data_dir=os.path.join(os.getcwd(), "data"),
    ):
        self.summand, self.mult = self.compute_renorms(
            raw_data_dir, dataset_type_train, dataset_type_validate
        )

    def compute_renorms(self, raw_data_dir, dataset_type_train, dataset_type_validate):
        if dataset_type_train == dataset_type_validate:
            return 0, 1
        X_la = np.load(os.path.join(raw_data_dir, "node_values.npy")).transpose(
            (1, 2, 0)
        )
        X_la = X_la.astype(np.float32)
        X_bay = np.load(os.path.join(raw_data_dir, "pems_node_values.npy")).transpose(
            (1, 2, 0)
        )
        X_bay = X_bay.astype(np.float32)

        dataset_norms_la = (
            np.mean(X_la, axis=(0, 2))[None, None, 0:1, None],
            np.std(X_la, axis=(0, 2))[None, None, 0:1, None],
        )  # single feature
        dataset_norms_bay = (
            np.mean(X_bay, axis=(0, 2))[None, None, 0:1, None],
            np.std(X_bay, axis=(0, 2))[None, None, 0:1, None],
        )

        if dataset_type_validate == "la":
            s1, m1 = dataset_norms_la  # rescale la to normal
            s2, m2 = dataset_norms_bay  # and then normalize by bay
        else:
            s1, m1 = dataset_norms_bay  # rescale bay to normal
            s2, m2 = dataset_norms_la  # and then normalize by la

        return (m1 - m2) / s2, s1 / s2

    def rescale(self, y):
        return y * self.mult + self.summand
    def rerescale(self, y):
        return (y - self.summand) / self.mult


# =====================================
# =====================================
# Validate a DCRNN model on the METR-LA dataset
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
dataset_type_train = "bay"
dataset_type_validate = "la"  # 'la' or 'bay'
test_proportion_dataset = 0.2

# Training
num_workers = 1
batch_size = 32
resume_training = True

# Paths
logs_path = "runs/logs"

assert dataset_type_train in ["la", "bay"]
checkpoint_path = (
    "runs/model_checkpoint_dcrnn_no_skip.pth"
    if dataset_type_train == "bay"
    else "runs/model_checkpoint_dcrnn_no_skip_LA.pth"
)


# =====================================
# Data
# =====================================
if __name__ == "__main__":
    datasets = {"la": get_metr_la_dataset, "bay": get_pems_bay_dataset}
    dataset = datasets[dataset_type_validate]()
    rescaler = Rescaler(dataset_type_train, dataset_type_validate)

    train_loader, val_loader, test_loader = get_loaders(
        dataset,
        val_ratio=0,
        test_ratio=test_proportion_dataset,
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

    final_epoch = 0
    iterations = 0
    best_val_loss = float("inf")
    writer = SummaryWriter(logs_path)

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_epoch = checkpoint["epoch"] + 1
        iterations = checkpoint["iterations"]
        best_val_loss = checkpoint["best_val_loss"]
        print(
            f"Loading trained that finished in epoch {final_epoch} with a validation loss of {best_val_loss:.4f}"
        )

    # Validation loop
    model.eval()
    metrics = Metrics(dataset_type_validate, device=device)
    with torch.no_grad():
        for s, data in enumerate(test_loader):
            x, edge_index, edge_weight, y = data
            edge_index = edge_index[0].to(device)
            edge_weight = edge_weight[0].to(device)

            x = rescaler.rescale(x)

            x = x.to(device)
            y = y.to(device)
            # print mean and std of x and y
            h = None
            y_hat = model(x, edge_index, edge_weight, h)

            y_hat = rescaler.rerescale(y_hat)

            metrics.update(y, y_hat)

            if s % (len(test_loader) // 100 + 1) == 0:
                print(
                    f"\r {s}/{len(test_loader)}, MSE: {metrics.compute()['MSE'].mean()}",
                    end="",
                )

    print()

    metrics = metrics.compute()

    print(f"Metrics: \n{metrics}")
