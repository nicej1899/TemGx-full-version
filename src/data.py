import torch
from torch.utils.data import DataLoader

from src.torch_geometric_temporal.metr_la import METRLADatasetLoader
from src.torch_geometric_temporal.pems_bay import PemsBayDatasetLoader
from src.torch_geometric_temporal.train_test_split import temporal_signal_split


class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, temporal_data):
        self.temporal_data = temporal_data

    def __len__(self):
        return self.temporal_data.snapshot_count

    def __getitem__(self, idx):
        snapshot = self.temporal_data[idx]
        return snapshot.x, snapshot.edge_index, snapshot.edge_attr, snapshot.y


def collate_fn(batch):
    x, edge_index, edge_weight, y = zip(*batch)
    return (
        torch.stack(x),
        torch.stack(edge_index),
        torch.stack(edge_weight),
        torch.stack(y),
    )


def get_pems_bay_dataset():
    try:
        loader = PemsBayDatasetLoader()
        dataset = loader.get_dataset()
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        print("Trying to download the dataset without SSL certificate verification")
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        loader = PemsBayDatasetLoader()
        dataset = loader.get_dataset()
    return dataset


def get_metr_la_dataset():
    try:
        loader = METRLADatasetLoader()
        dataset = loader.get_dataset()
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        print("Trying to download the dataset without SSL certificate verification")
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        loader = METRLADatasetLoader()
        dataset = loader.get_dataset()
    return dataset


def get_loaders(
    dataset,
    val_ratio=0.1,
    test_ratio=0.2,
    proportion_original_dataset=0.01,
    batch_size=16,
    num_workers=1,
) -> tuple:
    _, dataset = temporal_signal_split(
        dataset, train_ratio=1 - proportion_original_dataset
    )
    train_dataset_, test_dataset = temporal_signal_split(
        dataset, train_ratio=1 - test_ratio
    )
    train_dataset, val_dataset = temporal_signal_split(
        train_dataset_, train_ratio=1 - val_ratio
    )

    print(f"Train dataset length: {train_dataset.snapshot_count}")
    print(f"Validation dataset length: {val_dataset.snapshot_count}")
    print(f"Test dataset length: {test_dataset.snapshot_count}")

    train_loader = DataLoader(
        TemporalDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        TemporalDataset(val_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        TemporalDataset(test_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
