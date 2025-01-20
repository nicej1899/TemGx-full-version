from typing import Tuple

def temporal_signal_split(
    data_iterator, train_ratio: float = 0.8
) -> Tuple["Discrete_Signal", "Discrete_Signal"]:
    r"""Function to split a data iterator according to a fixed ratio.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **train_ratio** *(float)* - Graph edge indices.

    Return types:
        * **(train_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train and test data iterators.
    """

    train_snapshots = int(train_ratio * data_iterator.snapshot_count)
    
    train_iterator = data_iterator[0:train_snapshots]
    test_iterator = data_iterator[train_snapshots:]

    return train_iterator, test_iterator