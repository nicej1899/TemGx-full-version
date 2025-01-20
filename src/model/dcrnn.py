import torch
from torch import nn
from src.torch_geometric_temporal.dcrnn import DCRNN as DCRNN_TG
import torch.nn.functional as F


class DCRNN(nn.Module):
    def __init__(
        self,
        node_features,
        out_channels,
        K,
        hidden_state=None,
        stash_adj_matrix=False,
        predicted_time=None,
        predicted_feature=None,
    ):
        """
        Args:
            node_features (int): Number of input features.
            out_channels (int): Number of output features.
            K (int): Filter size :math:`K`.
        """

        if hidden_state is None:
            hidden_state = out_channels

        super(DCRNN, self).__init__()
        self.recurrent_encoder_1 = DCRNN_TG(
            node_features, hidden_state, K, stash_adj_matrix=stash_adj_matrix
        )
        self.recurrent_encoder_2 = DCRNN_TG(
            hidden_state, out_channels, K, stash_adj_matrix=stash_adj_matrix
        )
        self.recurrent_decoder_1 = DCRNN_TG(
            node_features, hidden_state, K, stash_adj_matrix=stash_adj_matrix
        )
        self.recurrent_decoder_2 = DCRNN_TG(
            hidden_state, out_channels, K, stash_adj_matrix=stash_adj_matrix
        )
        self.linear = torch.nn.Linear(out_channels, node_features)
        self.predicted_time = predicted_time
        self.predicted_feature = predicted_feature

    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        time_steps=None,
        h_1=None,
        h_2=None,
        training_target=None,
        target_sample_prob=0,
    ):
        """
        Args:
            x (Tensor): The input features [num_nodes, num_features, P]
            edge_index (LongTensor): The edge indices [2, num_edges]
            edge_weight (Tensor): The edge weights [num_edges]
            h (Tensor, optional): The hidden state [num_nodes, out_channels]
        """
        # if target_sample_prob is bigger than 0 then training_target must be provided
        if target_sample_prob > 0 and training_target is None:
            raise ValueError(
                "training_target must be provided when target_sample_prob is bigger than 0"
            )

        untimed = False
        if time_steps is not None:
            x = x.reshape(*x.shape[:-1], -1, time_steps)
            untimed = True

        unbatched = False
        if len(x.shape) == 3:
            unbatched = True
            x = x.unsqueeze(0)

        batch_size, num_nodes, num_features, P = x.shape
        # num_nodes, num_features, P = x.shape

        for t in range(P):
            h_1 = self.recurrent_encoder_1(x[:, :, :, t], edge_index, edge_weight, h_1)
            input_2 = F.relu(h_1)
            h_2 = self.recurrent_encoder_2(input_2, edge_index, edge_weight, h_2)

        out_seq = []
        out = x[:, :, :, P - 1]
        for t in range(P):
            h_1 = self.recurrent_decoder_1(out, edge_index, edge_weight, h_1)
            input_2 = F.relu(h_1)
            h_2 = self.recurrent_decoder_2(input_2, edge_index, edge_weight, h_2)
            out = self.linear(h_2)
            out_seq.append(out)

            # random sample
            if target_sample_prob > 0 and torch.rand(1) < target_sample_prob:
                out = training_target[:, :, :, t]
            else:
                out = out


        out_seq = torch.stack(
            out_seq, dim=3
        )  # [batch_size, num_nodes, num_features, P]
        if unbatched:
            out_seq = out_seq.squeeze(0)
        if untimed:
            if self.predicted_time is not None and self.predicted_feature is not None:
                out_seq = out_seq[:, self.predicted_feature, self.predicted_time]
            out_seq = out_seq.reshape(*out_seq.shape[:-2], -1)

        return out_seq
