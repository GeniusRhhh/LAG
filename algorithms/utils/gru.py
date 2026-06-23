import torch
import torch.nn as nn

class GRULayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(GRULayer, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)
        # NOTE: self.gru(x, hxs) needs x=[T, N, input_size] and hxs=[L, N, hidden_size]

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor):
        # NOTE: N = mini_batch_size; T = recurrent chunk length; L = gru layers

        # Ensure x and hxs are on the same device
        device = x.device
        hxs = hxs.to(device)
        masks = masks.to(device)

        # (T=1) x: [N, input_size], hxs: [N, L, hidden_size], masks: [N, 1]
        if x.size(0) == hxs.size(1):  # Check batch_size match
            # Reshape hxs to [L, N, hidden_size]
            hxs = hxs.transpose(1, 2).contiguous().view(self._num_layers, -1, self._hidden_size)
            # masks: [N, 1] => [L, N, 1]
            masks_expanded = masks.unsqueeze(0).repeat(self._num_layers, 1, 1)
            x, hxs = self.gru(x.unsqueeze(0), hxs * masks_expanded)  # x: [1, N, input_size], hxs: [L, N, hidden_size]
            x = x.squeeze(0)            # [1, N, input_size] => [N, input_size]
            hxs = hxs.transpose(0, 1).contiguous()  # [L, N, hidden_size] => [N, L, hidden_size]

        # (T>1): x=[T * N, input_size], hxs=[N, L, hidden_size], masks=[T * N, 1]
        else:
            N = hxs.size(0)
            T = int(x.size(0) / N)
            # unflatten x and masks
            x = x.view(T, N, x.size(1))  # [T * N, input_size] => [T, N, input_size]
            masks = masks.view(T, N)     # [T * N, 1] => [T, N]

            # Let's figure out which steps in the sequence have a zero for any agent
            has_zeros = ((masks[1:] == 0.0)
                        .any(dim=-1)       # [T, N] => [T]
                        .nonzero(as_tuple=False)
                        .squeeze(dim=-1)   # [T]
                        .cpu())
            has_zeros = (has_zeros + 1).numpy().tolist()
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(1, 2).contiguous().view(self._num_layers, -1, self._hidden_size)  # [N, L, hidden_size] => [L, N, hidden_size]
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = hxs * masks[start_idx].view(1, -1, 1).repeat(self._num_layers, 1, 1)
                rnn_scores, hxs = self.gru(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)
            x = torch.cat(outputs, dim=0)
            x = x.view(T * N, -1)       # [T, N, input_size] => [T * N, input_size]
            hxs = hxs.transpose(0, 1).contiguous()  # [L, N, hidden_size] => [N, L, hidden_size]

        x = self.norm(x)
        return x, hxs

    @property
    def output_size(self):
        return self._hidden_size