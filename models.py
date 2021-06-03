import torch
import torch.nn as nn


class predictConcentrationModel(nn.Module):
    def __init__(self, feature_dim=29, output_dim=200, water_dim=2, device='cuda'):
        super(predictConcentrationModel, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.water_dim = water_dim
        self.hid_dim = 256
        self.device = device

        self.first_layer = nn.Linear(self.feature_dim, self.hid_dim)
        self.lstm_layer = nn.LSTM(input_size=self.hid_dim, hidden_size=self.hid_dim, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.hid_dim, self.output_dim)

        self.relu = nn.ReLU()

    def forward(self, obs_input, hidden, cell_state, time_step=None):
        hidden = hidden.to(self.device)
        cell_state = cell_state.to(self.device)
        obs = torch.tensor(obs_input).to(self.device)

        out = self.first_layer(obs)
        # print(out.shape)
        out, hidden_out = self.lstm_layer(out, (hidden, cell_state))
        hidden, cell_state = hidden_out
        # print(out.shape)
        out = self.output_layer(out)
        # print(out.shape)

        if time_step is not None:
            out = out[:, time_step, :]
        return out, hidden, cell_state

    # jsut remove the last output layer. What I think may cause the difference between the single input vs entire sequecne
    # the output from LSTM layer will refeed to itself
    # otherwise, the last full connected layer will change that output.
    def test_forward(self, obs_input, hidden, cell_state, time_step=None):
        hidden = hidden.to(self.device)
        cell_state = cell_state.to(self.device)
        obs = torch.tensor(obs_input).to(self.device)

        out = self.first_layer(obs)
        # print(out.shape)
        out, hidden_out = self.lstm_layer(out, (hidden, cell_state))
        hidden, cell_state = hidden_out
        # print(out.shape)
        # out = self.output_layer(out)
        # print(out.shape)

        if time_step is not None:
            out = out[:, time_step, :]
        return out, hidden, cell_state

    def init_hidden_states(self, bsize):
        h = torch.zeros(1, bsize, self.hid_dim).float()
        c = torch.zeros(1, bsize, self.hid_dim).float()

        return h, c
