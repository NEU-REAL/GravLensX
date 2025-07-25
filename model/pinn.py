import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class GeoBlock(nn.Module):
    def __init__(self, in_dim, out_dim, inter_dim_factor=1, drop_ratio=0.1, activation=F.softplus):
        super().__init__()
        inter_dim = inter_dim_factor * out_dim
        self.fc_mul_in = nn.Linear(in_dim, inter_dim)
        self.fc_mul_out = nn.Linear(inter_dim, out_dim)
        self.dropout = nn.Dropout(drop_ratio)
        self.activation = activation

    def forward(self, x):
        x0 = x
        x = self.activation(self.fc_mul_in(x))
        x = self.dropout(self.fc_mul_out(x))
        return x + x0


def positional_encoding(x, num_encoding=16):
    if len(x.shape) == 2:
        freqs = torch.pow(2, torch.arange(num_encoding).float()).unsqueeze(0).unsqueeze(0).to(x.device)
        x = x.unsqueeze(2)
        x = torch.cat([torch.sin(freqs * x), torch.cos(freqs * x), x], dim=2).flatten(1, 2)
    elif len(x.shape) == 1:
        freqs = torch.pow(2, torch.arange(num_encoding).float()).unsqueeze(0).to(x.device)
        x = x.unsqueeze(1)
        x = torch.cat([torch.sin(freqs * x), torch.cos(freqs * x), x], dim=1).flatten(0, 1)
    return x



class GeodesicNet(nn.Module):
    def __init__(self, num_blocks=4, in_dim=127, hidden_dim=128, inter_dim_factor=1, drop_ratio=0.1, activation=F.softplus):
        super(GeodesicNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.geo_blocks = nn.ModuleList([
            GeoBlock(hidden_dim, hidden_dim, inter_dim_factor, drop_ratio, activation) for _ in range(num_blocks)
        ])
        # self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)
        self.init_weights()

    def forward(self, x):
        x = self.fc1(x)
        for blk in self.geo_blocks:
            x = blk(x)
        out = self.fc2(x)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)


def calculate_speed(network, init_p, direction, tau, norm_radius=None):
    tau.requires_grad = True
    input_data = torch.cat([init_p, direction, tau], dim=1)
    output_data = network(input_data)
    if norm_radius is not None:
        output_data = output_data * norm_radius
    predict_v = []
    for x_idx in range(3):
        dx_dtau = torch.autograd.grad(output_data[:, x_idx], tau, grad_outputs=torch.ones_like(output_data[:, 0]), create_graph=True, retain_graph=True)[0]
        predict_v.append(dx_dtau)
    predict_v = torch.cat(predict_v, dim=1)
    return output_data, predict_v


def calculate_speed_acc(network, init_p, direction, tau):
    tau.requires_grad = True
    input_data = torch.cat([init_p, direction, tau], dim=1)
    output_data = network(input_data)

    predict_v = []  # List to store the first derivatives
    predict_a = []  # List to store the second derivatives

    for x_idx in range(output_data.shape[1]):
        dx_dtau = torch.autograd.grad(output_data[:, x_idx], tau,
                                      grad_outputs=torch.ones_like(output_data[:, x_idx]),
                                      create_graph=True, retain_graph=True)[0]
        predict_v.append(dx_dtau)
        d2x_dtau2 = torch.autograd.grad(dx_dtau, tau,
                                        grad_outputs=torch.ones_like(dx_dtau),
                                        create_graph=True, retain_graph=True)[0]
        predict_a.append(d2x_dtau2)

    predict_v = torch.cat(predict_v, dim=1)
    predict_a = torch.cat(predict_a, dim=1)
    return output_data, predict_v, predict_a