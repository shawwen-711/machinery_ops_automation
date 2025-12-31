#
#  Copyright 2025 Battelle Energy Alliance, LLC.  All rights reserved.
# 

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------
# 1. TemporalBlock (using BatchNorm version)
# -------------------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if (in_channels != out_channels) or (stride > 1):
            self.downsample = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # First convolution block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Second convolution block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection (with downsampling if needed)
        if self.downsample is not None:
            residual = self.downsample(residual)

        # Adjust temporal dimension if needed
        if out.size(2) != residual.size(2):
            diff = residual.size(2) - out.size(2)
            if diff > 0:
                out = F.pad(out, (0, diff))
            else:
                out = out[:, :, :residual.size(2)]
        return out + residual

# -------------------------------------------------------------------
# 2. TemporalConvNet with custom_dilations support
# -------------------------------------------------------------------
class TemporalConvNet(nn.Module):
    def __init__(self, num_features, kernel_size, num_layers, num_channels, dropout=0.2, 
                 downsample_layer_index=None, custom_dilations=None):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(num_layers):
            # Use custom dilations if provided and valid, otherwise default to 2**i
            if custom_dilations is not None and len(custom_dilations) == num_layers:
                dilation_size = custom_dilations[i]
            else:
                dilation_size = 2 ** i

            in_channels = num_features if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            stride = 2 if (downsample_layer_index is not None and i in downsample_layer_index) else 1
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# -------------------------------------------------------------------
# 3. TCNSingleTarget: Each branch predicting one target, with a final linear output.
#    (We keep this as is, so that each branch has a dense layer to produce its output.)
# -------------------------------------------------------------------
class TCNSingleTarget(nn.Module):
    def __init__(self, input_dim, kernel_size, num_layers, num_channels=None, dropout=0.2, 
                 downsample_layer_index=None, custom_dilations=None):
        super(TCNSingleTarget, self).__init__()
        if num_channels is None:
            num_channels = [64] * num_layers
        self.tcn = TemporalConvNet(
            num_features=input_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            num_channels=num_channels,
            dropout=dropout,
            downsample_layer_index=downsample_layer_index,
            custom_dilations=custom_dilations
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(num_channels[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.tcn(x)               
        h = self.global_pool(h).squeeze(-1)  
        out = self.dense(h)           
        return out

# -------------------------------------------------------------------
# 4. TCNMultiTarget: Updated to include all features (including target's own history)
#    in the input feature space.
# -------------------------------------------------------------------
class TCNMultiTarget(nn.Module):
    def __init__(self, all_vars, target_vars, 
                 kernel_size=3, num_layers=5, 
                 num_channels=[32, 32, 32, 64, 64], # RF ~511
                 custom_dilations=[8, 16, 33, 66, 132],
                 dropout=0.2, 
                 downsample_layer_index=None ):
        """
        all_vars: list of all variable names (controls + states)
        target_vars: list of state variable names that are targets
        custom_dilations: list of custom dilation factors (length must equal num_layers)
        """
        super(TCNMultiTarget, self).__init__()
        self.all_vars = all_vars
        self.target_vars = target_vars
        # Precompute indices of state variables within the full feature list (if needed for logging)
        self.target_indices = {target: all_vars.index(target) for target in target_vars}
        # For each branch, we now use the full input feature space, so:
        input_dim = len(all_vars)  # all features are used now

        # Create safe keys for ModuleDict by replacing dots and spaces with underscores.
        self.target_keys = {target: target.replace('.', '_').replace(' ', '_') for target in target_vars}
        
        self.target_branches = nn.ModuleDict({
            self.target_keys[target]: TCNSingleTarget(
                input_dim=input_dim,
                kernel_size=kernel_size,
                num_layers=num_layers,
                num_channels=num_channels,
                dropout=dropout,
                downsample_layer_index=downsample_layer_index,
                custom_dilations=custom_dilations
            )
            for target in target_vars
        })

    def forward(self, x):
        """
        x: Tensor of shape (batch, len(all_vars), seq_length)
        Returns: dict mapping each state variable (target) to its prediction of shape (batch, 1)
        """
        outputs = {}
        # Instead of removing the target channel, we now pass the full input x to every branch.
        for target in self.target_vars:
            safe_key = self.target_keys[target]
            outputs[target] = self.target_branches[safe_key](x)
        return outputs