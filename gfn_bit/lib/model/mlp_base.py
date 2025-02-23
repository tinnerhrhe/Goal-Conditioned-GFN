import torch
import torch.nn as nn


class GFNMLP(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, partition_init=150.0, causal=False):
        super().__init__()

        self.input = nn.Linear(num_tokens * max_len, num_hid)

        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        
        self.hidden = nn.Sequential(*hidden_layers)
        
        self.output = nn.Linear(num_hid, num_outputs)

        self.num_tokens = num_tokens

        self._Z = nn.Parameter(torch.ones(64) * partition_init / 64)

    @property
    def Z(self):
        return self._Z.sum()

    def model_params(self):
        return list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output.parameters())

    def Z_param(self):
        return [self._Z]

    def forward(self, x, mask, return_all=False, lens=None):
        # x: batch_size, str_len x num_tokens
        if return_all:
            outputs = []
            for i in range(lens[0]):
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens * i), torch.zeros(x.shape[0], self.num_tokens * (lens[0] - i))), axis=1)
                mask = mask.to(x.device)

                # masked_input: batch_size, str_len x num_tokens
                masked_input = mask * x
                
                out = self.input(masked_input)
                out = self.hidden(out)
                out = self.output(out)
                outputs.append(out.unsqueeze(0))
            outputs = torch.cat(outputs, axis=0)
            return outputs

        out = self.input(x)
        out = self.hidden(out)
        return self.output(out)

    def forward_next_states(self, x, mask, return_all=False, lens=None):
        # x: batch_size, str_len x num_tokens
        if return_all:
            outputs = []
            for i in range(lens[0]):
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens * (i + 1)), torch.zeros(x.shape[0], self.num_tokens * (lens[0] - (i + 1)))), axis=1)
                mask = mask.to(x.device)

                # masked_input: batch_size, str_len x num_tokens
                masked_input = mask * x
                
                out = self.input(masked_input)
                out = self.hidden(out)
                out = self.output(out)
                outputs.append(out.unsqueeze(0))
            outputs = torch.cat(outputs, axis=0)
            return outputs

        out = self.input(x)
        out = self.hidden(out)
        return self.output(out)

    def forward_rnd(self, x, mask, return_all=False, lens=None):
        # x: batch_size, str_len x num_tokens
        if return_all:
            outputs = []
            for i in range(lens[0]): # str_len
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens * (i + 1)), torch.zeros(x.shape[0], self.num_tokens * (lens[0] - (i + 1)))), axis=1)
                mask = mask.to(x.device)
                
                masked_input = mask * x
                
                out = self.input(masked_input)
                out = self.hidden(out)
                out = self.output(out)

                outputs.append(out.unsqueeze(0))
            outputs = torch.cat(outputs, axis=0)
            return outputs
        
        out = self.input(x)
        out = self.hidden(out)
        return self.output(out)


class GFNConditionalMLP(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, arch='v0', input_dim=None): 
        super().__init__()

        self.input = nn.Linear(input_dim if input_dim is not None else num_tokens * max_len * 2, num_hid)

        hidden_layers = []
        for layer_idx in range(num_layers):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        
        self.hidden = nn.Sequential(*hidden_layers)
        
        self.output = nn.Linear(num_hid, num_outputs)

        self.num_tokens = num_tokens

    def model_params(self):
        params = list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output.parameters())
        return params

    def forward(self, x, outcome, mask, return_all=False, lens=None):
        # x: batch_size, str_len x num_tokens
        if return_all:
            outputs = []
            for i in range(lens[0]):
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens * i), torch.zeros(x.shape[0], self.num_tokens * (lens[0] - i))), axis=1)
                mask = mask.to(x.device)

                masked_input = torch.cat((mask * x, outcome), -1)
                out = self.input(masked_input)

                out = self.hidden(out)
                out = self.output(out)
                outputs.append(out.unsqueeze(0))
            outputs = torch.cat(outputs, axis=0)
            return outputs

        xy = torch.cat((x, outcome), -1)
        out = self.input(xy)
        out = self.hidden(out)
        return self.output(out)

    def forward_for_fl(self, x, outcome, mask, return_all=False, lens=None):
        # x: batch_size, str_len x num_tokens
        if return_all:
            outputs = []
            for i in range(lens[0] + 1):
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens * i), torch.zeros(x.shape[0], self.num_tokens * (lens[0] - i))), axis=1)
                mask = mask.to(x.device)

                masked_input = torch.cat((mask * x, outcome), -1)
                
                out = self.input(masked_input)
                out = self.hidden(out)
                out = self.output(out)
                outputs.append(out.unsqueeze(0))
            outputs = torch.cat(outputs, axis=0)
            return outputs
        xy = torch.cat((x, outcome), -1)
        out = self.input(xy)
        out = self.hidden(out)
        return self.output(out)
