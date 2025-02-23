import torch
import torch.nn as nn
import einops
class MLP(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, is_amortize_Q=False,
                 partition_init=150.0, sac=False, **kwargs):
        super().__init__()

        input_dim = num_tokens * max_len
        self.input = nn.Linear(input_dim, num_hid)

        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        self.hidden = nn.Sequential(*hidden_layers)

        self.output = nn.Linear(num_hid, num_outputs)

        self.max_len = max_len
        self.num_tokens = num_tokens

        if not sac:
            self._Z = nn.Parameter(torch.ones(64) * partition_init / 64)

    @property
    def Z(self):
        return self._Z.sum()

    def model_params(self):
        return self.parameters()

    def Z_param(self):
        return [self._Z]

    def forward(self, x, mask, return_all=False, lens=None):
        # x: batch_size, max_len x num_tokens
        if return_all:
            outputs = []
            for i in range(lens[0]):  # seq_len
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens * i),
                                  torch.zeros(x.shape[0], self.num_tokens * (lens[0] - i))), axis=1)
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

    def forward_next_states(self, x, mask, return_all=False, lens=None):
        # x: batch_size, max_len x num_tokens
        if return_all:
            outputs = []
            for i in range(lens[0]):  # seq_len
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens * (i + 1)),
                                  torch.zeros(x.shape[0], self.num_tokens * (lens[0] - (i + 1)))), axis=1)
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

    def forward_rnd(self, x, mask, return_all=False, lens=None):
        # x: batch_size, max_len x num_tokens
        if return_all:
            outputs = []
            for i in range(lens[0]):  # seq_len
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens * (i + 1)),
                                  torch.zeros(x.shape[0], self.num_tokens * (lens[0] - (i + 1)))), axis=1)
                mask = mask.to(x.device)
                #import pdb;pdb.set_trace()
                masked_input = mask * x[:,-1]

                out = self.input(masked_input)
                out = self.hidden(out)
                out = self.output(out)

                outputs.append(out.unsqueeze(0))
            outputs = torch.cat(outputs, axis=0)
            
            return einops.rearrange(outputs,'l B d -> B l d')

        out = self.input(x)
        out = self.hidden(out)
       
        return self.output(out)[:,:-1]
class GFNMLP(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, partition_init=150.0, causal=False):
        super().__init__()

        self.input = nn.Linear(num_tokens * max_len, num_hid)

        hidden_layers = []
        for _ in range(num_layers-1):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        
        self.hidden = nn.Sequential(*hidden_layers)

        self.output_f = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_outputs))
        self.output_b = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, 2))

        self.num_tokens = num_tokens

        self._Z = nn.Parameter(torch.ones(64) * partition_init / 64)

    @property
    def Z(self):
        return self._Z.sum()

    def model_params(self):
        params = list(self.input.parameters()) + list(self.hidden.parameters()) + list(
            self.output_f.parameters()) + list(self.output_b.parameters())
        return params

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
                out = self.output_f(out)
                outputs.append(out.unsqueeze(0))
            outputs = torch.cat(outputs, axis=0)
            return outputs

        out = self.input(x)
        out = self.hidden(out)
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        return p_f, p_b

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
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        return p_f, p_b

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
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        return p_f,p_b


class GFNConditionalMLP(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, arch='v0', input_dim=None): 
        super().__init__()

        self.input = nn.Linear(input_dim if input_dim is not None else num_tokens * max_len * 2, num_hid)

        hidden_layers = []
        for layer_idx in range(num_layers-1):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        
        self.hidden = nn.Sequential(*hidden_layers)
        # self.output = nn.Sequential(
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_outputs+2))
        
        self.output_f = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_outputs))
        self.output_b = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, 2))

        self.num_tokens = num_tokens

    def model_params(self):
        params = list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output_f.parameters()) + list(self.output_b.parameters())
        return params

    def forward(self, x, outcome, mask, return_all=False, lens=None):
        # x: batch_size, str_len x num_tokens
        if return_all:
            outputs = []
            for i in range(lens[0]):
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens //2 * i), torch.zeros(x.shape[0], self.num_tokens //2 * (lens[0] - i))), axis=1)
                mask = mask.to(x.device)

                masked_input = torch.cat((mask * x, outcome), -1)
                out = self.input(masked_input)

                out = self.hidden(out)
                out = self.output_f(out)
                outputs.append(out.unsqueeze(0))
            outputs = torch.cat(outputs, axis=0)
            return outputs

        xy = torch.cat((x, outcome), -1)
        out = self.input(xy)
        out = self.hidden(out)
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # p_f = self.output_f(out)
        # p_b = self.output_b(out)
        return p_f, p_b#self.output(out)

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
        #import pdb;pdb.set_trace()
        #print(xy.shape)
        out = self.input(xy)
        out = self.hidden(out)
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # out_b = self.hidden_b(out)
        # out_f = self.hidden_f(out)
        # p_f = self.output_f(out_f)
        # p_b = self.output_b(out_b)
        return p_f, p_b  # self.output(out)
    

    def forward_amortize(self, x, outcome, mask, return_all=False, lens=None):
        # x: batch_size, str_len x num_tokens
        if return_all:
            outputs = []
            for i in range(lens[0] + 1):
                mask = torch.cat((torch.ones(x.shape[0], self.num_tokens//2 * i), torch.zeros(x.shape[0], self.num_tokens//2 * (lens[0] - i))), axis=1)
                mask = mask.to(x.device)
                
                masked_input = torch.cat((mask * x, outcome[i]), -1)
                
                out = self.input(masked_input)
                out = self.hidden(out)
                out = self.output_f(out)
                outputs.append(out.unsqueeze(0))
            outputs = torch.cat(outputs, axis=0)
            return outputs
        xy = torch.cat((x, outcome), -1)
        #import pdb;pdb.set_trace()
        #print(xy.shape)
        out = self.input(xy)
        out = self.hidden(out)
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # out_b = self.hidden_b(out)
        # out_f = self.hidden_f(out)
        # p_f = self.output_f(out_f)
        # p_b = self.output_b(out_b)
        return p_f, p_b  # self.output(out)
class GFNConditionalMLPV1(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, arch='v0', input_dim=None): 
        super().__init__()

        self.input = nn.Linear(input_dim if input_dim is not None else num_tokens * max_len * 2, num_hid)

        hidden_layers = []
        for layer_idx in range(num_layers-1):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        
        self.hidden = nn.Sequential(*hidden_layers)
        # self.output = nn.Sequential(
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_outputs+2))
        
        self.output_f = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_outputs))
        self.output_b = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, 2))

        self.num_tokens = num_tokens

    def model_params(self):
        params = list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output_f.parameters()) + list(self.output_b.parameters())
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
        #out = self.hidden(out)
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # p_f = self.output_f(out)
        # p_b = self.output_b(out)
        return p_f, p_b#self.output(out)

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
        #import pdb;pdb.set_trace()
        #print(xy.shape)
        out = self.input(xy)
        #out = self.hidden(out)
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # out_b = self.hidden_b(out)
        # out_f = self.hidden_f(out)
        # p_f = self.output_f(out_f)
        # p_b = self.output_b(out_b)
        return p_f, p_b  # self.output(out)
class GFNConditionalMLPV2(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, arch='v0', input_dim=None): 
        super().__init__()

        self.input = nn.Linear(input_dim if input_dim is not None else num_tokens * max_len * 2, num_hid)

        hidden_layers = []
        for layer_idx in range(num_layers):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        
        self.hidden = nn.Sequential(*hidden_layers)
        # self.output = nn.Sequential(
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_outputs+2))
        
        self.output_f = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_outputs))
        self.output_b = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, 2))

        self.num_tokens = num_tokens

    def model_params(self):
        params = list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output_f.parameters()) + list(self.output_b.parameters())
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
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # p_f = self.output_f(out)
        # p_b = self.output_b(out)
        return p_f, p_b#self.output(out)

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
        #import pdb;pdb.set_trace()
        #print(xy.shape)
        out = self.input(xy)
        out = self.hidden(out)
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # out_b = self.hidden_b(out)
        # out_f = self.hidden_f(out)
        # p_f = self.output_f(out_f)
        # p_b = self.output_b(out_b)
        return p_f, p_b  # self.output(out)
class GFNConditionalMLPV3(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, arch='v0', input_dim=None): 
        super().__init__()

        self.input = nn.Linear(input_dim if input_dim is not None else num_tokens * max_len * 2, num_hid)

        hidden_layers = []
        for layer_idx in range(num_layers+1):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        
        self.hidden = nn.Sequential(*hidden_layers)
        # self.output = nn.Sequential(
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_outputs+2))
        
        self.output_f = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_outputs))
        self.output_b = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, 2))

        self.num_tokens = num_tokens

    def model_params(self):
        params = list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output_f.parameters()) + list(self.output_b.parameters())
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
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # p_f = self.output_f(out)
        # p_b = self.output_b(out)
        return p_f, p_b#self.output(out)

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
        #import pdb;pdb.set_trace()
        #print(xy.shape)
        out = self.input(xy)
        out = self.hidden(out)
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # out_b = self.hidden_b(out)
        # out_f = self.hidden_f(out)
        # p_f = self.output_f(out_f)
        # p_b = self.output_b(out_b)
        return p_f, p_b  # self.output(out)
class GFNConditionalTCN(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, arch='v0', input_dim=None): 
        super().__init__()
        self.max_len = max_len
        self.num_tokens = num_tokens
        self.conv1 = torch.nn.Conv1d(in_channels=2*num_tokens, out_channels=32, kernel_size=(5,), stride=(1,), padding=0, dilation=(1,), groups=1,
                                     bias=True, padding_mode='zeros', device=None, dtype=None)
        self.relu1 = nn.ReLU()

        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(5,), stride=(1,), padding=0,
                                     dilation=(1,), groups=1,
                                     bias=True, padding_mode='zeros', device=None, dtype=None)

        self.relu2 = nn.ReLU()

        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(5,), stride=(1,), padding=0,
                                     dilation=(1,), groups=1,
                                     bias=True, padding_mode='zeros', device=None, dtype=None)
        self.relu3 = nn.ReLU()
        # self.output = nn.Sequential(
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_outputs+2))
        
        self.output_f = nn.Sequential(
            nn.Linear(256, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            
            nn.Linear(num_hid, num_outputs))
        self.output_b = nn.Sequential(
            nn.Linear(256, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, 2))

        self.num_tokens = num_tokens

    def model_params(self):
        # params = list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output_f.parameters()) + list(self.output_b.parameters())
        return self.parameters()

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
        flag = False
        if x.ndim == 3:
            flag = True
            x = einops.rearrange(x, 'B L (a b)->(B L) a b',a=self.max_len,b=self.num_tokens)
            outcome = einops.rearrange(outcome, 'B L (a b)->(B L) a b',a=self.max_len,b=self.num_tokens)
        else:
            x = einops.rearrange(x, 'B (a b)->B a b',a=self.max_len,b=self.num_tokens)
            outcome = einops.rearrange(outcome, 'B (a b)->B a b',a=self.max_len,b=self.num_tokens)
        xy = torch.cat((x, outcome), -1)
        xy = einops.rearrange(xy, 'B L C->B C L')
        out = self.conv1(xy)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.flatten(1)
        
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # p_f = self.output_f(out)
        # p_b = self.output_b(out)
        return p_f, p_b#self.output(out)

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
        if x.ndim == 3:
            flag = True
            x = einops.rearrange(x, 'B L (a b)->(B L) a b',a=self.max_len,b=self.num_tokens)
            outcome = einops.rearrange(outcome, 'B L (a b)->(B L) a b',a=self.max_len,b=self.num_tokens)
        else:
            x = einops.rearrange(x, 'B (a b)->B a b',a=self.max_len,b=self.num_tokens)
            outcome = einops.rearrange(outcome, 'B (a b)->B a b',a=self.max_len,b=self.num_tokens)
        xy = torch.cat((x, outcome), -1)
        xy = einops.rearrange(xy, 'B L C->B C L')
        out = self.conv1(xy)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.flatten(1)
        if flag:
            out = einops.rearrange(out, '(B L) D-> B L D', L=self.max_len+1)
        p_f = self.output_f(out)
        p_b = self.output_b(out)
        
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # out_b = self.hidden_b(out)
        # out_f = self.hidden_f(out)
        # p_f = self.output_f(out_f)
        # p_b = self.output_b(out_b)
        return p_f, p_b  # self.output(out)
class QNetwork(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, arch='v0', input_dim=None): 
        super().__init__()
        self.max_len = max_len
        self.num_tokens = num_tokens
        self.conv1 = torch.nn.Conv1d(in_channels=2*num_tokens, out_channels=32, kernel_size=(5,), stride=(1,), padding=0, dilation=(1,), groups=1,
                                     bias=True, padding_mode='zeros', device=None, dtype=None)
        self.relu1 = nn.ReLU()

        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(5,), stride=(1,), padding=0,
                                     dilation=(1,), groups=1,
                                     bias=True, padding_mode='zeros', device=None, dtype=None)

        self.relu2 = nn.ReLU()

        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(5,), stride=(1,), padding=0,
                                     dilation=(1,), groups=1,
                                     bias=True, padding_mode='zeros', device=None, dtype=None)
        self.relu3 = nn.ReLU()
        self.output = nn.Sequential(
            nn.Linear(256, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_outputs))
        
        
        self.num_tokens = num_tokens

    def model_params(self):
        # params = list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output_f.parameters()) + list(self.output_b.parameters())
        return self.parameters()

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
        flag = False
        if x.ndim == 3:
            flag = True
            x = einops.rearrange(x, 'B L (a b)->(B L) a b',a=self.max_len,b=self.num_tokens)
            outcome = einops.rearrange(outcome, 'B L (a b)->(B L) a b',a=self.max_len,b=self.num_tokens)
        else:
            x = einops.rearrange(x, 'B (a b)->B a b',a=self.max_len,b=self.num_tokens)
            outcome = einops.rearrange(outcome, 'B (a b)->B a b',a=self.max_len,b=self.num_tokens)
        xy = torch.cat((x, outcome), -1)
        xy = einops.rearrange(xy, 'B L C->B C L')
        out = self.conv1(xy)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.flatten(1)
        
        # p_f = self.output_f(out)
        # p_b = self.output_b(out)
        
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # p_f = self.output_f(out)
        # p_b = self.output_b(out)
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
        if x.ndim == 3:
            flag = True
            x = einops.rearrange(x, 'B L (a b)->(B L) a b',a=self.max_len,b=self.num_tokens)
            outcome = einops.rearrange(outcome, 'B L (a b)->(B L) a b',a=self.max_len,b=self.num_tokens)
        else:
            x = einops.rearrange(x, 'B (a b)->B a b',a=self.max_len,b=self.num_tokens)
            outcome = einops.rearrange(outcome, 'B (a b)->B a b',a=self.max_len,b=self.num_tokens)
        xy = torch.cat((x, outcome), -1)
        xy = einops.rearrange(xy, 'B L C->B C L')
        out = self.conv1(xy)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.flatten(1)
        if flag:
            out = einops.rearrange(out, '(B L) D-> B L D', L=self.max_len)
        # p_f = self.output_f(out)
        # p_b = self.output_b(out)
        
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # out_b = self.hidden_b(out)
        # out_f = self.hidden_f(out)
        # p_f = self.output_f(out_f)
        # p_b = self.output_b(out_b)
        return self.output(out)
class QNetworkMLP(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid, num_layers, max_len=60, dropout=0.1, arch='v0', input_dim=None): 
        super().__init__()

        self.input = nn.Linear(input_dim if input_dim is not None else num_tokens * max_len * 2, num_hid)

        hidden_layers = []
        for layer_idx in range(num_layers-1):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        
        self.hidden = nn.Sequential(*hidden_layers)
        # self.output = nn.Sequential(
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_hid),
        #     nn.ReLU(),
        #     nn.Linear(num_hid, num_outputs+2))
        
        self.output = nn.Sequential(
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_outputs))
        

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
       
        # out = self.output(out)
        # p_f, p_b = out[...,:-2], out[...,-2:]
        # p_f = self.output_f(out)
        # p_b = self.output_b(out)
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