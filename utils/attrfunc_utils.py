import torch
import torch.nn as nn
from .network_utils import SinusoidalPosEmb
from typing import Optional
from pathlib import Path
import os


class StateOnlyPrefTransformer(nn.Module):
    def __init__(self,
        o_dim: int, attr_dim: int, 
        d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        self.attr_dim = attr_dim
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(o_dim, d_model), nn.LayerNorm(d_model))
        self.attr_emb = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), num_layers)
        self.out_layer = nn.Linear(d_model, attr_dim)
        
    def forward(self, traj: torch.Tensor):
        batch_size, traj_len = traj.shape[:2]
        pos = self.pos_emb(torch.arange(traj_len, device=traj.device))[None,]
        obs = self.obs_emb(traj)
        x = self.transformer(torch.cat([obs+pos, self.attr_emb.repeat(batch_size,1,1)], 1))
        return self.out_layer(x[:, -1])
    
class AttrFunc(nn.Module):
    def __init__(self, 
        o_dim: int, attr_dim: int,
        attr_clip: float = 20., ensemble_size: int = 3,
        lr: float = 1e-4, weight_decay: float = 1e-4,
        d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.o_dim = o_dim
        self.attr_dim = attr_dim
        self.attr_clip = attr_clip
        self.ensemble_size = ensemble_size
        
        self.attr_func_ensemble = nn.ModuleList([
            StateOnlyPrefTransformer(o_dim, attr_dim, d_model, nhead, num_layers)
            for _ in range(self.ensemble_size)])
        self.optim = [torch.optim.AdamW(ensemble.parameters(), lr=lr, weight_decay=weight_decay)
            for ensemble in self.attr_func_ensemble]

    def _predict_attr_ensemble(self, traj: torch.Tensor, ensemble_idx: int = 0):
        '''
        Input:
            - traj: (batch_size, traj_len, o_dim)
            - ensemble_idx: int
            
        Output:
            - attr_strength: (batch_size, attr_dim) 
        '''
        traj_attr = self.attr_func_ensemble[ensemble_idx](traj)
        return self.attr_clip * torch.tanh(traj_attr / self.attr_clip)
    
    def predict_attr(self, traj: torch.Tensor, ensemble_idx: Optional[int] = None):
        '''
        Input:
            - traj: (batch_size, traj_len, o_dim)
            - ensemble_idx: int
            
        Output:
            - attr_strength: (batch_size, attr_dim) 
        '''
        if ensemble_idx is not None:
            return self._predict_attr_ensemble(traj, ensemble_idx)
        else:
            sum_ensemble = [self._predict_attr_ensemble(traj, i) for i in range(self.ensemble_size)]
            return sum(sum_ensemble) / self.ensemble_size
        
    def predict_pref_prob(self, 
            traj0: torch.Tensor, traj1: torch.Tensor, 
            ensemble_idx: Optional[int] = None):
        """
        Compute P[t_0 > t_1] = exp[sum(r(t_0))]/{exp[sum(r(t_0))]+exp[sum(r(t_1))]}= 1 /{1+exp[sum(r(t_1) - r(t_0))]}
        ----
        Input:
            - traj0: (batch_size, traj_len, o_dim)
            - traj1: (batch_size, traj_len, o_dim)
        
        Output:
            - prob: (batch_size, attr_dim)
        """
        traj_attr_strength_0 = self.predict_attr(traj0, ensemble_idx) # (batch_size, attr_dim)
        traj_attr_strength_1 = self.predict_attr(traj1, ensemble_idx) # (batch_size, attr_dim)
        a1_minus_a0 = traj_attr_strength_1 - traj_attr_strength_0
        prob = 1.0 / (1.0 + torch.exp(a1_minus_a0))
        return prob
    
    def update(self, 
            traj0: torch.Tensor, traj1: torch.Tensor, 
            pref: torch.Tensor,
            ensemble_idx: Optional[int] = None
        ):
        """
        Update the parameters of the attribute function by minimizing the negative log-likelihood loss
        ----
        Input:
            - traj0: (batch_size, traj_len, o_dim)
            - traj1: (batch_size, traj_len, o_dim)
            - pref: (batch_size, attr_dim) # 0 means traj0 is preferred and 1 means traj1 is preferred
            - ensemble_idx: int # which ensemble to update
        
        Output:
            - loss: float
        """
        prob = self.predict_pref_prob(traj0, traj1, ensemble_idx)
        loss = - ((1-pref)*torch.log(prob+1e-8) + pref*torch.log(1-prob+1e-8)).mean()
        self.optim[ensemble_idx].zero_grad()
        loss.backward()
        self.optim[ensemble_idx].step()
        return loss.item()
    
    def save(self, filename: str):
        file_path = Path(__file__).parents[1]/"results/attr_func"
        if not os.path.exists(file_path): os.makedirs(file_path)
        torch.save(self.state_dict(), file_path/(filename+".pt"))
        
    def load(self, filename: str, map_location = None):
        file_path = Path(__file__).parents[1]/"results/attr_func"
        self.load_state_dict(torch.load(file_path/(filename+".pt"), map_location=map_location))