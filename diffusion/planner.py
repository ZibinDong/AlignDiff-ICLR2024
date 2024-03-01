import torch
from .ode import ODE
from utils import AttrFunc, get_grid_indices

class Planner():
    def __init__(self, 
        ode: ODE, attr_func: AttrFunc, normalizer, max_attr, min_attr):
        self.ode = ode
        self.attr_func = attr_func
        self.normalizer = normalizer
        self.max_attr, self.min_attr = max_attr, min_attr
        self.device = ode.device
        self.attr_cache, self.attr = None, None
        self.mask_cache, self.mask = None, None
        
    @torch.no_grad()
    def plan(self, o, traj_len, attr, mask, 
        n_samples = 64, w = 1.5, sample_steps = 5):
        if self.attr_cache is None or any([x-y for x,y in zip(attr,self.attr_cache)]):
            self.attr_cache = attr
            self.attr = torch.tensor(attr, dtype=torch.float32, device=self.device)[None,]
            self.disc_attr = get_grid_indices(self.attr, 0, 1, 100)
        if self.mask_cache is None or any([x-y for x,y in zip(mask,self.mask_cache)]):
            self.mask_cache = mask
            self.mask = torch.tensor(mask, dtype=torch.float32, device=self.device)[None,]
        o = torch.tensor(o, dtype=torch.float32, device=self.device)[None,]
        o = self.normalizer.normalize(o)
        x = self.ode.sample(o, self.disc_attr, self.mask, traj_len, n_samples, w, sample_steps)
        pred_attr = self.attr_func.predict_attr(x[:,:,:self.ode.o_dim], None)
        nor_pred_attr = (pred_attr - self.min_attr[None,]) / (self.max_attr - self.min_attr)[None,]
        mse = ((nor_pred_attr-self.attr)**2)*self.mask
        idx = mse.sum(-1).argmin().item()
        act = x[idx,0,self.ode.o_dim:]
        act = act.clip(-1.+1e-8,1.-1e-8).cpu().numpy()
        return act, nor_pred_attr[idx]

