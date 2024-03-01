import os
import torch
import numpy as np
import torch.nn as nn
from utils import DiT1d
from pathlib import Path
from copy import deepcopy

    
class ODE():
    def __init__(self, 
        o_dim: int, a_dim, attr_dim: int, sigma_data: float,
        sigma_min: float = 0.002, sigma_max: float = 80,
        rho: float = 7, p_mean: float = -1.2, p_std: float = 1.2, 
        d_model: int = 384, n_heads: int = 6, depth: int = 12,
        device: str = "cpu"):
        x_dim = o_dim + a_dim
        self.sigma_data, self.sigma_min, self.sigma_max = sigma_data, sigma_min, sigma_max
        self.rho, self.p_mean, self.p_std = rho, p_mean, p_std
        self.o_dim, self.x_dim = o_dim, x_dim
        
        self.device, self.M = device, 1000
        self.F = DiT1d(x_dim, attr_dim, d_model=d_model, n_heads=n_heads, depth=depth, dropout=0.1).to(device)
        self.F.train()
        self.F_ema = deepcopy(self.F).requires_grad_(False)
        self.F_ema.eval()
        self.optim = torch.optim.AdamW(self.F.parameters(), lr=2e-4, weight_decay=1e-4)
        self.set_N(5)
        
    def ema_update(self, decay=0.999):
        for p, p_ema in zip(self.F.parameters(), self.F_ema.parameters()):
            p_ema.data = decay*p_ema.data + (1-decay)*p.data

    def set_N(self, N):
        self.N = N
        self.sigma_s = (self.sigma_max**(1/self.rho)+torch.arange(N, device=self.device)/(N-1)*\
            (self.sigma_min**(1/self.rho)-self.sigma_max**(1/self.rho)))**self.rho
        self.t_s = self.sigma_s
        self.scale_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_sigma_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        if self.t_s is not None:
            self.coeff1 = (self.dot_sigma_s/self.sigma_s+self.dot_scale_s/self.scale_s)
            self.coeff2 = self.dot_sigma_s/self.sigma_s*self.scale_s
            
    def c_skip(self, sigma): return self.sigma_data**2/(self.sigma_data**2+sigma**2)
    def c_out(self, sigma): return sigma*self.sigma_data/(self.sigma_data**2+sigma**2).sqrt()
    def c_in(self, sigma): return 1/(self.sigma_data**2+sigma**2).sqrt()
    def c_noise(self, sigma): return 0.25*(sigma).log()
    def loss_weighting(self, sigma): return (self.sigma_data**2+sigma**2)/((sigma*self.sigma_data)**2)
    def sample_noise_distribution(self, N):
        log_sigma = torch.randn((N,1,1),device=self.device)*self.p_std + self.p_mean
        return log_sigma.exp()
    
    def D(self, x, sigma, condition = None, mask = None, use_ema = False):
        c_skip, c_out, c_in, c_noise = self.c_skip(sigma), self.c_out(sigma), self.c_in(sigma), self.c_noise(sigma)
        F = self.F_ema if use_ema else self.F
        return c_skip*x+c_out*F(c_in*x, c_noise.squeeze(-1), condition, mask)
    
    def update(self, x, condition):
        sigma = self.sample_noise_distribution(x.shape[0])
        eps = torch.randn_like(x) * sigma
        # preserve the first obs
        eps[:,0,:self.o_dim] = 0.
        loss_mask = torch.ones_like(x)
        loss_mask[:,0,:self.o_dim] = 0.
        mask = (torch.rand(*condition.shape, device=self.device) > 0.2).float()
        loss = (loss_mask * self.loss_weighting(sigma) * (self.D(x + eps, sigma, condition, mask) - x)**2).mean()
        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.F.parameters(), 10.)
        self.optim.step()
        self.ema_update()
        return loss.item(), grad_norm.item()
    
    @torch.no_grad()
    def sample(self, o, attr, attr_mask, traj_len, n_samples: int, w: float = 1.5, N: int = None):
        if N is not None and N != self.N: self.set_N(N)
        x = torch.randn((n_samples,traj_len,self.x_dim),device=self.device) * self.sigma_s[0] * self.scale_s[0]
        x[:,0,:self.o_dim] = o
        attr = attr.repeat(2*n_samples,1)
        attr_mask = attr_mask.repeat(2*n_samples,1)
        attr_mask[n_samples:] = 0
        for i in range(self.N):
            with torch.no_grad():
                D = self.D(x.repeat(2,1,1)/self.scale_s[i], torch.ones((2*n_samples,1,1),device=self.device)*self.sigma_s[i],
                    attr, attr_mask, use_ema=True)
                D = w*D[:n_samples]+(1-w)*D[n_samples:]
            delta = self.coeff1[i]*x-self.coeff2[i]*D
            dt = self.t_s[i]-self.t_s[i+1] if i != self.N-1 else self.t_s[i]
            x = x - delta*dt
            x[:,0,:self.o_dim] = o
        return x
    
    def save(self, file_name):
        file_path = Path(__file__).parents[1]/"results/diffusion"
        if not os.path.exists(file_path): os.makedirs(file_path)
        torch.save({'model': self.F.state_dict(), 'model_ema': self.F_ema.state_dict()}, file_path/(file_name+".pt"))
    def load(self, file_name):
        checkpoint = torch.load(Path(__file__).parents[1]/"results/diffusion"/(file_name+".pt"), map_location=self.device)
        self.F.load_state_dict(checkpoint['model'])
        self.F_ema.load_state_dict(checkpoint['model_ema'])