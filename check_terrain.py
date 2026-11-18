import torch
from hexa_env import HexaEnv
e=HexaEnv(ne=256)
a=torch.zeros(256,18,device=e.dev)
with torch.inference_mode():
    for _ in range(60):
        e.step(a)
    h=e.q[:,2]-e._hgt(e.q[:,0],e.q[:,1])
print("terrain max",float(e.hm.max()),"min",float(e.hm.min()))
print("height above terrain mean",float(h.mean()),"std",float(h.std()))
