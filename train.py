import argparse
import os
import json
import random
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from hexa_env import HexaEnv
from cfg import tc
from expcfg import mkcfg

p=argparse.ArgumentParser()
p.add_argument("-n","--name",type=str,default="full")
p.add_argument("-s","--seed",type=int,default=0)
p.add_argument("-e","--envs",type=int,default=2048)
p.add_argument("-i","--iters",type=int,default=800)
p.add_argument("-r","--resume",type=str,default=None)
p.add_argument("--set",nargs="*",default=[])
a=p.parse_args()

random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
c=mkcfg(a.set)
ld=os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs","exp",a.name+"_s"+str(a.seed))
os.makedirs(ld,exist_ok=True)
cd={k:getattr(c,k) for k in dir(c) if not k.startswith("_")}
json.dump({"name":a.name,"seed":a.seed,"iters":a.iters,"envs":a.envs,"overrides":a.set,"env_cfg":cd,"train_cfg":tc},open(os.path.join(ld,"cfg_used.json"),"w"),indent=1,default=str)
env=HexaEnv(ne=a.envs,c=c)
runner=OnPolicyRunner(env,tc,log_dir=ld,device="cuda:0")
if a.resume:
    runner.load(a.resume)
runner.learn(num_learning_iterations=a.iters)
