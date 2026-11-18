import argparse
import glob
import os
import time
import torch
import mujoco as mj
import mujoco.viewer as mv
from rsl_rl.runners import OnPolicyRunner
from hexa_env import HexaEnv,C
from cfg import tc

p=argparse.ArgumentParser()
p.add_argument("-c","--ckpt",type=str,default=None)
p.add_argument("--view",action="store_true")
p.add_argument("--cmd",type=float,nargs=3,default=[0.3,0.0,0.0])
a=p.parse_args()

cks=glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs","hexa","*","model_*.pt"))
if not a.ckpt and not cks:
    raise SystemExit("No checkpoint found in logs/hexa/. Train first with: python3 train.py (or pass -c path/to/model.pt)")
ck=a.ckpt or sorted(cks,key=os.path.getmtime)[-1]
print("checkpoint:",ck)
env=HexaEnv(ne=1 if a.view else 256)
runner=OnPolicyRunner(env,tc,log_dir=None,device="cuda:0")
runner.load(ck)
pol=runner.get_inference_policy(device="cuda:0")

@torch.inference_mode()
def run(cmd,n=320,w=64):
    env.reset()
    c=torch.tensor(cmd,device=env.dev)
    sl=torch.zeros(3,device=env.dev)
    nf=0
    for i in range(n):
        env.cmd[:]=c
        ob=env.get_observations()
        act=pol(ob)
        ob,r,dn,ex=env.step(act)
        if i>=w:
            sl+=torch.stack((env.lv[:,0].mean(),env.lv[:,1].mean(),env.av[:,2].mean()))
            nf+=int((dn.bool()&~ex["time_outs"]).sum())
    sl=sl/(n-w)
    print(f"cmd {cmd} -> vx {sl[0]:.2f} vy {sl[1]:.2f} wz {sl[2]:.2f} falls {nf}")

if not a.view:
    for cmd in ([0.0,0.0,0.0],[0.3,0.0,0.0],[-0.3,0.0,0.0],[0.0,0.3,0.0],[0.0,0.0,0.5],[0.0,0.0,-0.5]):
        run(cmd)
else:
    mm=mj.MjModel.from_xml_path(C.xml)
    mm.hfield_data[:]=env.hm.cpu().numpy().reshape(-1)
    dd=mj.MjData(mm)
    c=torch.tensor(a.cmd,device=env.dev)
    env.reset()
    with mv.launch_passive(mm,dd) as v:
        with torch.inference_mode():
            while v.is_running():
                env.cmd[:]=c
                act=pol(env.get_observations())
                env.step(act)
                dd.qpos[:]=env.q[0].cpu().numpy()
                mj.mj_forward(mm,dd)
                v.sync()
                time.sleep(env.dt)