import argparse
import glob
import os
import select
import sys
import termios
import time
import tty
import torch
import mujoco as mj
import mujoco.viewer as mv
from rsl_rl.runners import OnPolicyRunner
from hexa_env import HexaEnv,C
from cfg import tc

p=argparse.ArgumentParser()
p.add_argument("-c","--ckpt",type=str,default=None)
a=p.parse_args()

cks=glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs","hexa","*","model_*.pt"))
if not a.ckpt and not cks:
    raise SystemExit("No checkpoint found in logs/hexa/. Train first with: python3 train.py (or pass -c path/to/model.pt)")
ck=a.ckpt or sorted(cks,key=os.path.getmtime)[-1]
print("checkpoint:",ck)
env=HexaEnv(ne=1)
runner=OnPolicyRunner(env,tc,log_dir=None,device="cuda:0")
runner.load(ck)
pol=runner.get_inference_policy(device="cuda:0")

# clean demo: no noise, no random pushes, no 20s timeout
env.nz=False
env.pn=10**9
env.max_episode_length=10**9

mm=mj.MjModel.from_xml_path(C.xml)
mm.hfield_data[:]=env.hm.cpu().numpy().reshape(-1)
dd=mj.MjData(mm)

VX=0.3
VY=0.3
WZ=0.5
FIRST=0.6   # hold window on a fresh press (bridges the OS key-repeat delay)
REPEAT=0.12 # hold window while key-repeat events keep arriving

# arrow escape sequences -> (axis,value)
ARROWS=[
    (b"\x1b[A",(0,VX)),(b"\x1bOA",(0,VX)),
    (b"\x1b[B",(0,-VX)),(b"\x1bOB",(0,-VX)),
    (b"\x1b[D",(2,WZ)),(b"\x1bOD",(2,WZ)),
    (b"\x1b[C",(2,-WZ)),(b"\x1bOC",(2,-WZ)),
    (b"w",(1,VY)),(b"W",(1,VY)),
    (b"s",(1,-VY)),(b"S",(1,-VY)),
]
HELP="""
  UP/DOWN     forward/back
  LEFT/RIGHT  turn left/right
  W/S         strafe left/right
  (release = stop)     R reset     Q quit
"""

fd=sys.stdin.fileno()
old=termios.tcgetattr(fd)
act_until={}  # (axis,value) -> expiry time
last=0.0

try:
    tty.setcbreak(fd)
    print(HELP)
    env.reset()
    with mv.launch_passive(mm,dd) as v:
        v.cam.type=mj.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid=1
        v.cam.distance=1.6
        v.cam.elevation=-25
        v.cam.azimuth=135
        quit_=False
        with torch.inference_mode():
            while v.is_running() and not quit_:
                t0=time.time()
                while select.select([fd],[],[],0)[0]:
                    d=os.read(fd,64)
                    for sq,k in ARROWS:
                        if sq in d:
                            fresh=act_until.get(k,0.0)<t0
                            act_until[k]=t0+(FIRST if fresh else REPEAT)
                    low=d.lower()
                    if b"q" in low or b"\x03" in low:
                        quit_=True
                    if b"r" in low:
                        env.reset()
                cmd=[0.0,0.0,0.0]
                for (ax,val),ex in act_until.items():
                    if ex>=t0:
                        cmd[ax]=val
                env.cmd[:]=torch.tensor(cmd,device=env.dev)
                act=pol(env.get_observations())
                env.step(act)
                dd.qpos[:]=env.q[0].cpu().numpy()
                mj.mj_forward(mm,dd)
                v.sync()
                sys.stdout.write(f"\rvx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}   ")
                sys.stdout.flush()
                time.sleep(max(0.0,env.dt-(time.time()-t0)))
finally:
    termios.tcsetattr(fd,termios.TCSADRAIN,old)
    print()
