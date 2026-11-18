import os
import glob
import csv
import argparse
import torch
from rsl_rl.runners import OnPolicyRunner
from hexa_env import HexaEnv
from cfg import tc
from expcfg import mkcfg

p=argparse.ArgumentParser()
p.add_argument("runs",nargs="+")
p.add_argument("-n","--envs",type=int,default=256)
p.add_argument("--steps",type=int,default=1250)
p.add_argument("--seed",type=int,default=1234)
p.add_argument("--final",type=int,default=799,help="skip run dirs whose last checkpoint is below this iteration (0 = evaluate anything)")
p.add_argument("-o","--out",type=str,default=None)
p.add_argument("--hard",action="store_true",help="run only the harder combined stress conditions (writes results/eval_hard.csv)")
a=p.parse_args()
if a.out is None:
    a.out="results/eval_hard.csv" if a.hard else "results/eval.csv"

# every env uses the FULL randomization config unless overridden here, so all runs
# (including the ablations) are tested under identical conditions
TER={
 "flat":["tp=(1,0,0,0)"],
 "pebbles":["tp=(0,1,0,0)"],
 "rocks":["tp=(0,0,1,0)"],
 "slopes":["tp=(0,0,0,1)"],
 "mixed":[],
 "mixed_new":["ts=1"],
 # stress conditions, harder than anything in training
 "steep":["tp=(0,0,0,1)","hila=(15.0,20.0)"],
 "push_hi":["pv=0.8"],
 "push_vhi":["pv=1.5"],
 "noise2x":["nav=0.2","ngp=0.06","ndq=0.02","ndv=1.0"],
 "noise3x":["nav=0.3","ngp=0.09","ndq=0.03","ndv=1.5"],
 "delay_all":["pdl=1.0"],
 # harder conditions (--hard): beyond anything in the standard benchmark
 "push3":["pv=3.0"],
 "noise5x":["nav=0.5","ngp=0.15","ndq=0.05","ndv=2.5"],
 "noise10x":["nav=1.0","ngp=0.3","ndq=0.1","ndv=5.0"],
 "lowfric_push":["pv=1.5"],
 "lowfric_noise":["nav=0.3","ngp=0.09","ndq=0.03","ndv=1.5"],
 "combo":["pv=1.5","nav=0.3","ngp=0.09","ndq=0.03","ndv=1.5"],
}
BASE=["flat","pebbles","rocks","slopes","mixed","mixed_new"]
STRESS=["steep","push_hi","push_vhi","noise2x","noise3x","delay_all"]
# first 10 rows are the standard benchmark, the rest is stress
GRID=[(k,None) for k in BASE]+[("mixed",f) for f in (0.3,0.6,1.0,1.6)]+[("mixed",f) for f in (0.1,0.2)]+[(k,None) for k in STRESS]
if a.hard:
    GRID=[("push3",None),("noise5x",None),("noise10x",None),("lowfric_push",0.1),("lowfric_noise",0.1),("combo",0.1)]
TILT=-0.8660254  # gravity-z above this means body tilt > 30 deg (upright is -1)

def num(s):
    return int(os.path.basename(s)[6:-3])

cks=[]
for r in a.runs:
    if os.path.isdir(r):
        f=sorted(glob.glob(r+"/model_*.pt"),key=num)
        if not f or num(f[-1])<a.final:
            print("skip (no checkpoint >= %d):"%a.final,r)
            continue
        cks.append((os.path.basename(os.path.normpath(r)),f[-1]))
    else:
        cks.append((os.path.basename(os.path.dirname(os.path.abspath(r))),r))
if not cks:
    raise SystemExit("nothing to evaluate")

need={k for k,_ in GRID}|{"mixed"}  # "mixed" always built: supplies the train friction range and the runner
envs={k:HexaEnv(ne=a.envs,c=mkcfg(TER[k])) for k in need}
lo,hi=envs["mixed"].c.fmin,envs["mixed"].c.fmax
runner=OnPolicyRunner(envs["mixed"],tc,log_dir=None,device="cuda:0")

@torch.inference_mode()
def run(env,pol,n):
    torch.manual_seed(a.seed)  # same spawns/commands/noise for every policy -> paired comparison
    mx=env.max_episode_length
    ob=env.reset()
    lin=0.0
    ang=0.0
    tilt=0.0
    falls=0
    exits=0
    for _ in range(n):
        e0=env.eps.clone()
        ob,r,dn,ex=env.step(pol(ob))
        to=ex["time_outs"]
        lg=ex["log"]
        lin+=float(lg["/rew_lin"])
        ang+=float(lg["/rew_ang"])
        tilt+=float((env.gp[:,2]>TILT).float().mean())
        falls+=int((dn.bool()&~to).sum())
        exits+=int((to&(e0<mx)).sum())
    k=mx/(n*env.num_envs)
    c=env.c
    return lin/n/(c.r_lin*env.dt),ang/n/(c.r_ang*env.dt),tilt/n,falls*k,exits*k

os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True)
with open(a.out,"w",newline="") as fh:  # overwrite: no duplicate rows on re-runs
    w=csv.writer(fh)
    w.writerow(["run","ckpt","terrain","friction","lin_frac","falls_per_ep","exits_per_ep","ang_frac","tilt30_frac"])
    for name,ck in cks:
        runner.load(ck)
        pol=runner.get_inference_policy(device="cuda:0")
        for k,f in GRID:
            e=envs[k]
            e.c.fmin,e.c.fmax=(lo,hi) if f is None else (f,f)
            lf,af,tf,fl,ex=run(e,pol,a.steps)
            w.writerow([name,os.path.basename(ck),k,"train" if f is None else f,round(lf,4),round(fl,4),round(ex,4),round(af,4),round(tf,4)])
            fh.flush()
            print(name,k,f,"lin",round(lf,3),"ang",round(af,3),"tilt30",round(tf,3),"falls/ep",round(fl,3),"exits/ep",round(ex,3))
