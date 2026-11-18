import sys
import re
import json
import numpy as np
import torch
import mujoco as mj
from rsl_rl.runners import OnPolicyRunner
from hexa_env import HexaEnv,C
from cfg import tc

ck=sys.argv[1]
base=sys.argv[2] if len(sys.argv)>2 else "hexa_policy"
d=torch.load(ck,map_location="cpu",weights_only=False)
print("checkpoint keys:",list(d.keys()))
ak=[k for k in d if "actor" in k and "state" in k]
assert len(ak)==1,"cannot find actor weights, send me the key list above"
sd=d[ak[0]]
print("actor tensors:",{k:tuple(v.shape) for k,v in sd.items()})
assert not any("normalizer" in k for k in sd),"actor has an obs normalizer, send me the tensor list above"

pat=re.compile(r"mlp\.(\d+)\.(weight|bias)$")
ws={}
for k,v in sd.items():
    m=pat.search(k)
    if m:
        ws[(int(m.group(1)),m.group(2))]=v.cpu().numpy().astype(np.float32)
idx=sorted({i for i,_ in ws})
W=[ws[(i,"weight")] for i in idx]
B=[ws[(i,"bias")] for i in idx]
assert len(W)==4 and W[0].shape[1]==63 and W[-1].shape[0]==18,"unexpected network shape"

def fwd(x):
    for i in range(len(W)):
        x=x@W[i].T+B[i]
        if i<len(W)-1:
            x=np.where(x>0,x,np.expm1(np.minimum(x,0)))
    return x

mm=mj.MjModel.from_xml_path(C.xml)
names=[]
for i in range(mm.nu):
    jid=int(mm.actuator_trnid[i,0])
    assert int(mm.jnt_qposadr[jid])==7+i and int(mm.jnt_dofadr[jid])==6+i,"joint order mismatch"
    names.append(mj.mj_id2name(mm,mj.mjtObj.mjOBJ_JOINT,jid))
cfg={
    "joint_names":names,
    "dp":[float(x) for x in mm.key_qpos[0][7:]],
    "ctrl_lo":[float(x) for x in mm.actuator_ctrlrange[:,0]],
    "ctrl_hi":[float(x) for x in mm.actuator_ctrlrange[:,1]],
    "dt":float(mm.opt.timestep*C.dec),
    "slv":C.slv,"sav":C.sav,"sdq":C.sdq,"sdv":C.sdv,"clo":C.clo,
    "asc":C.asc,"acl":C.acl,
    "vx":C.vx,"vy":C.vy,"wz":C.wz,
}
arr={}
for i in range(len(W)):
    arr["W"+str(i)]=W[i]
    arr["B"+str(i)]=B[i]
np.savez(base+".npz",**arr)
json.dump(cfg,open(base+".json","w"),indent=1)
print("saved",base+".npz and",base+".json")
print("joint order:",names)

with torch.inference_mode():
    env=HexaEnv(ne=64)
    runner=OnPolicyRunner(env,tc,log_dir=None,device="cuda:0")
    runner.load(ck)
    pol=runner.get_inference_policy(device="cuda:0")
    ob=env.reset()
    err=0.0
    for i in range(100):
        act=pol(ob)
        ref=act.cpu().numpy()
        mine=fwd(ob["policy"].cpu().numpy())
        err=max(err,float(np.abs(ref-mine).max()))
        ob,r,dn,ex=env.step(act)
print("max abs diff vs rsl_rl policy:",err)
print("OK" if err<1e-3 else "MISMATCH, do not deploy")
