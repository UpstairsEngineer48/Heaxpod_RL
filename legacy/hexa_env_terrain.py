import os
import math
import numpy as np
import mujoco as mj
import mujoco_warp as mjw
import warp as wp
import torch
from tensordict import TensorDict

class C:
    xml=os.path.join(os.path.dirname(os.path.abspath(__file__)),"hexapod_terrain.xml")
    dec=4
    ep=20.0
    asc=0.2
    acl=1.0
    rs=10.0
    vx=0.4
    vy=0.4
    wz=0.8
    slv=2.0
    sav=0.25
    sdq=1.0
    sdv=0.05
    clo=100.0
    jn=0.05
    zmin=0.06
    gmin=-0.5
    slin=0.05
    sang=0.008
    ncon=40
    njmax=256
    r_term=-200.0
    r_lin=8.0
    r_ang=6.0
    r_acc=-2.5e-5
    r_rate=-0.02
    r_def=-0.01
    r_ori=-5.0
    nav=0.1
    ngp=0.03
    ndq=0.01
    ndv=0.5
    pdl=0.5
    ps=4.0
    pv=0.3
    fmin=0.4
    fmax=1.25
    L=12.0
    hn=481
    pm=3.0
    tp=(0.3,0.25,0.25,0.2)
    pebr=(0.005,0.015)
    rokr=(0.015,0.035)
    hila=(3.0,12.0)

def blocks(rng,k1,r,amp):
    m=-(-k1//r)
    a=rng.random((m,m))*amp
    return np.kron(a,np.ones((r,r)))[:k1,:k1]

def mkter(c):
    rng=np.random.default_rng(0)
    n=c.hn
    res=c.L/(n-1)
    k=int(round(c.pm/res))
    g=int(round(c.L/c.pm))
    k1=k+1
    assert k*g+1==n
    h=np.zeros((n,n),dtype=np.float32)
    for i in range(g):
        for j in range(g):
            t=rng.choice(4,p=c.tp)
            if t==0:
                continue
            if t==1:
                p=blocks(rng,k1,2,rng.uniform(*c.pebr))
            elif t==2:
                p=blocks(rng,k1,6,rng.uniform(*c.rokr))
            else:
                a=np.deg2rad(rng.uniform(*c.hila))
                u=np.linspace(-1,1,k1)
                d=1-np.maximum(np.abs(u)[None,:],np.abs(u)[:,None])
                p=d*(c.pm/2)*np.tan(a)
            s=h[i*k:i*k+k1,j*k:j*k+k1]
            h[i*k:i*k+k1,j*k:j*k+k1]=np.maximum(s,p)
    return h

def qinv(q,v):
    w=q[:,0:1]
    u=-q[:,1:4]
    t=2*torch.cross(u,v,dim=1)
    return v+w*t+torch.cross(u,t,dim=1)

class HexaEnv:
    def __init__(self,ne=2048,dev="cuda:0",c=C):
        self.c=c
        self.dev=dev
        self.device=dev
        self.num_envs=ne
        self.num_actions=18
        self.cfg={}
        self.nz=True
        wp.set_device(dev)
        mm=mj.MjModel.from_xml_path(c.xml)
        assert mm.nu==18 and mm.nkey>=1
        assert mm.nhfield==1 and mm.hfield_nrow[0]==c.hn
        h=mkter(c)
        mm.hfield_data[:]=h.reshape(-1)
        self.hm=torch.tensor(h,device=dev)
        self.hp=torch.nn.functional.max_pool2d(self.hm[None,None],25,stride=1,padding=12)[0,0]
        self.rl=c.L/2
        self.dt=mm.opt.timestep*c.dec
        self.max_episode_length=math.ceil(c.ep/self.dt)
        self.pn=int(c.ps/self.dt)
        self.m=mjw.put_model(mm)
        print("geom_friction shape:",self.m.geom_friction.shape)
        fr=np.repeat(self.m.geom_friction.numpy()[:1],ne,axis=0)
        self.m.geom_friction=wp.array(fr,dtype=wp.vec3,device=dev)
        self.gf=wp.to_torch(self.m.geom_friction)
        self.d=mjw.make_data(mm,nworld=ne,nconmax=c.ncon,njmax=c.njmax)
        self.q=wp.to_torch(self.d.qpos)
        self.qv=wp.to_torch(self.d.qvel)
        self.ct=wp.to_torch(self.d.ctrl)
        self.wm=wp.to_torch(self.d.qacc_warmstart)
        self.hq=torch.tensor(mm.key_qpos[0],dtype=torch.float32,device=dev)
        self.dp=self.hq[7:].clone()
        self.g0=torch.tensor([0.0,0.0,-1.0],device=dev)
        self.cs=torch.tensor([c.slv,c.slv,c.sav],device=dev)
        self.cmd=torch.zeros(ne,3,device=dev)
        self.a=torch.zeros(ne,18,device=dev)
        self.pa=torch.zeros(ne,18,device=dev)
        self.dly=torch.zeros(ne,dtype=torch.bool,device=dev)
        self.eps=torch.zeros(ne,dtype=torch.long,device=dev)
        s=self.dt
        self.sc={"lin":c.r_lin*s,"ang":c.r_ang*s,"acc":c.r_acc*s,"rate":c.r_rate*s,"def":c.r_def*s,"ori":c.r_ori*s,"term":c.r_term*s}
        ids=torch.arange(ne,device=dev)
        self._rst(ids)
        mjw.step(self.m,self.d)
        self.g=None
        try:
            with wp.ScopedCapture() as cp:
                mjw.step(self.m,self.d)
            self.g=cp.graph
        except Exception as e:
            print("graph capture failed, using plain steps:",e)
        self._rst(ids)
        self.eps=torch.randint(0,self.max_episode_length,(ne,),device=dev)
        self._state()

    def _hgt(self,x,y,pool=False):
        m=self.hp if pool else self.hm
        n=self.c.hn
        u=((x+self.rl)/(2*self.rl)*(n-1)).round().long().clamp(0,n-1)
        v=((y+self.rl)/(2*self.rl)*(n-1)).round().long().clamp(0,n-1)
        return m[v,u]

    def _cmd(self,ids):
        n=len(ids)
        c=self.c
        r=torch.rand(n,3,device=self.dev)*2-1
        self.cmd[ids,0]=r[:,0]*c.vx
        self.cmd[ids,1]=r[:,1]*c.vy
        self.cmd[ids,2]=r[:,2]*c.wz
        k=torch.norm(self.cmd[ids,:2],dim=1)>0.02
        self.cmd[ids,:2]*=k[:,None]

    def _rst(self,ids):
        n=len(ids)
        if n==0:
            return
        c=self.c
        self.q[ids]=self.hq
        x=(torch.rand(n,device=self.dev)*2-1)*(self.rl-1.5)
        y=(torch.rand(n,device=self.dev)*2-1)*(self.rl-1.5)
        self.q[ids,0]=x
        self.q[ids,1]=y
        self.q[ids,2]=self.hq[2]+self._hgt(x,y,True)+0.01
        yaw=(torch.rand(n,device=self.dev)*2-1)*math.pi
        self.q[ids,3]=torch.cos(yaw/2)
        self.q[ids,6]=torch.sin(yaw/2)
        self.q[ids,7:]=self.dp+(torch.rand(n,18,device=self.dev)*2-1)*c.jn
        self.qv[ids]=0
        self.wm[ids]=0
        self.ct[ids]=self.dp
        self.a[ids]=0
        self.pa[ids]=0
        self.eps[ids]=0
        self.dly[ids]=torch.rand(n,device=self.dev)<c.pdl
        self.gf[ids,:,0]=(torch.rand(n,device=self.dev)*(c.fmax-c.fmin)+c.fmin)[:,None]
        self._cmd(ids)

    def _state(self):
        n=self.num_envs
        qt=self.q[:,3:7]
        self.lv=qinv(qt,self.qv[:,0:3])
        self.av=self.qv[:,3:6]
        self.gp=qinv(qt,self.g0.expand(n,3))
        self.z=self.q[:,2]
        self.dq=self.q[:,7:]
        self.dv=self.qv[:,6:]

    def _obs(self):
        c=self.c
        n=self.num_envs
        av=self.av
        gp=self.gp
        dq=self.dq
        dv=self.dv
        if self.nz:
            d=self.dev
            av=av+torch.randn(n,3,device=d)*c.nav
            gp=gp+torch.randn(n,3,device=d)*c.ngp
            dq=dq+torch.randn(n,18,device=d)*c.ndq
            dv=dv+torch.randn(n,18,device=d)*c.ndv
        o=torch.cat((av*c.sav,gp,self.cmd*self.cs,(dq-self.dp)*c.sdq,dv*c.sdv,self.a),dim=1)
        o=torch.clamp(o,-c.clo,c.clo)
        return TensorDict({"policy":o},batch_size=[self.num_envs],device=self.dev)

    def _rew(self,pv,tm):
        s=self.sc
        c=self.c
        e1=((self.cmd[:,:2]-self.lv[:,:2])**2).sum(1)
        e2=(self.cmd[:,2]-self.av[:,2])**2
        t={}
        t["lin"]=torch.exp(-e1/c.slin)*s["lin"]
        t["ang"]=torch.exp(-e2/c.sang)*s["ang"]
        t["acc"]=(((self.dv-pv)/self.dt)**2).sum(1)*s["acc"]
        t["rate"]=((self.pa-self.a)**2).sum(1)*s["rate"]
        t["def"]=((self.dq-self.dp)**2).sum(1)*s["def"]
        t["ori"]=(self.gp[:,:2]**2).sum(1)*s["ori"]
        t["term"]=tm.float()*s["term"]
        r=sum(t.values())
        return r,{"/rew_"+k:v.mean() for k,v in t.items()}

    def get_observations(self):
        return self._obs()

    def reset(self):
        self._rst(torch.arange(self.num_envs,device=self.dev))
        self._state()
        return self._obs()

    def step(self,act):
        c=self.c
        pv=self.dv.clone()
        self.pa=self.a
        self.a=torch.clamp(act*c.asc,-c.acl,c.acl)
        ap=torch.where(self.dly[:,None],self.pa,self.a)
        self.ct[:]=self.dp+ap
        pi=torch.nonzero((self.eps>0)&(self.eps%self.pn==0)).flatten()
        if len(pi)>0:
            self.qv[pi,0:2]+=(torch.rand(len(pi),2,device=self.dev)*2-1)*c.pv
        for _ in range(c.dec):
            if self.g is None:
                mjw.step(self.m,self.d)
            else:
                wp.capture_launch(self.g)
        wp.synchronize_device(self.dev)
        self.eps+=1
        self._state()
        ri=torch.nonzero(self.eps%int(c.rs/self.dt)==0).flatten()
        self._cmd(ri)
        x=self.q[:,0]
        y=self.q[:,1]
        out=(x.abs()>self.rl-0.5)|(y.abs()>self.rl-0.5)
        to=(self.eps>self.max_episode_length)|out
        dn=to|(self.gp[:,2]>c.gmin)|(self.z-self._hgt(x,y)<c.zmin)
        rw,lg=self._rew(pv,dn&~to)
        self._rst(torch.nonzero(dn).flatten())
        self._state()
        ex={"time_outs":to,"log":lg}
        return self._obs(),rw,dn.to(torch.long),ex
