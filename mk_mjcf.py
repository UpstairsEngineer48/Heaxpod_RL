import mujoco as mj
import numpy as np
import re,sys,os
src=sys.argv[1]
dst=sys.argv[2]
md=os.path.abspath(sys.argv[3])
kp=float(sys.argv[4]) if len(sys.argv)>4 else 20.0
t=open(src).read()
t=re.sub(r'package://[^"]*/','',t)
t=re.sub(r'<material[^>]*/>','',t)
t=re.sub(r'<material[^>]*>.*?</material>','',t,flags=re.S)
t=re.sub(r'<joint name="base_\w+?_to_\w+"[^>]*>.*?</joint>','',t,flags=re.S)
t=re.sub(r'<link name="base_(footprint|link)"\s*(/>|>\s*</link>)','',t)
hd='<mujoco><compiler meshdir="'+md+'/" balanceinertia="true" inertiafromgeom="false" discardvisual="false"/></mujoco>'
t=re.sub(r'(<robot[^>]*>)',lambda m:m.group(1)+hd,t,count=1)
sp=mj.MjSpec.from_string(t)
sp.option.timestep=0.004
rt=sp.worldbody.first_body()
rt.add_joint(name="root",type=mj.mjtJoint.mjJNT_FREE)
rt.add_site(name="imu",pos=[0,0,0.0186],size=[0.01,0,0])
for i in range(1,7):
    b=sp.body("Tibia_"+str(i))
    b.add_site(name="foot"+str(i),pos=[-0.0716,-0.0404,0.1549],size=[0.012,0,0])
cols={"Base":[.15,.15,.15,1],"Coxa":[.1,.35,.75,1],"Femur":[.85,.45,.05,1],"Tibia":[.05,.05,.05,1]}
sb=False
for g in sp.geoms:
    ms=g.type==mj.mjtGeom.mjGEOM_MESH
    bs=ms and g.meshname=="Base"
    k=g.type==mj.mjtGeom.mjGEOM_SPHERE or (bs and not sb)
    sb=sb or bs
    g.contype=int(k)
    g.conaffinity=0
    if ms:
        g.rgba=cols[g.meshname.split("_")[0]]
sp.worldbody.add_geom(name="floor",type=mj.mjtGeom.mjGEOM_PLANE,size=[0,0,0.05],contype=0,conaffinity=1)
for j in sp.joints:
    if j.type!=mj.mjtJoint.mjJNT_HINGE:
        continue
    j.armature=0.01
    j.damping=[0.5,0,0]
    a=sp.add_actuator(name=j.name,target=j.name,trntype=mj.mjtTrn.mjTRN_JOINT)
    a.gainprm=[kp]+[0]*9
    a.biasprm=[0,-kp]+[0]*8
    a.biastype=mj.mjtBias.mjBIAS_AFFINE
    a.ctrlrange=list(j.range)
    a.ctrllimited=mj.mjtLimited.mjLIMITED_TRUE
    a.forcerange=[-5,5]
    a.forcelimited=mj.mjtLimited.mjLIMITED_TRUE
sp.add_sensor(name="gyro",type=mj.mjtSensor.mjSENS_GYRO,objtype=mj.mjtObj.mjOBJ_SITE,objname="imu")
sp.add_sensor(name="acc",type=mj.mjtSensor.mjSENS_ACCELEROMETER,objtype=mj.mjtObj.mjOBJ_SITE,objname="imu")
sp.add_sensor(name="quat",type=mj.mjtSensor.mjSENS_FRAMEQUAT,objtype=mj.mjtObj.mjOBJ_SITE,objname="imu")
m=sp.compile()
d=mj.MjData(m)
mj.mj_forward(m,d)
ids=[i for i in range(m.ngeom) if m.geom_type[i]==mj.mjtGeom.mjGEOM_SPHERE]
print("foot spheres",len(ids),"max foot reach xy",round(float(max(np.hypot(d.geom_xpos[i,0],d.geom_xpos[i,1]) for i in ids)),3))
z=min(d.geom_xpos[i,2]-m.geom_size[i,0] for i in ids)
q=m.qpos0.copy()
q[2]=-z+0.005
sp.add_key(name="home",qpos=q.tolist(),ctrl=[0.0]*m.nu)
m=sp.compile()
d=mj.MjData(m)
mj.mj_resetDataKeyframe(m,d,0)
z0=float(d.qpos[2])
for _ in range(500):
    mj.mj_step(m,d)
print("nq",m.nq,"nv",m.nv,"nu",m.nu,"mass",round(float(m.body_subtreemass[1]),3),"z start",round(z0,3),"z after 2s",round(float(d.qpos[2]),3))
rel=os.path.relpath(md,os.path.dirname(os.path.abspath(dst)))
open(dst,"w").write(sp.to_xml().replace(md,rel))
print("saved",dst)