import re,sys
src=sys.argv[1]
dst=sys.argv[2]
L=float(sys.argv[3]) if len(sys.argv)>3 else 12.0
n=int(sys.argv[4]) if len(sys.argv)>4 else 481
t=open(src).read()
t,k=re.subn(r'<geom[^>]*name="floor"[^>]*/>','<geom name="floor" type="hfield" hfield="terrain" contype="0" conaffinity="1"/>',t)
assert k==1,"floor geom not found"
t,k=re.subn(r'<asset>','<asset><hfield name="terrain" nrow="'+str(n)+'" ncol="'+str(n)+'" size="'+str(L/2)+' '+str(L/2)+' 1 0.1"/>',t,count=1)
assert k==1,"asset block not found"
open(dst,"w").write(t)
print("saved",dst)
