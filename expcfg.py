import ast
from hexa_env import C

def mkcfg(sets=None):
    ov={}
    for s in sets or []:
        k,v=s.split("=",1)
        assert hasattr(C,k),"unknown config key "+k
        ov[k]=ast.literal_eval(v)
    return type("Cfg",(C,),ov)
