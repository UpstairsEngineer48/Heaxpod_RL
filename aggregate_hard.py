import argparse
import os
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument("csv",nargs="?",default="results/eval_hard.csv")
p.add_argument("-o","--out",default="results")
a=p.parse_args()
ORDER=["full","flat","nofric","nopush","nodelay","nonoise"]
HARD=["push3","noise5x","noise10x","lowfric_push","lowfric_noise","combo"]
df=pd.read_csv(a.csv)
df["cfg"]=df.run.str.rsplit("_s",n=1).str[0]
cfgs=[c for c in ORDER if c in set(df.cfg)]
print("seeds per config:",df.groupby("cfg").run.nunique().to_dict())
os.makedirs(a.out,exist_ok=True)
def tab(metric,name,title,fmt="{:.3f}"):
    g=df.groupby(["cfg","terrain"])[metric]
    m=g.mean().unstack().reindex(index=cfgs,columns=HARD)
    s=g.std(ddof=1).unstack().reindex(index=cfgs,columns=HARD).fillna(0)
    t=m.copy().astype(object)
    for i in m.index:
        for j in m.columns:
            t.loc[i,j]=fmt.format(m.loc[i,j])+" ± "+fmt.format(s.loc[i,j])
    t.index.name="config"
    with open(os.path.join(a.out,name+".md"),"w") as f:
        f.write("**"+title+"**\n\n"+t.to_markdown()+"\n")
    t.to_csv(os.path.join(a.out,name+".csv"))
    print("\n"+title);print(t.to_string())
    return m
m=tab("lin_frac","tab_lin_hard","Linear tracking score, harder combined stress")
tab("falls_per_ep","tab_falls_hard","Falls per episode, harder combined stress")
print("\nDelta vs full (lin_frac):");print((m-m.loc["full"]).round(3).to_string())
