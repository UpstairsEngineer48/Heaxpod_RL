import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument("csv",nargs="?",default="results/eval.csv")
p.add_argument("-o","--out",default="results")
a=p.parse_args()
os.makedirs(a.out,exist_ok=True)

ORDER=["full","flat","nofric","nopush","nodelay","nonoise"]
BASE=["flat","pebbles","rocks","slopes","mixed","mixed_new"]
STRESS=["steep","push_hi","push_vhi","noise2x","noise3x","delay_all"]

df=pd.read_csv(a.csv)
df["cfg"]=df["run"].str.rsplit("_s",n=1).str[0]
df["seed"]=df["run"].str.rsplit("_s",n=1).str[1].astype(int)
df["friction"]=df["friction"].astype(str)
cfgs=[c for c in ORDER if c in set(df.cfg)]+sorted(set(df.cfg)-set(ORDER))
ns=df.groupby("cfg").seed.nunique()
print("seeds per config:",ns.to_dict())

def agg(d,metric):
    # seed is the unit: one value per run, then mean/std (ddof=1) across seeds
    g=d.groupby(["cfg","terrain"])[metric]
    return g.mean().unstack(),g.std(ddof=1).unstack()

def table(terrains,metric,fname,title,fmt="{:.3f}"):
    d=df[(df.friction=="train")&df.terrain.isin(terrains)]
    m,s=agg(d,metric)
    m=m.reindex(index=cfgs,columns=terrains)
    s=s.reindex(index=cfgs,columns=terrains)
    t=m.copy().astype(object)
    for i in m.index:
        for j in m.columns:
            t.loc[i,j]=(fmt.format(m.loc[i,j])+" ± "+fmt.format(0 if np.isnan(s.loc[i,j]) else s.loc[i,j]))
    t.index.name="config"
    with open(os.path.join(a.out,fname+".md"),"w") as f:
        f.write("**"+title+"**\n\n"+t.to_markdown()+"\n")
    with open(os.path.join(a.out,fname+".tex"),"w") as f:
        f.write(t.to_latex(escape=False,caption=title,label="tab:"+fname).replace("±","$\\pm$"))
    t.to_csv(os.path.join(a.out,fname+".csv"))
    print("\n"+title);print(t.to_string())

table(BASE,"lin_frac","tab_lin_base","Linear-velocity tracking score (mean ± std over seeds), standard benchmark")
table(BASE,"falls_per_ep","tab_falls_base","Falls per 20 s episode, standard benchmark")
table(STRESS,"lin_frac","tab_lin_stress","Linear-velocity tracking score under stress conditions")
table(STRESS,"falls_per_ep","tab_falls_stress","Falls per episode under stress conditions")
table(BASE,"ang_frac","tab_ang_base","Yaw-rate tracking score, standard benchmark")

# friction sweep (mixed terrain, fixed friction)
fr=df[(df.terrain=="mixed")&(df.friction!="train")].copy()
fr["friction"]=fr.friction.astype(float)
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
for k,(metric,yl) in enumerate([("lin_frac","lin tracking score"),("falls_per_ep","falls / episode")]):
    for c in cfgs:
        d=fr[fr.cfg==c].groupby("friction")[metric]
        m=d.mean();s=d.std(ddof=1).fillna(0)
        if len(m)==0: continue
        ax[k].plot(m.index,m.values,marker="o",label=c)
        ax[k].fill_between(m.index,m-s,m+s,alpha=0.15)
    ax[k].set_xlabel("sliding friction coefficient");ax[k].set_ylabel(yl);ax[k].grid(alpha=.3)
ax[0].legend(fontsize=8)
plt.tight_layout();plt.savefig(os.path.join(a.out,"fig_friction.png"),dpi=200);plt.close()

# stress bar chart
fig,ax=plt.subplots(1,2,figsize=(11,3.6))
for k,(metric,yl) in enumerate([("lin_frac","lin tracking score"),("falls_per_ep","falls / episode")]):
    d=df[(df.friction=="train")&df.terrain.isin(STRESS)]
    m,s=agg(d,metric)
    m=m.reindex(index=cfgs,columns=STRESS);s=s.reindex(index=cfgs,columns=STRESS).fillna(0)
    w=0.8/len(cfgs);x=np.arange(len(STRESS))
    for i,c in enumerate(cfgs):
        ax[k].bar(x+i*w,m.loc[c],w,yerr=s.loc[c],label=c,capsize=2)
    ax[k].set_xticks(x+0.4-w/2);ax[k].set_xticklabels(STRESS,rotation=30,ha="right");ax[k].set_ylabel(yl)
ax[0].legend(fontsize=8,ncol=2)
plt.tight_layout();plt.savefig(os.path.join(a.out,"fig_stress.png"),dpi=200);plt.close()

# ablation summary: delta vs full, in seed-std units, on pooled stress+base rows
d=df[df.friction=="train"]
per=d.groupby(["cfg","seed"])[["lin_frac","falls_per_ep","tilt30_frac"]].mean()
summ=per.groupby("cfg").agg(["mean","std"]).reindex(cfgs)
summ.to_csv(os.path.join(a.out,"summary_all_conditions.csv"))
print("\nPooled over all 12 train-friction conditions:\n",summ.round(4).to_string())
print("\nwrote tables + figures to",a.out)
