import numpy as np, pandas as pd, json, time, sys
sys.path.insert(0,'.')
import agregadores as A, esquema as E, experimento as X, fis, metricas as M, decisivo as D
S=E.SENALES_FUSION
df=X._offendes(); tr,va,te=[df[df.split==s] for s in ("train","validation","test")]
print("sizes",len(tr),len(va),len(te), "harm test",int(te.peligroso.sum()),"NOE test",int(te.expletivo_benigno.sum()))
# LR coefs
lr=A.Supervisado(S).ajusta(tr); print("LR coefs (std):",dict(zip(S,lr.mod.coef_[0].round(3))))
wm=A.MediaPonderada(S).ajusta(tr); print("WM weights:",dict(zip(S,wm.pesos.round(3))))
res={}
for ag in [A.MaximoActual(),A.MediaPonderada(S),A.Supervisado(S)]:
    ag.ajusta(tr); c=D.elige_cortes(ag.riesgo_df(va),va); niv=D.nivel_continuo(ag.riesgo_df(te),c)
    res[ag.nombre]=niv
t=time.time()
fis_runs={}
for sem in D.SEMILLAS:
    m,c=D.ajusta_fis(va,sem); r=m.riesgo(te[S].to_numpy(float)); niv=M.a_nivel(r,c)
    fis_runs[sem]=(niv,c,float(r.max()),m)
    print("seed",sem,"cuts",np.round(c,1),"max r test",round(float(r.max()),2),D.mide(niv,te)["proteccion"].__round__(3), round(time.time()-t))
    res[f"FIS{sem}"]=niv
lev=pd.DataFrame({k:np.bincount(v,minlength=4)/len(v) for k,v in res.items()},index=["L0","L1","L2","L3"]).T
print(lev.round(3))
json.dump({k:v.tolist() for k,v in res.items()},open('levels.json','w'))
