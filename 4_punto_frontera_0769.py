import numpy as np, sys
sys.path.insert(0,'.')
import esquema as E, experimento as X, decisivo as D, experimentos_finales as F, fis, metricas as M
S=E.SENALES_FUSION
df=X._offendes(); tr,va,te=[df[df.split==s] for s in ("train","validation","test")]
pts=[]
for sem in F.SEMILLAS: pts+=F.puntos_fis(F.frente_fis(tr,va,sem),va,te)
for b in [0.10,0.20]:
    p=F.en_presupuesto(pts,b); m,c=D.construye(p["x"],fis.reglamento_v2())
    niv=M.a_nivel(m.riesgo(te[S].to_numpy(float)),c)
    print(b,"cuts",np.round(c,1),"levels",np.round(np.bincount(niv,minlength=4)/len(niv),3),{k:round(v,3) for k,v in p["test"].items()})
