import os, pickle
import numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
OUTDIR='out'; os.makedirs(OUTDIR, exist_ok=True)
DEFAULT=['AAPL','MSFT','NVDA','GOOGL','AMZN','META','AVGO','TSLA','LLY','JPM','V','XOM','UNH','MA','HD','PG','COST','ABBV','ORCL','JNJ']
def rsi(s,w=14):
    d=s.diff(); up=(d.clip(lower=0)).ewm(alpha=1/w,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/w,adjust=False).mean(); rs=up/(dn+1e-9); return 100-(100/(1+rs))
def bbpos(c,w=20,k=2.0):
    ma=c.rolling(w).mean(); sd=c.rolling(w).std(); up=ma+k*sd; lo=ma-k*sd; sp=(up-lo); return ((c-lo)/sp.replace(0,np.nan)).clip(0,1)
def build(months=9):
    tick=[t.replace('.','-') for t in DEFAULT]
    #d=yf.download(tick,period=f'{months}mo',interval='1d',progress=False); s=yf.download('SPY',period=f'{months}mo',interval='1d',progress=False)
    d = yf.download(tick, period=f'{months}mo', interval='1d',
                progress=False, auto_adjust=False)
    s = yf.download('SPY', period=f'{months}mo', interval='1d',
                progress=False, auto_adjust=False)

    if d.empty or s.empty: return pd.DataFrame()
    C=d['Adj Close']; V=d['Volume']; S=s['Adj Close']; rows=[]
    for t in C.columns:
        c=C[t].dropna(); v=V[t].dropna()
        if len(c)<60: continue
        r5=c.pct_change(5); r20=c.pct_change(20); rs=rsi(c); b=bbpos(c)
        v20=v.rolling(20).mean(); vsp=(v/v20); sma20=c.rolling(20).mean(); sma50=c.rolling(50).mean()
        ab20=(c>sma20).astype(float); ab50=(c>sma50).astype(float); Sr20=S.pct_change(20).reindex(c.index)
        idx=c.index
        for i in range(len(idx)-2):
            d0=idx[i]; d2=idx[i+2]
            try: s0=S.loc[d0]; s2=S.loc[d2]
            except: continue
            rt=(c.iloc[i+2]/c.iloc[i])-1.0; rs2=(s2/s0)-1.0; ex=rt-rs2
            rows.append(dict(Date=d0.strftime('%Y%m%d'),Ticker=t.replace('-','.'),ret5=r5.iloc[i],ret20=r20.iloc[i],rel20=(r20.iloc[i]-(Sr20.iloc[i] if pd.notna(Sr20.iloc[i]) else 0.)),rsi14=rs.iloc[i],vol_spike=(vsp.iloc[i] if pd.notna(vsp.iloc[i]) else 1.),bb_pos=b.iloc[i],above_sma20=ab20.iloc[i],above_sma50=ab50.iloc[i],Outcome2D_Label=int(ex>0),Outcome2D_Excess=ex))
    return pd.DataFrame(rows)
if __name__=='__main__':
    df=build(9); 
    if df.empty: raise SystemExit(0)
    feat=['ret5','ret20','rel20','rsi14','vol_spike','bb_pos','above_sma20','above_sma50']
    X=df[feat].fillna(0).values; y=df['Outcome2D_Label'].astype(int).values
    base=RandomForestClassifier(n_estimators=250,max_depth=8,random_state=42,class_weight='balanced_subsample')
    clf=CalibratedClassifierCV(base,method='sigmoid',cv=3); clf.fit(X,y)
    import pickle; pickle.dump({'clf':clf,'feat_cols':feat}, open(os.path.join(OUTDIR,'model_init.pkl'),'wb'))
    df.to_csv(os.path.join(OUTDIR,'pretrain_dataset.csv'),index=False)
