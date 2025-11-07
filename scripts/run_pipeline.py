
import os, re, time, smtplib, ssl, glob, math, pickle
from email.message import EmailMessage
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from datetime import datetime
import requests, numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

SCREENERS = {
    "01_new_highs_liquid": "https://finviz.com/screener.ashx?s=ta_newhigh&v=111&f=sh_price_o5,sh_avgvol_o500",
    "02_top_gainers_liquid": "https://finviz.com/screener.ashx?s=ta_topgainers&v=111&f=sh_price_o5,sh_avgvol_o500",
    "03_unusual_volume_liquid": "https://finviz.com/screener.ashx?s=ta_unusualvolume&v=111&f=sh_price_o5,sh_avgvol_o500",
    "04_most_active_liquid": "https://finviz.com/screener.ashx?s=ta_mostactive&v=111&f=sh_price_o5,sh_avgvol_o500",
    "05_channel_up_pullbacks": "https://finviz.com/screener.ashx?s=ta_channelup&v=111&f=sh_price_o5,sh_avgvol_o500",
    "06_new_lows_liquid": "https://finviz.com/screener.ashx?s=ta_newlow&v=111&f=sh_price_o5,sh_avgvol_o500",
    "07_gap_watch_liquid": "https://finviz.com/screener.ashx?s=ta_gapper&v=111&f=sh_price_o5,sh_avgvol_o500",
    "08_overbought": "https://finviz.com/screener.ashx?s=ta_overbought&v=111&f=sh_price_o5,sh_avgvol_o500",
    "09_oversold": "https://finviz.com/screener.ashx?s=ta_oversold&v=111&f=sh_price_o5,sh_avgvol_o500",
    "10_pattern_wedge_resist": "https://finviz.com/screener.ashx?s=ta_p_wedgeresistance&v=111&f=sh_price_o5,sh_avgvol_o500",
}
OUTDIR = "out"; os.makedirs(OUTDIR, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FinvizScreenerBot/1.0)"}
MIN_AVG_VOL = 1_000_000

def _with_r(url, rstart):
    u = urlparse(url); q = parse_qs(u.query); q["r"] = [str(rstart)]
    return urlunparse((u.scheme,u.netloc,u.path,u.params, urlencode({k:v[0] for k,v in q.items()}), u.fragment))

def fetch_tickers(url):
    tickers, seen = [], set()
    for r in range(1, 1000, 20):
        resp = requests.get(_with_r(url,r), headers=HEADERS, timeout=30)
        if resp.status_code != 200: break
        html = resp.text
        if "screener-body-table" not in html and "screener-body-table-nw" not in html:
            if r>1: break
        found = re.findall(r"quote\.ashx\?t=([A-Z\.\-]+)", html)
        new = [t for t in found if t not in seen]
        if not new and r>1: break
        for t in new: seen.add(t); tickers.append(t)
        time.sleep(0.2)
    return tickers

def yahoo_ticker(t): return t.replace(".", "-")

def rsi(series, window=14):
    d = series.diff(); up=(d.clip(lower=0)).ewm(alpha=1/window,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/window,adjust=False).mean()
    rs=up/(dn+1e-9); return 100-(100/(1+rs))

def bollinger_pos(close, window=20, num_std=2.0):
    ma=close.rolling(window).mean(); sd=close.rolling(window).std()
    up=ma+num_std*sd; lo=ma-num_std*sd; sp=(up-lo); return ((close-lo)/sp.replace(0,np.nan)).clip(0,1)

def base_features(close, vol):
    ret5=close.pct_change(5).iloc[-1]; ret20=close.pct_change(20).iloc[-1]
    sma20=close.rolling(20).mean().iloc[-1]; sma50=close.rolling(50).mean().iloc[-1] if len(close)>=50 else np.nan
    above20=float(close.iloc[-1]>sma20) if not np.isnan(sma20) else 0.0
    above50=float((not np.isnan(sma50)) and (close.iloc[-1]>sma50)) if not np.isnan(sma50) else 0.0
    rsi14=rsi(close).iloc[-1]; vol20=vol.rolling(20).mean().iloc[-1]
    vol_spike=float(vol.iloc[-1]/vol20) if vol20 and vol20>0 else 1.0
    bb_pos=bollinger_pos(close).iloc[-1]
    return dict(ret5=ret5,ret20=ret20,rsi14=rsi14,vol_spike=vol_spike,bb_pos=bb_pos,above_sma20=above20,above_sma50=above50,avg_vol20=vol20)

def logistic(x): return 1/(1+math.exp(-x))

def baseline_prob(m, rel20, screener_key):
    mom5=np.tanh(m["ret5"]*5); mom20=np.tanh(m["ret20"]*3); rel20_n=np.tanh(rel20*5); vol_n=np.tanh((m["vol_spike"]-1.0))
    trend_n=0.6*m["above_sma20"]+0.8*m["above_sma50"]
    if any(k in screener_key for k in ("new_highs","top_gainers","unusual_volume","most_active","gap_watch")):
        bb_tilt=(m["bb_pos"]-0.5)*1.0; rsi_tilt=(m["rsi14"]-50.0)/30.0
    elif any(k in screener_key for k in ("new_lows","oversold")):
        bb_tilt=(0.35-abs(m["bb_pos"]-0.2)); rsi_tilt=(40.0-m["rsi14"])/40.0
    else:
        bb_tilt=(m["bb_pos"]-0.5)*0.5; rsi_tilt=(50.0-abs(m["rsi14"]-50.0))/50.0
    score=1.2*mom5+1.0*mom20+1.0*rel20_n+0.8*vol_n+0.7*trend_n+0.6*bb_tilt+0.6*rsi_tilt
    return logistic(score)

def get_upcoming_earnings_date(ticker):
    yt=yahoo_ticker(ticker)
    try:
        tk=yf.Ticker(yt)
        try:
            ed=tk.get_earnings_dates(limit=5)
            if ed is not None and not ed.empty:
                today=datetime.utcnow().date()
                for idx in ed.index:
                    d=pd.Timestamp(idx).date()
                    if d>=today: return d
        except Exception: pass
        try:
            cal=tk.calendar
            if cal is not None and not cal.empty:
                if "Earnings Date" in cal.index:
                    val=cal.loc["Earnings Date"].values[0]
                else:
                    val=list(cal.values.flatten())[0]
                if pd.notna(val):
                    return pd.to_datetime(val).date()
        except Exception: pass
    except Exception: pass
    return None

def build_html_summary(per_screener_frames, recent_winrate=None, filtered_counts=None):
    style = """
    <style>
      body { font-family: Arial, Helvetica, sans-serif; line-height: 1.35; color: #111; }
      .wrap { max-width: 960px; margin: 0 auto; }
      .head { padding: 12px 0; }
      .head h1 { margin: 0 0 4px 0; font-size: 18px; }
      .meta { color: #555; font-size: 12px; margin-bottom: 12px; }
      .scn { margin: 12px 0 24px 0; }
      .scn h3 { margin: 0 0 8px 0; font-size: 15px; }
      .link { font-size: 12px; margin-bottom: 6px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; text-align: left; }
      th { background: #f6f6f6; }
      .num { text-align: right; }
    </style>
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"<html><head>{style}</head><body><div class='wrap'>",
             "<div class='head'><h1>Finviz Overnight Swing — Hybrid ML</h1>",
             f"<div class='meta'>Generated {now} · Top rows sorted by ML% (if available)</div>"]
    if filtered_counts:
        total = filtered_counts.get("low_volume",0)+filtered_counts.get("near_earnings",0)
        parts.append(f"<div class='meta'><b>Filtered out {total} symbols ({filtered_counts.get('low_volume',0)} low volume, {filtered_counts.get('near_earnings',0)} near earnings)</b></div>")
    if recent_winrate is not None:
        parts.append(f"<div class='meta'><b>Recent top-quartile baseline win-rate (2d vs SPY): {recent_winrate:.1f}%</b></div>")
    parts.append("</div>")
    for scn_name, info in per_screener_frames.items():
        df = info["df"]; url = info["url"]
        if df is None or df.empty: continue
        cols=["Ticker","Probability"]; 
        if "ML_Probability" in df.columns: cols.append("ML_Probability")
        cols += ["rsi14","vol_spike","bb_pos","above_sma20","above_sma50"]
        dfp = df.copy()
        if "ML_Probability" in dfp.columns:
            dfp = dfp.sort_values(by=["ML_Probability","Probability"], ascending=False, na_position="last")
        else:
            dfp = dfp.sort_values(by=["Probability"], ascending=False, na_position="last")
        dfp = dfp.head(15)
        if "Probability" in dfp.columns: dfp["Probability"] = dfp["Probability"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
        if "ML_Probability" in dfp.columns: dfp["ML_Probability"] = dfp["ML_Probability"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
        parts.append(f"<div class='scn'><h3>{scn_name}</h3>")
        parts.append(f"<div class='link'><a href='{url}'>Open screener</a></div>")
        parts.append(dfp[cols].to_html(index=False, escape=False))
        parts.append("</div>")
    parts.append("</div></body></html>"); return "".join(parts)

def send_email(subject, text_body, html_body, attachments):
    to_email=os.environ["TO_EMAIL"]; from_email=os.environ["FROM_EMAIL"]
    server=os.environ["SMTP_SERVER"]; port=int(os.environ.get("SMTP_PORT","587"))
    username=os.environ["SMTP_USERNAME"]; password=os.environ["SMTP_PASSWORD"]
    msg=EmailMessage(); msg["Subject"]=subject; msg["From"]=from_email; msg["To"]=to_email
    msg.set_content(text_body); msg.add_alternative(html_body, subtype="html")
    for path in attachments:
        with open(path,"rb") as f: data=f.read()
        msg.add_attachment(data, maintype="text", subtype="csv", filename=os.path.basename(path))
    ctx=ssl.create_default_context(); 
    with smtplib.SMTP(server,port) as smtp:
        smtp.starttls(context=ctx); smtp.login(username,password); smtp.send_message(msg)

def main():
    pre_ds_path=os.path.join(OUTDIR,"pretrain_dataset.csv"); training_path=os.path.join(OUTDIR,"training_data.csv")

    now = datetime.utcnow()
    today = now.strftime("%Y%m%d")
    timestamp = now.strftime("%H%M")  # adds hour+minute for uniqueness
    attachments = []
    all_rows = []
    per_screener_frames = {}
    filtered_rows = []

    def ts_name(base):
        """Helper: build timestamped filenames like run_20251107_1940.csv"""
        return f"{base}_{today}_{timestamp}.csv"


#    today=datetime.utcnow().strftime("%Y%m%d")
#    attachments=[]; all_rows=[]; per_screener_frames={}; filtered_rows=[]
#    spy=yf.download("SPY",period="4mo",interval="1d",progress=False); spy_ret20=spy["Adj Close"].pct_change(20)
    spy = yf.download("SPY", period="4mo", interval="1d", progress=False, auto_adjust=False)
    spy_ret20 = spy["Adj Close"].pct_change(20)

    for name,url in SCREENERS.items():
        tickers=fetch_tickers(url); rows=[]
        if tickers:
            data=yf.download([yahoo_ticker(t) for t in tickers],period="4mo",interval="1d",auto_adjust=False,progress=False,threads=True)
            for t in tickers:
                yt=yahoo_ticker(t)
                try:
                    close=data["Adj Close"][yt].dropna(); vol=data["Volume"][yt].dropna()
                except Exception: continue
                if len(close)<30: continue
                feats=base_features(close,vol)
                if pd.notna(feats["avg_vol20"]) and feats["avg_vol20"]<MIN_AVG_VOL:
                    filtered_rows.append({"Ticker":t,"Screener":name,"Reason":"low_volume"}); continue
                nxt=get_upcoming_earnings_date(t)
                if nxt is not None:
                    delta=(nxt - datetime.utcnow().date()).days
                    if 0<=delta<=2:
                        filtered_rows.append({"Ticker":t,"Screener":name,"Reason":"near_earnings"}); continue
                #aligned=spy_ret20.align(close.pct_change(20), join="right"); spy_rs=aligned[0].iloc[-1] if not aligned[0].empty else 0.0
                aligned = spy_ret20.align(close.pct_change(20), join="right", axis=0)
                spy_rs = aligned[0].iloc[-1] if not aligned[0].empty else 0.0

                #ret20=close.pct_change(20).iloc[-1]; rel20=(ret20 - (spy_rs if pd.notna(spy_rs) else 0.0))
                ret20 = close.pct_change(20).iloc[-1]

                # make sure spy_rs is a scalar
                if isinstance(spy_rs, (pd.Series, np.ndarray, list)):
                    spy_rs = spy_rs.iloc[-1] if hasattr(spy_rs, "iloc") else float(spy_rs[-1])
                try:
                    spy_rs = float(spy_rs)
                except Exception:
                    spy_rs = 0.0

                rel20 = ret20 - spy_rs

                p=baseline_prob(feats, rel20, name)
                rows.append({"Date":today,"Screener":name,"Ticker":t,"Probability":p*100,
                            "ret5":feats["ret5"],"ret20":feats["ret20"],"rel20":rel20,
                            "rsi14":feats["rsi14"],"vol_spike":feats["vol_spike"],"bb_pos":feats["bb_pos"],
                            "above_sma20":feats["above_sma20"],"above_sma50":feats["above_sma50"]})
        df_sc=pd.DataFrame(rows); out_path=os.path.join(OUTDIR,f"{name}.csv")
        #(df_sc[["Ticker","Probability","ret5","rel20","rsi14","vol_spike","bb_pos","above_sma20","above_sma50"]] if not df_sc.empty else df_sc).to_csv(out_path,index=False)

        if not df_sc.empty:
            # 🧩 Keep only the latest row per ticker for this screener
            df_sc = (
                df_sc.sort_values(["Ticker", "Date"])
                .groupby("Ticker", as_index=False)
                .tail(1)
            )

            df_sc[["Ticker", "Probability", "ret5", "rel20", "rsi14",
                   "vol_spike", "bb_pos", "above_sma20", "above_sma50"]
                  ].to_csv(out_path, index=False)
        else:
            df_sc.to_csv(out_path, index=False)


        attachments.append(out_path); per_screener_frames[name]={"df":df_sc.copy(),"url":url}; all_rows.extend(rows)
    #df_today=pd.DataFrame(all_rows); df_today.to_csv(os.path.join(OUTDIR,f"run_{today}.csv"),index=False)
    # Combine all results
    df_today = pd.DataFrame(all_rows)

    # 🧩 Fix: keep only the latest prediction per ticker
    if not df_today.empty:
        df_today = (
            df_today.sort_values(["Ticker", "Date"])
            .groupby("Ticker", as_index=False)
            .tail(1)
        )

    # Save the cleaned, unique output
    df_today.to_csv(os.path.join(OUTDIR, f"run_{today}.csv"), index=False)

    # save filtered
    pd.DataFrame(filtered_rows).to_csv(os.path.join(OUTDIR,"Filtered_Out.csv"), index=False); attachments.append(os.path.join(OUTDIR,"Filtered_Out.csv"))
    # augment training with outcomes after 3 days
    runs=sorted(glob.glob(os.path.join(OUTDIR,"run_*.csv"))); rows_new=[]
    for rpath in runs:
        m=re.match(r"run_(\d{8})\.csv", os.path.basename(rpath)); 
        if not m: continue
        ref=m.group(1); ref_dt=datetime.strptime(ref,"%Y%m%d")
        if (datetime.utcnow()-ref_dt).days<3: continue
        d=pd.read_csv(rpath)
        if "Outcome2D_Label" in d.columns: continue
        def outcome(tk):
            y_t=yahoo_ticker(tk); hist=yf.download([y_t,"SPY"],period="6mo",interval="1d",auto_adjust=False,progress=False)
            if hist.empty: return np.nan, np.nan
            if isinstance(hist.columns, pd.MultiIndex):
                px_t=hist["Adj Close"][y_t].dropna(); px_s=hist["Adj Close"]["SPY"].dropna()
            else:
                px_t=hist["Adj Close"].dropna(); px_s=yf.download("SPY",period="6mo",interval="1d",progress=False)["Adj Close"].dropna()
            idx=px_t.index.tz_localize(None); d0=pd.Timestamp(datetime.strptime(ref,"%Y%m%d").date())
            if d0 not in idx:
                prev=idx[idx<=d0]; 
                if prev.empty: return np.nan,np.nan
                d0=prev.max()
            try: d2=idx[idx>d0][1]
            except Exception: return np.nan,np.nan
            ret_t=(px_t.loc[d2]/px_t.loc[d0])-1.0; idxs=px_s.index.tz_localize(None); d0s=d0 if d0 in idxs else idxs[idxs<=d0].max()
            try: d2s=idxs[idxs>d0s][1]
            except Exception: return np.nan,np.nan
            ret_s=(px_s.loc[d2s]/px_s.loc[d0s])-1.0; ex=ret_t-ret_s
            return (1 if ex>0 else 0), ex
        labs, exs = [], []
        for _,r in d.iterrows():
            L,E = outcome(r["Ticker"]); labs.append(L); exs.append(E); time.sleep(0.05)
        d["Outcome2D_Label"]=labs; d["Outcome2D_Excess"]=exs; d.to_csv(rpath,index=False); rows_new.append(d)
    if rows_new:
        all_new=pd.concat(rows_new, ignore_index=True)
        tp=os.path.join(OUTDIR,"training_data.csv")
        if os.path.exists(tp):
            old=pd.read_csv(tp); combo=pd.concat([old,all_new], ignore_index=True).drop_duplicates(subset=["Date","Screener","Ticker"])
        else:
            combo=all_new
        combo.to_csv(tp,index=False)
    # Train hybrid model
    model=None; feat=["ret5","ret20","rel20","rsi14","vol_spike","bb_pos","above_sma20","above_sma50"]
    Xs,Ys,Ws=[],[],[]
    if os.path.exists(pre_ds_path):
        d0=pd.read_csv(pre_ds_path).dropna(subset=["Outcome2D_Label"])
        if not d0.empty:
            Xs.append(d0[feat].fillna(0).values); Ys.append(d0["Outcome2D_Label"].astype(int).values); Ws.append(np.ones(len(d0))*1.0)
    tp=os.path.join(OUTDIR,"training_data.csv")
    if os.path.exists(tp):
        d1=pd.read_csv(tp).dropna(subset=["Outcome2D_Label"])
        if not d1.empty:
            Xs.append(d1[feat].fillna(0).values); Ys.append(d1["Outcome2D_Label"].astype(int).values); Ws.append(np.ones(len(d1))*3.0)
    if Xs:
        X=np.vstack(Xs); y=np.concatenate(Ys); w=np.concatenate(Ws)
        base=RandomForestClassifier(n_estimators=250,max_depth=8,random_state=42,class_weight='balanced_subsample')
        clf=CalibratedClassifierCV(base,method='sigmoid',cv=3); clf.fit(X,y,sample_weight=w)
        model={"clf":clf,"feat_cols":feat}
    if model is not None and not df_today.empty:
        proba=model["clf"].predict_proba(df_today[feat].fillna(0).values)[:,1]; df_today["ML_Probability"]=proba*100.0
        for name in SCREENERS.keys():
            path=os.path.join(OUTDIR,f"{name}.csv"); 
            if not os.path.exists(path): continue
            d=pd.read_csv(path); m=df_today[df_today["Screener"]==name][["Ticker","ML_Probability"]].copy()
            d=d.merge(m,on="Ticker", how="left"); d.to_csv(path,index=False)
            per_screener_frames[name]["df"]=d
    # Build bodies
    recent=None; tp=os.path.join(OUTDIR,'training_data.csv')
    if os.path.exists(tp):
        try:
            tr=pd.read_csv(tp).dropna(subset=['Outcome2D_Label'])
            if not tr.empty:
                top=tr.sort_values('Probability', ascending=False).head(max(10,len(tr)//4))
                recent=top['Outcome2D_Label'].mean()*100.0
        except Exception: pass
    filtered_counts={'low_volume': sum(1 for r in filtered_rows if r['Reason']=='low_volume'),
                     'near_earnings': sum(1 for r in filtered_rows if r['Reason']=='near_earnings')}
    lines=["Finviz Overnight Swing — Hybrid ML (Filtered)",
           f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
           f"Filtered out {filtered_counts['low_volume']+filtered_counts['near_earnings']} symbols ({filtered_counts['low_volume']} low volume, {filtered_counts['near_earnings']} near earnings)"]
    if recent is not None: lines.append(f"Recent top-quartile baseline win-rate (2d vs SPY): {recent:.1f}%")
    for name in SCREENERS.keys():
        path=os.path.join(OUTDIR,f"{name}.csv")
        try:
            d=pd.read_csv(path); n=len(d); show='ML_Probability' if 'ML_Probability' in d.columns else 'Probability'
            prev=", ".join([f"{row.Ticker}({row[show]:.1f}%)" for _,row in d.sort_values(by=[show], ascending=False).head(10).iterrows()])
            lines.append(f"• {name}: {n} tickers"); 
            if prev: lines.append(f"  {prev}{' ...' if n>10 else ''}")
        except Exception: pass
    text_body="\n".join(lines); html_body=build_html_summary(per_screener_frames, recent_winrate=recent, filtered_counts=filtered_counts)
    send_email("Finviz Overnight Swing — Daily Digest (Hybrid ML + Filters + HTML)", text_body, html_body, attachments + [os.path.join(OUTDIR,'Filtered_Out.csv')])
    df_today.to_csv(os.path.join(OUTDIR,f"run_{today}.csv"),index=False)

if __name__=='__main__':
    main()
