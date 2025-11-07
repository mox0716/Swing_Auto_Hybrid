import os, pickle
import numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

OUTDIR = 'out'
os.makedirs(OUTDIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTDIR, 'model_init.pkl')
DATA_PATH = os.path.join(OUTDIR, 'pretrain_dataset.csv')

# 🛑 Skip retraining if the model already exists
if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
    print(f"✅ Pretraining skipped — model already exists at {MODEL_PATH}")
    raise SystemExit(0)

DEFAULT = [
    'AAPL','MSFT','NVDA','GOOGL','AMZN','META','AVGO','TSLA',
    'LLY','JPM','V','XOM','UNH','MA','HD','PG','COST','ABBV','ORCL','JNJ'
]

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def bbpos(c, w=20, k=2.0):
    ma = c.rolling(w).mean()
    sd = c.rolling(w).std()
    up = ma + k * sd
    lo = ma - k * sd
    span = (up - lo)
    return ((c - lo) / span.replace(0, np.nan)).clip(0, 1)

def build(months=9):
    tickers = [t.replace('.', '-') for t in DEFAULT]
    print(f"Downloading {len(tickers)} symbols for {months} months...")
    d = yf.download(
        tickers, period=f'{months}mo', interval='1d',
        progress=False, auto_adjust=False, group_by='ticker'
    )
    s = yf.download('SPY', period=f'{months}mo', interval='1d',
                    progress=False, auto_adjust=False)

    rows = []
    for t in tickers:
        try:
            df = d[t]
        except Exception:
            continue
        if df.empty or 'Adj Close' not in df.columns:
            continue

        c = df['Adj Close'].dropna()
        v = df['Volume'].dropna()
        if len(c) < 60:
            continue

        r5 = c.pct_change(5)
        r20 = c.pct_change(20)
        rs = rsi(c)
        b = bbpos(c)
        v20 = v.rolling(20).mean()
        vsp = v / v20
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        ab20 = (c > sma20).astype(float)
        ab50 = (c > sma50).astype(float)

        sr20 = s['Adj Close'].pct_change(20).reindex(c.index)

        for i in range(len(c) - 2):
            d0 = c.index[i]
            d2 = c.index[i + 2]
            try:
                s0 = s.loc[d0, 'Adj Close']
                s2 = s.loc[d2, 'Adj Close']
            except Exception:
                continue

            rt = (c.iloc[i + 2] / c.iloc[i]) - 1.0
            rs2 = (s2 / s0) - 1.0
            ex = rt - rs2
            try:
                spy_ret = float(sr20.iloc[i])
            except Exception:
                spy_ret = 0.0
            # --- robust scalar extraction for SPY 20-day return
            val = sr20.iloc[i]

            # handle weird multi-value or misaligned types safely
            if isinstance(val, (pd.Series, np.ndarray, list)):
                try:
                    val = float(val.iloc[0] if hasattr(val, "iloc") else val[0])
                except Exception:
                    val = 0.0
            elif pd.isna(val):
                val = 0.0
            else:
                val = float(val)

            spy_ret = val

            rows.append({
                "Date": d0.strftime("%Y%m%d"),
                "Ticker": t.replace("-", "."),
                "ret5": r5.iloc[i],
                "ret20": r20.iloc[i],
                "rel20": (r20.iloc[i] - spy_ret),
                "rsi14": rs.iloc[i],
                "vol_spike": float(vsp.iloc[i]) if not pd.isna(vsp.iloc[i]) else 1.0,
                "bb_pos": float(b.iloc[i]) if not pd.isna(b.iloc[i]) else 0.5,
                "above_sma20": float(ab20.iloc[i]),
                "above_sma50": float(ab50.iloc[i]),
                "Outcome2D_Label": int(ex > 0),
                "Outcome2D_Excess": float(ex),
            })

    return pd.DataFrame(rows)

if __name__ == '__main__':
    df = build(9)
    if df.empty:
        print("⚠️ No data downloaded — exiting.")
        raise SystemExit(0)

    feat = ['ret5','ret20','rel20','rsi14','vol_spike','bb_pos','above_sma20','above_sma50']
    X = df[feat].fillna(0).values
    y = df['Outcome2D_Label'].astype(int).values

    base = RandomForestClassifier(
        n_estimators=250, max_depth=8,
        random_state=42, class_weight='balanced_subsample'
    )
    clf = CalibratedClassifierCV(base, method='sigmoid', cv=3)
    clf.fit(X, y)

    pickle.dump({'clf': clf, 'feat_cols': feat}, open(MODEL_PATH, 'wb'))
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Pretraining complete — model saved to {MODEL_PATH}")
