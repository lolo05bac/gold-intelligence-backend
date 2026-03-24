import os
import threading
import time
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="GoldIntel API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

current_signal = {"status": "initializing"}

def compute_rsi(prices, period=14):
    import numpy as np
    deltas = list(map(float, [prices[i] - prices[i-1] for i in range(-(period), 0)]))
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains)/len(gains) if gains else 0
    avg_loss = sum(losses)/len(losses) if losses else 0.001
    return round(100 - (100 / (1 + avg_gain/avg_loss)), 1)

def compute_ema(prices, period):
    m = 2/(period+1)
    e = [prices[0]]
    for p in prices[1:]:
        e.append((p - e[-1])*m + e[-1])
    return e

def update_signal():
    global current_signal
    try:
        import yfinance as yf
        import numpy as np
        gold = yf.download("GC=F", period="250d", progress=False)
        dxy_data = yf.download("DX-Y.NYB", period="60d", progress=False)
        vix_data = yf.download("^VIX", period="60d", progress=False)
        spx_data = yf.download("^GSPC", period="60d", progress=False)
        oil_data = yf.download("CL=F", period="60d", progress=False)
        silver_data = yf.download("SI=F", period="60d", progress=False)
        if len(gold) < 50: return
        c = gold["Close"].values.flatten().astype(float)
        h = gold["High"].values.flatten().astype(float)
        l = gold["Low"].values.flatten().astype(float)
        o = gold["Open"].values.flatten().astype(float)
        rsi14 = compute_rsi(c, 14)
        rsi7 = compute_rsi(c, 7)
        ema12 = compute_ema(c.tolist(), 12)
        ema26 = compute_ema(c.tolist(), 26)
        macd_val = round(ema12[-1]-ema26[-1], 2)
        macd_prev = round(ema12[-2]-ema26[-2], 2)
        macd_sig = "Bullish" if macd_val > macd_prev else "Bearish"
        ma20 = float(np.mean(c[-20:]))
        std20 = float(np.std(c[-20:]))
        bb_up = round(ma20+2*std20, 2)
        bb_lo = round(ma20-2*std20, 2)
        bb_pos = round((c[-1]-bb_lo)/(bb_up-bb_lo)*100, 1) if bb_up != bb_lo else 50
        trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(-14, 0)]
        atr = round(float(np.mean(trs)), 2)
        atr_pct = round(atr/c[-1]*100, 2)
        ret1 = (c[-1]-c[-2])/c[-2]*100
        ret5 = (c[-1]-c[-6])/c[-6]*100 if len(c)>6 else 0
        ret20 = (c[-1]-c[-21])/c[-21]*100 if len(c)>21 else 0
        ma_sigs = []
        for p in [9,21,50,100,200]:
            if len(c)>=p:
                mv = round(float(np.mean(c[-p:])), 2)
                ab = bool(c[-1]>mv)
                d = round((c[-1]-mv)/mv*100, 2)
                ma_sigs.append({"period": f"MA {p}", "value": mv, "signal": "Bullish" if ab else "Bearish", "distance": f"{d:+.2f}%"})
        cross = {}
        if len(c)>=200:
            m50=np.mean(c[-50:]); m200=np.mean(c[-200:])
            cross = {"type": "Above (Bullish)" if m50>m200 else "Below (Bearish)", "signal": "Bullish" if m50>m200 else "Bearish"}
        hi20 = round(float(np.max(h[-20:])), 2)
        lo20 = round(float(np.min(l[-20:])), 2)
        pivot = round((hi20+lo20+float(c[-1]))/3, 2)
        patterns = []
        if len(c)>=10:
            if c[-1]>c[-5] and c[-5]>c[-10]: patterns.append({"name":"Higher Highs","type":"bullish","strength":75})
            if c[-1]<c[-5] and c[-5]<c[-10]: patterns.append({"name":"Lower Lows","type":"bearish","strength":75})
        if len(c)>=20:
            if c[-1]>max(h[-20:-1]): patterns.append({"name":"20-Day Breakout","type":"bullish","strength":80})
            if c[-1]<min(l[-20:-1]): patterns.append({"name":"20-Day Breakdown","type":"bearish","strength":80})
        if len(c)>=20:
            dist20 = (c[-1]-ma20)/ma20*100
            if dist20>3: patterns.append({"name":f"Extended Above MA20 ({dist20:+.1f}%)","type":"bearish","strength":60})
            if dist20<-3: patterns.append({"name":f"Extended Below MA20 ({dist20:+.1f}%)","type":"bullish","strength":60})
        if rsi14>70: patterns.append({"name":"RSI Overbought","type":"bearish","strength":55})
        if rsi14<30: patterns.append({"name":"RSI Oversold","type":"bullish","strength":55})
        dxy_v = float(dxy_data["Close"].values.flatten()[-1]) if len(dxy_data)>0 else 0
        dxy_ch = float((dxy_data["Close"].values.flatten()[-1]-dxy_data["Close"].values.flatten()[-2])/dxy_data["Close"].values.flatten()[-2]*100) if len(dxy_data)>=2 else 0
        vix_v = float(vix_data["Close"].values.flatten()[-1]) if len(vix_data)>0 else 20
        vix_ch = float(vix_data["Close"].values.flatten()[-1]-vix_data["Close"].values.flatten()[-2]) if len(vix_data)>=2 else 0
        spx_r = float((spx_data["Close"].values.flatten()[-1]-spx_data["Close"].values.flatten()[-2])/spx_data["Close"].values.flatten()[-2]*100) if len(spx_data)>=2 else 0
        oil_p = float(oil_data["Close"].values.flatten()[-1]) if len(oil_data)>0 else 0
        oil_r = float((oil_data["Close"].values.flatten()[-1]-oil_data["Close"].values.flatten()[-2])/oil_data["Close"].values.flatten()[-2]*100) if len(oil_data)>=2 else 0
        sil_p = float(silver_data["Close"].values.flatten()[-1]) if len(silver_data)>0 else 0
        sil_r = float((silver_data["Close"].values.flatten()[-1]-silver_data["Close"].values.flatten()[-2])/silver_data["Close"].values.flatten()[-2]*100) if len(silver_data)>=2 else 0
        gs_ratio = round(c[-1]/sil_p, 1) if sil_p>0 else 0
        score = 5.0
        if rsi14>70: score-=0.8
        elif rsi14<30: score+=0.8
        ma_b = sum(1 for m in ma_sigs if m["signal"]=="Bullish")
        ma_br = sum(1 for m in ma_sigs if m["signal"]=="Bearish")
        score += (ma_b-ma_br)*0.25
        if macd_val>0 and macd_val>macd_prev: score+=0.4
        elif macd_val<0 and macd_val<macd_prev: score-=0.4
        if ret5>2: score+=0.7
        elif ret5>0: score+=0.2
        elif ret5<-2: score-=0.7
        elif ret5<0: score-=0.2
        if dxy_ch<-0.3: score+=0.6
        elif dxy_ch>0.3: score-=0.6
        if vix_ch>2: score+=0.5
        elif vix_ch<-2: score-=0.4
        if spx_r<-1: score+=0.5
        elif spx_r>1: score-=0.3
        for p in patterns:
            if p["type"]=="bullish": score+=p["strength"]/250
            else: score-=p["strength"]/250
        if bb_pos>90: score-=0.3
        elif bb_pos<10: score+=0.3
        score = max(1.0, min(10.0, score))
        ws = max(1.0, min(10.0, score + ret5*0.15))
        bull, bear = [], []
        if dxy_ch<0: bull.append({"name":f"USD weakness (DXY {dxy_ch:+.2f}%)","impact":min(int(abs(dxy_ch)*70+40),99)})
        else: bear.append({"name":f"USD strength (DXY {dxy_ch:+.2f}%)","impact":min(int(abs(dxy_ch)*70+30),99)})
        if vix_ch>0: bull.append({"name":f"VIX rising ({vix_v:.1f}, {vix_ch:+.1f})","impact":min(int(vix_ch*12+42),99)})
        else: bear.append({"name":f"VIX declining ({vix_v:.1f}, {vix_ch:+.1f})","impact":min(int(abs(vix_ch)*10+28),99)})
        if ret5>0: bull.append({"name":f"5d momentum +({ret5:+.2f}%)","impact":min(int(ret5*14+42),99)})
        else: bear.append({"name":f"5d momentum -({ret5:+.2f}%)","impact":min(int(abs(ret5)*14+32),99)})
        if spx_r<0: bull.append({"name":f"Equities falling (SPX {spx_r:+.2f}%)","impact":min(int(abs(spx_r)*22+32),99)})
        else: bear.append({"name":f"Equities rising (SPX {spx_r:+.2f}%)","impact":min(int(spx_r*14+22),99)})
        if macd_sig=="Bullish": bull.append({"name":f"MACD bullish ({macd_val:+.1f})","impact":min(int(abs(macd_val)*1.5+38),99)})
        else: bear.append({"name":f"MACD bearish ({macd_val:+.1f})","impact":min(int(abs(macd_val)*1.5+28),99)})
        if oil_r>0.5: bull.append({"name":f"Oil rising ({oil_r:+.2f}%)","impact":min(int(oil_r*10+32),99)})
        elif oil_r<-0.5: bear.append({"name":f"Oil falling ({oil_r:+.2f}%)","impact":min(int(abs(oil_r)*8+22),99)})
        for p in patterns:
            e = {"name":p["name"],"impact":p["strength"]}
            if p["type"]=="bullish": bull.append(e)
            else: bear.append(e)
        bull.sort(key=lambda x:x["impact"], reverse=True)
        bear.sort(key=lambda x:x["impact"], reverse=True)
        if vix_ch>2 and spx_r<-0.5: regime="Risk-Off / Flight to Safety"
        elif abs(dxy_ch)>0.3: regime="USD-Dominant"
        elif ret20>5: regime="Trend / Momentum"
        elif oil_r>1.5: regime="Inflation Hedge"
        elif abs(ret5)<1 and atr_pct<1: regime="Low Vol / Range-Bound"
        else: regime="Mixed / Transitional"
        prob = max(25,min(85,int(score*8+8)))
        conf = max(20,min(90,int(abs(score-5)*14+35)))
        fc = []
        dvol = float(np.std(np.diff(c[-20:])/c[-21:-1])) if len(c)>20 else 0.01
        trend = (c[-1]-c[-5])/c[-5] if len(c)>5 else 0
        today = datetime.now()
        cum = 0
        for i in range(5):
            d = today+timedelta(days=i+1)
            while d.weekday()>=5: d+=timedelta(days=1)
            dm = trend*0.3+(5-ws)*0.001*(1+i*0.1)
            if rsi14>70: dm-=0.003*(i+1)
            elif rsi14<30: dm+=0.003*(i+1)
            cn = max(25,int(70-i*8))
            db = max(1,min(10,ws+dm*200))
            cum+=dm*100
            pt = round(float(c[-1])*(1+cum/100), 2)
            fc.append({"day":d.strftime("%a %d"),"bias":round(db,1),"expected_move":f"{dm*100:+.2f}%","price_target":pt,"confidence":cn})
        current_signal = {
            "signal_date": datetime.now().strftime("%Y-%m-%d"),
            "last_updated": datetime.now().isoformat(),
            "weekly_bias": round(ws,1), "weekly_move": f"{ret5:+.2f}%",
            "weekly_probability": prob, "weekly_confidence": conf,
            "daily_bias": round(score,1), "daily_move": f"{ret1:+.2f}%",
            "regime": regime,
            "bullish_drivers": bull[:8], "bearish_drivers": bear[:8],
            "gold_price": round(float(c[-1]),2), "gold_open": round(float(o[-1]),2),
            "gold_high": round(float(h[-1]),2), "gold_low": round(float(l[-1]),2),
            "dxy": round(float(dxy_v),2), "dxy_change": round(float(dxy_ch),2),
            "vix": round(float(vix_v),1), "vix_change": round(float(vix_ch),2),
            "spx_return": round(float(spx_r),2),
            "oil_price": round(float(oil_p),2), "oil_return": round(float(oil_r),2),
            "silver_price": round(float(sil_p),2), "silver_return": round(float(sil_r),2),
            "gold_silver_ratio": gs_ratio,
            "rsi_14": rsi14, "rsi_7": rsi7,
            "macd": {"value":macd_val,"signal":macd_sig,"prev":macd_prev},
            "atr": atr, "atr_pct": atr_pct,
            "bollinger": {"upper":bb_up,"lower":bb_lo,"position":bb_pos},
            "ma_signals": ma_sigs, "ma_cross": cross,
            "support_resistance": {"resistance_1":round(float(np.percentile(h[-20:],75)),2),"resistance_2":hi20,"support_1":round(float(np.percentile(l[-20:],25)),2),"support_2":lo20,"pivot":pivot},
            "patterns": patterns,
            "five_day_forecast": fc,
            "returns": {"1d":round(ret1,2),"5d":round(ret5,2),"20d":round(ret20,2)},
        }
        print(f"Updated: D{score:.1f} W{ws:.1f} RSI{rsi14} MACD{macd_sig} {datetime.now()}")
    except Exception as e:
        print(f"Update failed: {e}")
        import traceback; traceback.print_exc()

def bg():
    while True:
        try: update_signal()
        except: pass
        time.sleep(600)

@app.on_event("startup")
async def startup():
    threading.Thread(target=bg, daemon=True).start()
    print("Updater started — 10min refresh")

@app.get("/")
async def root():
    return {"service":"GoldIntel API","version":"3.0","status":"operational"}

@app.get("/health")
async def health():
    return {"status":"healthy"}

@app.get("/api/signal/latest")
async def latest():
    return current_signal

@app.get("/api/refresh")
async def refresh():
    update_signal()
    return {"status":"refreshed"}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8080)))
