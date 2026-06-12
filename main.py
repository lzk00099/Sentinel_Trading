import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
import requests
from requests_html import HTMLSession

session = HTMLSession()
# 在初始化 ticker 时传入 session
ticker = yf.Ticker(symbol, session=session)

# --- 系统配置常量 ---
LEVERAGED_ETFS = ["TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXU", "TNA", "TZA", "NVDL", "FAS", "FAZ"]
ENV_TICKERS = ["QQQ", "SPY", "IWM"]

st.set_page_config(page_title="Sentinel Quant Engine", page_icon="🏛️", layout="wide")

# --- 核心算法 1：支撑阻力计算 ---
def calculate_pivots(data_1d, ticker):
    try:
        if ticker in data_1d.columns.levels[1]:
            df = data_1d.xs(ticker, level=1, axis=1)
        else:
            df = data_1d
        df = df.dropna()
        if len(df) < 2: return 0, 0, 0, 0
        prev_day = df.iloc[-2]
        high, low, close = prev_day['High'], prev_day['Low'], prev_day['Close']
        pivot = (high + low + close) / 3
        return round((2 * pivot) - high, 2), round(pivot - (high - low), 2), round((2 * pivot) - low, 2), round(pivot + (high - low), 2)
    except: return 0, 0, 0, 0

# --- 核心算法 2：筹码与期权墙 ---
def get_market_structure(ticker, data_5m):
    try:
        if isinstance(data_5m.columns, pd.MultiIndex):
            df = data_5m.xs(ticker, level=1, axis=1)
        else:
            df = data_5m
        df = df.dropna()
        if df.empty: return 0, 0, 0
        curr_p = df['Close'].iloc[-1]
        bins = np.linspace(df['Low'].min(), df['High'].max(), 20)
        df['Price_Bin'] = pd.cut(df['Close'], bins=bins)
        volume_profile = df.groupby('Price_Bin', observed=False)['Volume'].sum()
        poc = volume_profile.idxmax().mid if not volume_profile.empty else curr_p
        step = 5 if curr_p > 50 else 1
        return round(poc, 2), round(np.ceil((curr_p * 1.03) / step) * step, 2), round(np.floor((curr_p * 0.97) / step) * step, 2)
    except: return 0, 0, 0

# --- 核心算法 3：财报预警 ---
def get_earnings_status(ticker):
    try:
        tick = yf.Ticker(ticker)
        # 增加 session 模拟，降低被拦截概率
        earnings_dates = tick.calendar 
        
        # 增加对数据为空的判断逻辑
        if earnings_dates is None or (isinstance(earnings_dates, pd.DataFrame) and earnings_dates.empty):
            # 尝试通过 info 字段作为补充备选（有时 calendar 失效，info 仍有数据）
            next_date_str = tick.info.get('nextEarningsDate')
            if next_date_str:
                next_date = pd.to_datetime(next_date_str).date()
            else:
                return "⚪ 未知", 999
        else:
            next_date = earnings_dates.index[0].date()
            
        days_left = (next_date - date.today()).days
        # ... 后续逻辑不变 ...
        days_left = (next_date - date.today()).days
        
        if days_left < 0: return "📅 已发布", 0
        if days_left <= 5: return f"🔴 {days_left}天", days_left
        if days_left <= 15: return f"🟡 {days_left}天", days_left
        return f"🟢 {days_left}天", days_left
    except:
        return "⚪ 获取失败", 999

# --- 核心算法 4：技术形态综合研判 ---
def analyze_technical_position(curr_p, vwap, poc, s1, s2, r1, r2, is_div):
    if is_div:
        return "⚠️ 量价背离 (警惕诱多)"
    
    # 基础位置判断
    if curr_p > r2: return "🚀 极度强势 (突破R2)"
    if curr_p > r1: return "📈 多头趋势 (上方R1)"
    if curr_p > vwap and curr_p > poc: return "🎯 筹码支撑 (多头持有)"
    if curr_p < s2: return "📉 严重超卖 (下方S2)"
    if curr_p < s1: return "🛡️ 支撑测试 (下方S1)"
    return "☕ 中性震荡 (关注POC)"

# --- 主程序 ---
def main():
    st.markdown("## 🏛️ Sentinel 独立诊断引擎 (Pro V2)")
    st.markdown("技术结构、筹码热点与财报风险扫描。")
    st.markdown("---")
    
    user_input = st.text_input("🔍 **输入股票代码 (最多5个，逗号分隔):**", "")
    
    if user_input and st.button("🚀 启动全维度计算"):
        raw_tickers = [t.strip().upper() for t in user_input.split(",")]
        tickers = list(set([t for t in raw_tickers if t]))[:5]
        
        with st.spinner("正在扫描市场结构..."):
            data_5m = yf.download(tickers, period="5d", interval="5m", progress=False, auto_adjust=True)
            data_1d = yf.download(tickers, period="60d", interval="1d", progress=False, auto_adjust=True)
            
            results = []
            for t in tickers:
                try:
                    is_leveraged = t in LEVERAGED_ETFS
                    # 数据适配
                    c_5 = data_5m["Close"][t].dropna() if len(tickers) > 1 else data_5m["Close"].dropna()
                    v_5 = data_5m["Volume"][t].dropna() if len(tickers) > 1 else data_5m["Volume"].dropna()
                    if c_5.empty: continue
                    
                    curr_p = c_5.iloc[-1]
                    vwap = (c_5 * v_5).cumsum() / v_5.cumsum()
                    c_vwap = vwap.iloc[-1]
                    
                    # 线性回归计算量价背离
                    tail_c, tail_v = c_5.tail(5), v_5.tail(5)
                    is_div = False
                    if len(tail_c) >= 2:
                        slope_c = np.polyfit(np.arange(len(tail_c)), tail_c.values, 1)[0]
                        slope_v = np.polyfit(np.arange(len(tail_v)), tail_v.values, 1)[0]
                        is_div = (slope_c > 0 and slope_v < 0)
                    
                    poc, call_w, put_w = get_market_structure(t, data_5m)
                    s1, s2, r1, r2 = calculate_pivots(data_1d, t)
                    earnings_str, _ = get_earnings_status(t)
                    verdict = analyze_technical_position(curr_p, c_vwap, poc, s1, s2, r1, r2, is_div)
                    
                    results.append({
                        "资产": f"{t} {'⚡' if is_leveraged else ''}",
                        "当前价": f"${curr_p:.2f}",
                        "VWAP乖离": f"{(curr_p/c_vwap - 1):+.2%}",
                        "量价状态": "⚠️ 背离" if is_div else "✅ 同步",
                        "POC/墙": f"${poc} ({call_w}|{put_w})",
                        "支撑/阻力": f"S:{s1}/{s2} | R:{r1}/{r2}",
                        "财报预警": earnings_str,
                        "形态决策": verdict
                    })
                except Exception as e:
                    st.error(f"处理 {t} 时出错: {e}")
            
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
