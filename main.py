import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import io
import os
import urllib.parse
import urllib.request

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

# --- 核心算法 3：财报预警数据源 ---
ALPHA_VANTAGE_EARNINGS_URL = "https://www.alphavantage.co/query"


def _get_alpha_vantage_key():
    try:
        secrets_key = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
    except Exception:
        secrets_key = ""
    return secrets_key or os.getenv("ALPHAVANTAGE_API_KEY", "")


def _http_get_text(url, params=None, timeout=10):
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    request = urllib.request.Request(
        f"{url}{query}",
        headers={"User-Agent": "SentinelQuantEngine/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _coerce_future_dates(raw_value):
    today = datetime.date.today()
    dates = []

    def collect(value):
        if value is None:
            return
        if isinstance(value, pd.DataFrame):
            collect(value.index)
            for col in value.columns:
                collect(value[col])
            return
        if isinstance(value, pd.Series):
            collect(value.index)
            collect(value.tolist())
            return
        if isinstance(value, dict):
            for dict_value in value.values():
                collect(dict_value)
            return
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
            for item in value:
                collect(item)
            return

        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            return
        if pd.isna(parsed):
            return

        next_date = parsed.date() if hasattr(parsed, "date") else None
        if next_date and next_date >= today:
            dates.append(next_date)

    collect(raw_value)
    return sorted(set(dates))


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _fetch_alpha_vantage_earnings_date(ticker_symbol, api_key):
    if not api_key:
        return None

    params = {
        "function": "EARNINGS_CALENDAR",
        "symbol": ticker_symbol,
        "horizon": "3month",
        "apikey": api_key,
    }
    try:
        csv_text = _http_get_text(ALPHA_VANTAGE_EARNINGS_URL, params=params)
        if not csv_text.strip() or csv_text.lstrip().startswith("{"):
            return None

        earnings_df = pd.read_csv(io.StringIO(csv_text))
        if "reportDate" not in earnings_df.columns:
            return None

        future_dates = _coerce_future_dates(earnings_df["reportDate"])
        return future_dates[0] if future_dates else None
    except Exception:
        return None


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _fetch_yfinance_earnings_date(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        raw_dates = []

        try:
            raw_dates.append(ticker.get_earnings_dates(limit=12))
        except Exception:
            pass

        try:
            raw_dates.append(ticker.calendar)
        except Exception:
            pass

        future_dates = _coerce_future_dates(raw_dates)
        return future_dates[0] if future_dates else None
    except Exception:
        return None


def _format_earnings_status(next_date, source_label):
    if not next_date:
        return "⚪ 无数据", 999

    days_left = (next_date - datetime.date.today()).days
    date_label = next_date.strftime("%m-%d")

    if days_left <= 0:
        return f"📅 近期发布 ({date_label}, {source_label})", 0
    if days_left <= 5:
        return f"🔴 {days_left}天 ({date_label}, {source_label})", days_left
    if days_left <= 15:
        return f"🟡 {days_left}天 ({date_label}, {source_label})", days_left
    return f"🟢 {days_left}天 ({date_label}, {source_label})", days_left

def get_earnings_status(ticker_symbol):
    """
    Alpha Vantage 的文档化接口优先；没有 API Key 或接口无数据时，
    使用 yfinance 作为轻量备选，避免财报预警拖垮主流程。
    """
    api_key = _get_alpha_vantage_key()

    next_date = _fetch_alpha_vantage_earnings_date(ticker_symbol, api_key)
    if next_date:
        return _format_earnings_status(next_date, "AV")

    next_date = _fetch_yfinance_earnings_date(ticker_symbol)
    if next_date:
        return _format_earnings_status(next_date, "YF备选")

    if not api_key:
        return "⚪ 无数据/未配AV Key", 999
    return "⚪ 无数据", 999

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
                    verdict = analyze_technical_position(curr_p, c_vwap, poc, s1, s2, r1, r2, is_div)
                    earnings_str, _ = get_earnings_status(t)
                    
                    results.append({
                        "资产": f"{t} {'⚡' if is_leveraged else ''}",
                        "当前价": f"${curr_p:.2f}",
                        "VWAP乖离": f"{(curr_p/c_vwap - 1):+.2%}",
                        "量价状态": "⚠️ 背离" if is_div else "✅ 同步",
                        "POC/墙": f"${poc} ({call_w}|{put_w})",
                        "支撑/阻力": f"S:{s1}/{s2} | R:{r1}/{r2}",
                        "形态决策": verdict,
                        "财报预警": earnings_str
                    })
                except Exception as e:
                    st.error(f"处理 {t} 时出错: {e}")
            
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
