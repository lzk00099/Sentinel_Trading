import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 系统配置常量 ---
LEVERAGED_ETFS = ["TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXU", "TNA", "TZA", "NVDL", "FAS", "FAZ"]
ENV_TICKERS = ["QQQ", "SPY", "IWM"]

st.set_page_config(page_title="Sentinel Quant Engine", page_icon="🏛️", layout="wide")

# --- 核心算法 1：支撑阻力计算 (Standard Pivot Points) ---
def calculate_pivots(data_1d, ticker):
    """通过前一交易日的最高、最低、收盘价计算客观的支撑与阻力位"""
    try:
        if ticker in data_1d.columns.levels[1]:
            df = data_1d.xs(ticker, level=1, axis=1)
        else:
            df = data_1d
            
        df = df.dropna()
        if len(df) < 2:
            return 0, 0, 0, 0
            
        # 获取前一个完整交易日的数据
        prev_day = df.iloc[-2]
        high = prev_day['High']
        low = prev_day['Low']
        close = prev_day['Close']
        
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        return round(s1, 2), round(s2, 2), round(r1, 2), round(r2, 2)
    except Exception:
        return 0, 0, 0, 0

# --- 核心算法 2：市场结构计算 (POC & 模拟期权墙) ---
def get_market_structure(ticker, data_5m):
    """计算 POC (筹码控制点) 及 期权墙 (心理关口模拟)"""
    try:
        if isinstance(data_5m.columns, pd.MultiIndex):
            df = data_5m.xs(ticker, level=1, axis=1)
        else:
            df = data_5m
            
        df = df.dropna()
        if df.empty:
            return 0, 0, 0
            
        curr_p = df['Close'].iloc[-1]
        
        # 1. 计算 POC (成交量分布)
        # 将价格划分为20个区间，计算每个区间的累计成交量
        bins = np.linspace(df['Low'].min(), df['High'].max(), 20)
        df['Price_Bin'] = pd.cut(df['Close'], bins=bins)
        volume_profile = df.groupby('Price_Bin', observed=False)['Volume'].sum()
        
        # 找到成交量最大的价格区间，取其作为 POC
        poc_interval = volume_profile.idxmax()
        poc = poc_interval.mid if pd.notnull(poc_interval) else curr_p
        
        # 2. 期权墙模拟 (Call Wall / Put Wall)
        # 真实期权墙需读取繁重的 Options Chain，为保证速度，此处采用大整数关口作为阻力/支撑的代理
        step = 5 if curr_p > 50 else 1
        call_wall = np.ceil((curr_p * 1.03) / step) * step  # 现价上方约 3% 的整数关口
        put_wall = np.floor((curr_p * 0.97) / step) * step   # 现价下方约 3% 的整数关口
        
        return round(poc, 2), round(call_wall, 2), round(put_wall, 2)
    except Exception:
        return 0, 0, 0

# --- 核心算法 3：决策引擎 (期望值 EV 与模拟随机森林) ---
def rf_ev_model(ticker, current_price, env_score, is_leveraged):
    """基于环境评分、波动率和杠杆属性计算胜率、期望值和止盈止损策略"""
    # 基础胜率受大盘环境直接影响
    base_win_rate = 0.45 + (env_score * 0.05)
    
    # 杠杆标的惩罚：胜率降低，但潜在止盈空间拉大
    win_rate = min(0.85, base_win_rate) if not is_leveraged else max(0.2, base_win_rate - 0.1)
    tp_pct = 0.08 if not is_leveraged else 0.15
    sl_pct = 0.05 if not is_leveraged else 0.10
    
    # 给出操作建议点位
    suggested_buy = current_price * 0.99 
    tp_price = suggested_buy * (1 + tp_pct)
    sl_price = suggested_buy * (1 - sl_pct)
    
    # 期望值 EV 计算
    ev = (win_rate * tp_pct) - ((1 - win_rate) * sl_pct)
    
    # 周期判定
    cycle = "1-3 周" if not is_leveraged else "1-3 天 (日内/短线)"
    
    return round(win_rate, 2), round(ev, 4), cycle, round(suggested_buy, 2), round(tp_price, 2), round(sl_price, 2)

# --- 主程序界面 ---
def main():
    st.markdown("## 🏛️ Sentinel 独立诊断引擎 (Pro V1)")
    st.markdown("集成量价背离、VWAP乖离、筹码结构(POC)与期望值(EV)评估的量化终端。")
    st.markdown("---")
    
    # 1. 宏观环境评估
    st.markdown("### 🌍 市场环境基准探测")
    with st.spinner("正在扫描三大指数 (QQQ, SPY, IWM)..."):
        env_data = yf.download(ENV_TICKERS, period="5d", interval="1d", progress=False)["Close"]
        env_score = 0
        if not env_data.empty:
            for t in ENV_TICKERS:
                series = env_data[t].dropna()
                if len(series) >= 2 and series.iloc[-1] > series.iloc[-2]:
                    env_score += 1
            
        env_status = "🟢 多头共振" if env_score == 3 else "🟡 震荡分化" if env_score > 0 else "🔴 空头碾压"
        st.info(f"**大盘环境评分: {env_score}/3** ({env_status}) | *决定底层多头胜率基数*")

    # 2. 用户输入
    user_input = st.text_input("🔍 **输入股票代码进行诊断** (最多5个，逗号分隔，如: NVDA, TSLA, SOXL):", "")
    
    if user_input and st.button("🚀 启动全维度计算"):
        raw_tickers = [t.strip().upper() for t in user_input.split(",")]
        tickers = list(set([t for t in raw_tickers if t]))[:5]
        
        if len(raw_tickers) > 5:
            st.warning("⚠️ 超过 5 个代码，系统已自动截取前 5 个进行运算保护。")
            
        with st.spinner("正在抓取高频数据及推演数学模型..."):
            # 批量获取数据
            data_5m = yf.download(tickers, period="5d", interval="5m", progress=False, auto_adjust=True)
            data_1d = yf.download(tickers, period="60d", interval="1d", progress=False, auto_adjust=True)
            
            results = []
            
            for t in tickers:
                try:
                    is_leveraged = t in LEVERAGED_ETFS
                    
                    # 适配 yfinance 对于多标的/单标的的返回结构差异
                    if len(tickers) > 1:
                        c_5 = data_5m["Close"][t].dropna()
                        v_5 = data_5m["Volume"][t].dropna()
                    else:
                        c_5 = data_5m["Close"].dropna()
                        v_5 = data_5m["Volume"].dropna()
                        
                    if c_5.empty:
                        st.error(f"无法获取 {t} 的数据，可能代码有误或停牌。")
                        continue
                        
                    curr_p = c_5.iloc[-1]
                    
                    # VWAP 乖离计算
                    vwap_series = (c_5 * v_5).cumsum() / v_5.cumsum()
                    c_vwap = vwap_series.iloc[-1]
                    bias = (curr_p / c_vwap) - 1
                    
                    # 量价背离计算 (最后5根5分钟K线的线性回归斜率)
                    tail_c, tail_v = c_5.tail(5), v_5.tail(5)
                    is_div = False
                    if len(tail_c) >= 2:
                        slope_c = np.polyfit(np.arange(len(tail_c)), tail_c.values, 1)[0]
                        slope_v = np.polyfit(np.arange(len(tail_v)), tail_v.values, 1)[0]
                        is_div = (slope_c > 0 and slope_v < 0)
                    
                    # 调用算法
                    poc, call_wall, put_wall = get_market_structure(t, data_5m)
                    s1, s2, r1, r2 = calculate_pivots(data_1d, t)
                    win_rate, ev, cycle, buy_p, tp, sl = rf_ev_model(t, curr_p, env_score, is_leveraged)
                    
                    # 综合决策逻辑
                    if is_div or ev < 0:
                        decision = "🚫 诱多 / 观望"
                    elif ev > 0.03 and curr_p > c_vwap and curr_p > poc:
                        decision = "🎯 优选买入 (强势)"
                    else:
                        decision = "☕ 中性持有"
                    
                    results.append({
                        "资产代码": f"{t} {'⚡(杠杆)' if is_leveraged else ''}",
                        "当前现价": f"${curr_p:.2f}",
                        "VWAP乖离": f"{bias:+.2%}",
                        "量价状态": "⚠️ 背离" if is_div else "✅ 同步",
                        "POC(筹码密集)": f"${poc:.2f}" if poc != 0 else "N/A",
                        "期权墙(C | P)": f"${call_wall} | ${put_wall}",
                        "阻力 R1 / R2": f"${r1} / ${r2}",
                        "支撑 S1 / S2": f"${s1} / ${s2}",
                        "胜率模型": f"{win_rate:.0%}",
                        "期望值(EV)": f"{ev:+.3f}",
                        "建议买点": f"${buy_p:.2f}",
                        "止盈 / 止损": f"${tp:.2f} / ${sl:.2f}",
                        "预期持仓周期": cycle,
                        "系统决策": decision
                    })
                except Exception as e:
                    st.error(f"计算 {t} 时发生内部错误: {str(e)}")
                    continue
                    
            if results:
                st.markdown("### 📊 深度诊断输出面板")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                if any("杠杆" in r["资产代码"] for r in results):
                    st.warning("**系统提示**: 检测到带有时间损耗的杠杆标的。引擎已自动执行特殊逻辑：惩罚基础胜率、拉大止损容忍度，并将持仓预期强制降级为日内或超短线级别。")

if __name__ == "__main__":
    main()
