import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.optimize import newton


# ==========================================
# 6.2.2 Dec/5/2025
# 基本沒問題
# ==========================================

# ==========================================
# 0. 頁面配置
# ==========================================
st.set_page_config(page_title="RunSing | MCBDCA Backtest", layout="wide", page_icon="🏦")

st.title("🚀 RunSing Capital | MCB策略深度回測")

# ==========================================
# 1. 側邊欄：策略參數
# ==========================================
with st.sidebar:
    st.header("⚙️ 策略核心參數")
    symbol = st.text_input("股票代碼", value="WMT")
    unit_size = st.number_input("單筆金額", value=1000)
    st.divider()
    st.subheader("指標參數")
    wt_ch_len = st.number_input("Channel", value=9)
    wt_avg_len = st.number_input("Average", value=12)
    key_lvl_1 = st.slider("一倍買入閾值", -100, 0, -35)
    key_lvl_2 = st.slider("兩倍買入閾值", -100, 0, -60)
    st.divider()
    # st.caption("數據來源: Yahoo Finance")
    st.caption("系統版本: Dec/5/2025 6.2.2")

# ==========================================
# 2. 計算核心
# ==========================================
def calculate_xirr(transactions):
    if not transactions or len(transactions) < 2: return 0
    dates = [t[0] for t in transactions]
    amounts = [t[1] for t in transactions]
    start_date = min(dates)
    days = [(d - start_date).days for d in dates]
    
    def xnpv(rate, amounts, days):
        if rate <= -1.0: return float('inf')
        return sum([a / ((1 + rate) ** (d / 365.0)) for a, d in zip(amounts, days)])
    
    try:
        return newton(lambda r: xnpv(r, amounts, days), 0.1) * 100
    except:
        return 0

def calculate_metrics(returns_series, risk_free_rate=0.04):
    if len(returns_series) < 2 or returns_series.std() == 0:
        return 0, 0, 0
    volatility = returns_series.std() * np.sqrt(252) * 100
    daily_rf = risk_free_rate / 252
    excess_returns = returns_series - daily_rf
    sharpe = (excess_returns.mean() / returns_series.std()) * np.sqrt(252)
    
    downside_returns = returns_series[returns_series < 0]
    if len(downside_returns) > 0 and downside_returns.std() != 0:
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    else:
        sortino = 0
    return sharpe, sortino, volatility

def calculate_max_drawdown(series):
    """通用最大回撤計算"""
    if len(series) < 1: return 0
    s = pd.Series(series)
    if (s <= 0).all(): return 0
    running_max = s.cummax()
    drawdown = (s - running_max) / running_max
    return drawdown.min() * 100

def get_simple_roi(series):
    """計算簡單持有回報率"""
    if len(series) < 2: return 0
    start_price = series.iloc[0]
    end_price = series.iloc[-1]
    if start_price == 0: return 0
    return ((end_price - start_price) / start_price) * 100

@st.cache_data
def get_data(ticker):
    try:
        df = yf.download(ticker, start="1995-01-01", progress=False, auto_adjust=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=wt_ch_len, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=wt_ch_len, adjust=False).mean()
        ci = (ap - esa) / (0.015 * d)
        wt1 = ci.ewm(span=wt_avg_len, adjust=False).mean()
        wt2 = wt1.rolling(window=3).mean()
        
        df['WT1'] = wt1
        df['WT2'] = wt2
        return df
    except Exception as e:
        return None

def run_simulation(df, start_date, end_date=None):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) if end_date else df.index[-1]
    
    df_test = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()
    if len(df_test) < 10: return None
    
    df_test['Daily_Ret'] = df_test['Close'].pct_change().fillna(0)
    
    holdings = 0
    total_invested = 0
    total_units = 0
    cash_flows = []
    buy_signals = []
    equity_curve = []
    active_returns = []
    
    for i in range(1, len(df_test) - 1):
        wt1_c = df_test['WT1'].iloc[i]
        wt2_c = df_test['WT2'].iloc[i]
        wt1_p = df_test['WT1'].iloc[i-1]
        wt2_p = df_test['WT2'].iloc[i-1]
        
        crossover = (wt1_p <= wt2_p) and (wt1_c > wt2_c)
        mult = 0
        if crossover:
            if wt1_c <= key_lvl_2: mult = 2
            elif wt1_c <= key_lvl_1: mult = 1
            
        current_date = df_test.index[i+1]
        current_open = df_test['Open'].iloc[i+1]
        current_close = df_test['Close'].iloc[i+1]
        
        if mult > 0:
            cost = unit_size * mult
            shares = cost / current_open
            holdings += shares
            total_invested += cost
            total_units += mult
            
            cash_flows.append((current_date, -cost))
            buy_signals.append({
                'Date': current_date,
                'Price': current_open,
                'Units': mult,
                'Cost': cost
            })
            
        current_value = holdings * current_close
        equity_curve.append(current_value)
        
        if holdings > 0:
            active_returns.append(df_test['Daily_Ret'].iloc[i+1])

    last_price = df_test['Close'].iloc[-1]
    final_val = holdings * last_price
    
    if total_invested == 0: return None
    
    cash_flows.append((df_test.index[-1], final_val))
    res_xirr = calculate_xirr(cash_flows)
    total_profit = final_val - total_invested
    roi = (total_profit / total_invested) * 100
    sharpe, sortino, volatility = calculate_metrics(pd.Series(active_returns))
    max_dd = calculate_max_drawdown(equity_curve)
    
    calmar = 0
    if max_dd != 0: calmar = res_xirr / abs(max_dd)
    
    return {
        "Period": f"{start_dt.strftime('%Y-%m')} ~ {end_dt.strftime('%Y-%m')}",
        "Invested": total_invested,
        "Final Value": final_val,
        "Profit": total_profit,
        "ROI (%)": roi,
        "XIRR (%)": res_xirr,
        "Max DD (%)": max_dd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Volatility (%)": volatility,
        "Calmar": calmar,
        "Total Units": total_units,
        "Buy Signals": buy_signals,
        "DataFrame": df_test,
        "Equity Curve": equity_curve
    }


# ==========================================
# 3. 劇本定義 (已更新 RunSing Capital 壓力測試版)
# ==========================================
SCENARIOS = [
    # --- 歷史大考 ---
    {"name": "互聯網泡沫", "start": "2000-03-24", "end": "2002-10-09"},
    {"name": "2008 金融海嘯", "start": "2007-10-01", "end": "2009-03-09"},
    {"name": "美股失落十年", "start": "2000-01-01", "end": "2013-01-01"},

    # --- 牛市與盤整 (測試策略是否被磨損) ---
    {"name": "2010-2020 長牛", "start": "2010-01-01", "end": "2020-01-01"},
    {"name": "2015-2016 盤整震盪", "start": "2015-01-01", "end": "2016-12-31"}, # [新增] 測試假信號與磨損

    # --- 極端行情 (測試指標滯後性) ---
    {"name": "2020 新冠熔斷 V轉", "start": "2020-02-01", "end": "2020-08-31"}, # [新增] 測試暴跌暴漲反應速度
    {"name": "2022 加息熊市", "start": "2022-01-01", "end": "2022-12-31"},

    # --- 近期表現 ---
    {"name": "2019至今", "start": "2019-01-01", "end": datetime.now().strftime('%Y-%m-%d')},
    {"name": "2021至今", "start": "2021-01-01", "end": datetime.now().strftime('%Y-%m-%d')},
    {"name": "近期 AI 浪潮", "start": "2023-01-01", "end": datetime.now().strftime('%Y-%m-%d')}
]

# ==========================================
# 4. 主邏輯 UI
# ==========================================
full_df = get_data(symbol)

if full_df is not None:
    tab1, tab2 = st.tabs(["📊 歷史週期報告", "🛠️ 自定義參數回測"])
    
    # --- Tab 1: 體檢報告單 ---
    with tab1:
        st.subheader(f"📝 {symbol} 歷史重要時段回測")
        results_list = []
        my_bar = st.progress(0)
        
        for idx, sc in enumerate(SCENARIOS):
            res = run_simulation(full_df, sc["start"], sc["end"])
            if res:
                results_list.append({
                    "劇本場景": sc["name"],
                    "時間範圍": res["Period"],
                    "ROI (%)": f"{res['ROI (%)']:.1f}",
                    "XIRR (年化 %)": f"{res['XIRR (%)']:.2f}",
                    "最大回撤 (%)": f"{res['Max DD (%)']:.2f}",
                    "夏普比率": f"{res['Sharpe']:.2f}",
                    "索提諾": f"{res['Sortino']:.2f}",
                    "波動率 (%)": f"{res['Volatility (%)']:.1f}",
                    "累計份數": res['Total Units']
                })
            my_bar.progress((idx + 1) / len(SCENARIOS))
        my_bar.empty()
        
        if results_list:
            df_res = pd.DataFrame(results_list)
            cols = ["劇本場景", "時間範圍", "累計份數", "ROI (%)", "XIRR (年化 %)", "最大回撤 (%)", "夏普比率", "索提諾", "波動率 (%)"]
            st.dataframe(df_res[cols], use_container_width=True, height=400)

    # --- Tab 2: 自定義詳細回測 (含對比功能) ---
    with tab2:
        col_ctrl, col_chart = st.columns([1, 4])
        
        with col_ctrl:
            st.markdown("### 1️⃣ 時間設定")
            min_date = full_df.index[0].date()
            max_date = full_df.index[-1].date()
            default_start = max(min_date, datetime.now().date() - timedelta(days=365*5))
            
            start_input = st.date_input("開始", value=default_start, min_value=min_date, max_value=max_date)
            end_input = st.date_input("結束", value=max_date, min_value=min_date, max_value=max_date)
            
            st.divider()
            st.markdown("### 2️⃣ 對比選項")
            show_bnh = st.checkbox("顯示 Buy & Hold 對比", value=False)
            show_bench = st.checkbox("加入對標資產 (如 QQQ)", value=False)
            bench_symbol = st.text_input("對標代碼", value="QQQ", disabled=not show_bench)
            
            sim_res = None
            if start_input < end_input:
                sim_res = run_simulation(full_df, start_input, end_input)

        with col_chart:
            if sim_res:
                df_period = sim_res['DataFrame']

                # --------------------------
                # A. 專業數據儀表板 (三行式)
                # --------------------------
                st.markdown("#### 📊 策略核心績效")
                
                # 第一行：資金概況
                m1, m2, m3 = st.columns(3)
                m1.metric("💰 總投入成本", f"${sim_res['Invested']:,.0f}", help="所有買入信號觸發的總金額")
                m2.metric("💵 最終資產市值", f"${sim_res['Final Value']:,.0f}", delta=f"${sim_res['Profit']:,.0f} (獲利)")
                m3.metric("📦 累計投入份數", f"{sim_res['Total Units']} 份")

                # 第二行：回報指標
                r1, r2, r3 = st.columns(3)
                r1.metric("🚀 總回報率 (ROI)", f"{sim_res['ROI (%)']:.1f}%", help="總利潤 / 總投入成本")
                r2.metric("📅 複合年化 (XIRR)", f"{sim_res['XIRR (%)']:.2f}%", help="考慮資金時間價值的真實年化回報")
                r3.metric("📉 最大回撤 (MDD)", f"-{sim_res['Max DD (%)']:.2f}%", help="資產從最高點回落的最大幅度")

                # 第三行：風險係數
                k1, k2, k3 = st.columns(3)
                k1.metric("🛡️ 夏普比率 (Sharpe)", f"{sim_res['Sharpe']:.2f}", help="衡量每單位風險帶來的超額回報 (越早越好)")
                k2.metric("⚖️ 索提諾比率 (Sortino)", f"{sim_res['Sortino']:.2f}", help="僅考慮下行風險的回報比率")
                k3.metric("🌊 年化波動率", f"{sim_res['Volatility (%)']:.1f}%")

                st.divider()

                # --------------------------
                # B. 對比數據表格
                # --------------------------
                if show_bnh or show_bench:
                    st.markdown("#### 🆚 資產 PK：策略 vs 其他")
                    comparison_data = []
                    # 1. 策略本身
                    comparison_data.append({
                        "項目": f"🔵 策略 ({symbol})",
                        "總回報 (ROI)": f"{sim_res['ROI (%)']:.1f}%",
                        "最大回撤": f"-{sim_res['Max DD (%)']:.2f}%",
                        "夏普比率": f"{sim_res['Sharpe']:.2f}",
                        "年化波動率": f"{sim_res['Volatility (%)']:.1f}%"
                    })
                    # 2. Buy & Hold
                    if show_bnh:
                        bnh_roi = get_simple_roi(df_period['Close'])
                        bnh_dd = calculate_max_drawdown(df_period['Close'])
                        bnh_vol = df_period['Close'].pct_change().std() * np.sqrt(252) * 100
                        # 簡單夏普 (無風險暫設0.04)
                        bnh_sharpe = ((df_period['Close'].pct_change().mean() * 252 - 0.04) / (df_period['Close'].pct_change().std() * np.sqrt(252))) if df_period['Close'].pct_change().std() != 0 else 0
                        comparison_data.append({
                            "項目": f"🟠 Buy & Hold ({symbol})",
                            "總回報 (ROI)": f"{bnh_roi:.1f}%",
                            "最大回撤": f"-{bnh_dd:.2f}%",
                            "夏普比率": f"{bnh_sharpe:.2f}",
                            "年化波動率": f"{bnh_vol:.1f}%"
                        })
                    # 3. Benchmark
                    bench_df = None
                    if show_bench and bench_symbol:
                        bench_df = get_data(bench_symbol)
                        if bench_df is not None:
                            bench_period = bench_df[(bench_df.index >= pd.to_datetime(start_input)) & (bench_df.index <= pd.to_datetime(end_input))]
                            if not bench_period.empty:
                                bench_roi = get_simple_roi(bench_period['Close'])
                                bench_dd = calculate_max_drawdown(bench_period['Close'])
                                bench_vol = bench_period['Close'].pct_change().std() * np.sqrt(252) * 100
                                bench_sharpe = ((bench_period['Close'].pct_change().mean() * 252 - 0.04) / (bench_period['Close'].pct_change().std() * np.sqrt(252))) if bench_period['Close'].pct_change().std() != 0 else 0
                                comparison_data.append({
                                    "項目": f"🟣 對標 ({bench_symbol})",
                                    "總回報 (ROI)": f"{bench_roi:.1f}%",
                                    "最大回撤": f"-{bench_dd:.2f}%",
                                    "夏普比率": f"{bench_sharpe:.2f}",
                                    "年化波動率": f"{bench_vol:.1f}%"
                                })
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

                # --------------------------
                # C. 圖表繪製
                # --------------------------
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # 主圖
                fig.add_trace(go.Scatter(
                    x=df_period.index, y=df_period['Close'],
                    mode='lines', name=f'{symbol} 股價',
                    line=dict(color='gray', width=1)
                ), secondary_y=False)

                # 買入信號
                buys = sim_res['Buy Signals']
                if buys:
                    b_dates = [b['Date'] for b in buys]
                    b_prices = [b['Price'] for b in buys]
                    b_sizes = [b['Units'] * 6 + 4 for b in buys]
                    fig.add_trace(go.Scatter(
                        x=b_dates, y=b_prices,
                        mode='markers', name='買入操作',
                        marker=dict(symbol='triangle-up', size=b_sizes, color='#00CC96', line=dict(width=1, color='white')),
                        text=[f"買入{b['Units']}份<br>${b['Price']:.1f}" for b in buys],
                        hoverinfo='text'
                    ), secondary_y=False)

                # 對標
                if show_bench and bench_symbol and 'bench_df' in locals() and bench_df is not None:
                    bench_period = bench_df[(bench_df.index >= pd.to_datetime(start_input)) & (bench_df.index <= pd.to_datetime(end_input))]
                    if not bench_period.empty:
                        fig.add_trace(go.Scatter(
                            x=bench_period.index, y=bench_period['Close'],
                            mode='lines', name=f'{bench_symbol} (對標)',
                            line=dict(color='#AB63FA', width=1.5, dash='dot')
                        ), secondary_y=True)

                fig.update_layout(
                    title=f"策略回測與資產走勢: {start_input} 至 {end_input}",
                    height=600,
                    template="plotly_dark",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_yaxes(title_text=f"{symbol} 價格", secondary_y=False)
                if show_bench:
                    fig.update_yaxes(title_text=f"{bench_symbol} 價格", secondary_y=True, showgrid=False)

                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("查看詳細交易記錄"):
                    st.dataframe(pd.DataFrame(buys), use_container_width=True)
