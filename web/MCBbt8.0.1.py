import sys
import subprocess
import os



# ==========================================
# RunSing Capital System v8.0 (Final Stable)
# ==========================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import optimize

# ==========================================
# 0. 頁面配置
# ==========================================
st.set_page_config(page_title="RunSing | MCB v8.0", layout="wide", page_icon="🦁")

st.title("🦁 RunSing Capital | MCB 戰略指揮中心 v8.0")
st.markdown("---")

# ==========================================
# 1. 資產池配置 (RunSing Universe 50+)
# ==========================================

# 1. RunSing 15 核心精銳 (帶資金權重)
RUNSING_CORE_ASSETS = {
    'Core (重炮 | 3份)': ['NVDA', 'LLY', 'WMT', 'AVGO', 'GOOGL'],
    'Standard (中堅 | 2份)': ['META', 'COST', 'CAT', 'QQQ', 'AAPL'],
    'Aggressive (博弈 | 1份)': ['TQQQ', 'TSLA', 'TSM', 'MSTR', 'IBIT']
}

# 2. 全市場觀察名單 (Universe 50+) - 用於掃描器
SCANNER_TICKERS = [
    # --- 指數 ETF ---
    'QQQ', 'SPY', 'DIA', 'IWM', 'SMH', 'VIG', 'XLV', 'XLF', 'TLT',
    # --- 科技巨頭 (Magnificent 7 + Others) ---
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'ADBE', 'CRM', 'ORCL',
    # --- 半導體 ---
    'AMD', 'AVGO', 'QCOM', 'TXN', 'INTC', 'MU', 'TSM', 'ASML',
    # --- 消費與零售 ---
    'WMT', 'COST', 'TGT', 'HD', 'LOW', 'MCD', 'SBUX', 'NKE', 'KO', 'PEP', 'PG',
    # --- 金融 ---
    'JPM', 'BAC', 'V', 'MA', 'AXP', 'BLK', 'GS',
    # --- 醫療 ---
    'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'PFE',
    # --- 工業與能源 ---
    'CAT', 'DE', 'XOM', 'CVX', 'LMT',
    # --- 加密與區塊鏈 ---
    'IBIT', 'MSTR', 'COIN'
]

# 去重並排序
TICKERS_LIST = sorted(list(set(SCANNER_TICKERS)))

# 建立權重映射 (核心資產用設定權重，其他默認為 1)
WEIGHT_MAP = {}
for cat, tickers in RUNSING_CORE_ASSETS.items():
    w = 3 if 'Core' in cat else (2 if 'Standard' in cat else 1)
    for t in tickers:
        WEIGHT_MAP[t] = w
# 其他資產默認權重為 1
for t in TICKERS_LIST:
    if t not in WEIGHT_MAP:
        WEIGHT_MAP[t] = 1

# ==========================================
# 2. 側邊欄：策略參數
# ==========================================
with st.sidebar:
    st.header("⚙️ 策略核心參數")
    
    # 這裡的 symbol 僅用於 Tab 1 & 2 的單獨回測
    symbol = st.text_input("單一回測代碼", value="WMT") 
    benchmark_symbol = st.text_input("對標資產", value="QQQ")
    
    # 全局資金設定
    base_unit_size = st.number_input("基礎單筆金額 (1份) $", value=50, help="這是權重為1時的買入金額。權重3將自動買入3倍金額。")
    
    st.divider()
    st.subheader("指標參數 (全域)")
    wt_ch_len = st.number_input("Channel Length", value=9)
    wt_avg_len = st.number_input("Average Length", value=12)
    key_lvl_1 = st.slider("一倍買入閾值", -100, 0, -35)
    key_lvl_2 = st.slider("兩倍買入閾值", -100, 0, -60)
    
    st.divider()
    st.caption("Version: 8.0 (Final Stable) Dec/22/2025")

# ==========================================
# 3. 計算核心 (通用函數)
# ==========================================

def calculate_xirr(transactions):
    """計算 XIRR (內部收益率)"""
    if not transactions or len(transactions) < 2: return 0.0
    
    transactions.sort(key=lambda x: x[0])
    dates = [t[0] for t in transactions]
    amounts = [t[1] for t in transactions]
    
    has_pos = any(a > 0 for a in amounts)
    has_neg = any(a < 0 for a in amounts)
    if not (has_pos and has_neg): return 0.0
    
    start_date = dates[0]
    days = [(d - start_date).days for d in dates]
    
    def xnpv(rate):
        if rate <= -1.0: return float('inf')
        return sum([a / ((1 + rate) ** (d / 365.0)) for a, d in zip(amounts, days)])
    
    try:
        return optimize.brentq(xnpv, -0.9999, 100.0, maxiter=100) * 100
    except:
        return 0.0

def calculate_metrics(returns_series, risk_free_rate=0.04):
    """計算夏普、索提諾、波動率"""
    if len(returns_series) < 2 or returns_series.std() == 0:
        return 0.0, 0.0, 0.0
    volatility = returns_series.std() * np.sqrt(252) * 100
    excess_returns = returns_series - (risk_free_rate / 252)
    sharpe = (excess_returns.mean() / returns_series.std()) * np.sqrt(252)
    downside = returns_series[returns_series < 0]
    sortino = (excess_returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() != 0 else 0.0
    return sharpe, sortino, volatility

def calculate_max_drawdown(series):
    """計算最大回撤"""
    if len(series) < 1: return 0.0
    s = pd.Series(series)
    if (s <= 0).all(): return 0.0
    running_max = s.cummax()
    drawdown = (s - running_max) / running_max
    return drawdown.min() * 100

def calculate_runsing_score(res):
    """RunSing 資產評分系統"""
    if not res: return 0
    xirr = res.get('XIRR (%)', 0)
    score_offense = min(40, max(0, (xirr / 25) * 40))
    mdd = abs(res.get('Max DD (%)', 0))
    score_defense = max(0, 40 - (mdd / 50 * 40))
    score_eff = 0
    sharpe = res.get('Sharpe', 0)
    sortino = res.get('Sortino', 0)
    if sharpe >= 1.0: score_eff += 10
    elif sharpe >= 0.5: score_eff += 5
    if sortino >= 1.5: score_eff += 10
    elif sortino >= 1.0: score_eff += 5
    return int(score_offense + score_defense + score_eff)

@st.cache_data(ttl=3600)
def get_data(ticker):
    try:
        df = yf.download(ticker, start="1995-01-01", progress=False, auto_adjust=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        df = df.ffill().bfill()
        
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=wt_ch_len, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=wt_ch_len, adjust=False).mean()
        ci = (ap - esa) / (0.015 * d)
        wt1 = ci.ewm(span=wt_avg_len, adjust=False).mean()
        wt2 = wt1.rolling(window=3).mean()
        
        df['WT1'] = wt1
        df['WT2'] = wt2
        return df
    except Exception:
        return None

def run_simulation(df, start_date, end_date=None, unit_size=1000):
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
    avg_cost_curve = []
    active_returns = []
    
    # 1. 歷史回測循環
    for i in range(1, len(df_test) - 1):
        wt1_p = df_test['WT1'].iloc[i-1]
        wt2_p = df_test['WT2'].iloc[i-1]
        wt1_c = df_test['WT1'].iloc[i]
        wt2_c = df_test['WT2'].iloc[i]
        
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
                'Cost': cost,
                'Status': 'Executed'
            })
            
        current_value = holdings * current_close
        equity_curve.append(current_value)
        avg_cost = (total_invested / holdings) if holdings > 0 else None
        avg_cost_curve.append(avg_cost)
        
        if holdings > 0:
            active_returns.append(df_test['Daily_Ret'].iloc[i+1])

    # ==========================================
    # 2. 檢查最後一天 (捕捉 PENDING 信號)
    # 解決掃描器有信號，但回測因為沒有次日數據而漏掉的問題
    # ==========================================
    last_idx = len(df_test) - 1
    if last_idx > 0:
        wt1_last = df_test['WT1'].iloc[last_idx]
        wt2_last = df_test['WT2'].iloc[last_idx]
        wt1_prev = df_test['WT1'].iloc[last_idx-1]
        wt2_prev = df_test['WT2'].iloc[last_idx-1]
        
        last_cross = (wt1_prev <= wt2_prev) and (wt1_last > wt2_last)
        last_mult = 0
        
        if last_cross:
            if wt1_last <= key_lvl_2: last_mult = 2
            elif wt1_last <= key_lvl_1: last_mult = 1
            
            if last_mult > 0:
                last_date = df_test.index[last_idx]
                ref_price = df_test['Close'].iloc[last_idx]
                buy_signals.append({
                    'Date': last_date,
                    'Price': ref_price,
                    'Units': last_mult,
                    'Cost': unit_size * last_mult,
                    'Status': 'PENDING'
                })

    last_price = df_test['Close'].iloc[-1]
    final_val = holdings * last_price
    
    if total_invested == 0 and not buy_signals: return None
    
    # 只有當有實際投資時才算這些
    if total_invested > 0:
        cash_flows.append((df_test.index[-1], final_val))
        res_xirr = calculate_xirr(cash_flows)
        total_profit = final_val - total_invested
        roi = (total_profit / total_invested) * 100
        sharpe, sortino, volatility = calculate_metrics(pd.Series(active_returns))
        max_dd = calculate_max_drawdown(equity_curve)
    else:
        res_xirr = 0
        total_profit = 0
        roi = 0
        sharpe, sortino, volatility = 0, 0, 0
        max_dd = 0
    
    # DCA 簡化計算
    monthly_groups = df_test.resample('MS').first()
    dca_roi = 0.0
    if len(monthly_groups) > 0 and total_invested > 0:
        monthly_amt = total_invested / len(monthly_groups)
        dca_shares = sum([monthly_amt/r['Open'] for _, r in monthly_groups.iterrows() if not pd.isna(r['Open'])])
        dca_roi = ((dca_shares * last_price - total_invested) / total_invested * 100)
    
    return {
        "Period": f"{start_dt.strftime('%Y-%m')} ~ {end_dt.strftime('%Y-%m')}",
        "Invested": total_invested,
        "Final Value": final_val,
        "Profit": total_profit,
        "ROI (%)": roi,
        "DCA ROI (%)": dca_roi,
        "XIRR (%)": res_xirr,
        "Max DD (%)": max_dd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Volatility (%)": volatility,
        "Total Units": total_units,
        "Buy Signals": buy_signals,
        "DataFrame": df_test,
        "Avg Cost Curve": avg_cost_curve
    }

# ==========================================
# 4. 劇本定義
# ==========================================
SCENARIOS = [
    {"name": "📚 歷史全週期", "start": "1995-01-01", "end": datetime.now().strftime('%Y-%m-%d')},
    {"name": "📉 互聯網泡沫", "start": "2000-03-24", "end": "2002-10-09"},
    {"name": "🌊 金融海嘯", "start": "2007-10-01", "end": "2009-03-09"},
    {"name": "🐢 美股失落十年", "start": "2000-01-01", "end": "2013-01-01"},
    {"name": "😴 2015-2016 盤整", "start": "2015-01-01", "end": "2016-12-31"},
    {"name": "🦠 新冠熔斷 V轉", "start": "2020-02-01", "end": "2020-08-31"},
    {"name": "🚀 2021至今", "start": "2021-01-01", "end": datetime.now().strftime('%Y-%m-%d')},
]

# ==========================================
# 5. 主程式介面
# ==========================================

# 建立分頁
tab1, tab2, tab3 = st.tabs(["📊 歷史週期評分 (Single)", "🛠️ 自定義詳細回測 (Deep Dive)", "📡 RunSing Universe 50+ (Scanner)"])

# ==================================================
# TAB 1: 歷史週期評分 (單一資產)
# ==================================================
with tab1:
    full_df = get_data(symbol)
    bench_df = get_data(benchmark_symbol) if benchmark_symbol else None
    
    if full_df is not None:
        st.subheader(f"📝 {symbol} 資產體檢與評分報告")
        st.caption(f"RS 評分說明：綜合考慮『XIRR (進攻)』、『最大回撤 (防守)』與『夏普比率 (效率)』。滿分100，80分以上為優質資產。")
        
        all_cols = ["累計份數", "ROI (%)", "對標 ROI", "DCA ROI (%)", "XIRR (%)", "最大回撤", "夏普", "索提諾", "波動率"]
        default_cols = ["累計份數", "ROI (%)", "對標 ROI", "DCA ROI (%)", "XIRR (%)", "最大回撤", "夏普", "索提諾", "波動率"]
        selected_cols = st.multiselect("選擇展示數據:", all_cols, default=default_cols)
        
        results_list = []
        
        # 這裡的回測使用 Base Unit * 1 (僅作展示)
        calc_unit = base_unit_size
        
        my_bar = st.progress(0)
        
        for idx, sc in enumerate(SCENARIOS):
            res_main = run_simulation(full_df, sc["start"], sc["end"], unit_size=calc_unit)
            res_bench = None
            if bench_df is not None:
                res_bench = run_simulation(bench_df, sc["start"], sc["end"], unit_size=calc_unit)

            if res_main:
                rs_score = calculate_runsing_score(res_main)
                bench_roi_str = f"{res_bench['ROI (%)']:.1f}%" if res_bench else "N/A"
                
                row_data = {
                    "劇本場景": sc["name"],
                    "時間範圍": res_main["Period"],
                    "RS 評分": rs_score,
                }
                
                raw_data = {
                    "累計份數": res_main['Total Units'],
                    "ROI (%)": f"{res_main['ROI (%)']:.1f}%",
                    "對標 ROI": bench_roi_str,
                    "DCA ROI (%)": f"{res_main['DCA ROI (%)']:.1f}%",
                    "XIRR (%)": f"{res_main['XIRR (%)']:.2f}%",
                    "最大回撤": f"{res_main['Max DD (%)']:.2f}%",
                    "夏普": f"{res_main['Sharpe']:.2f}",
                    "索提諾": f"{res_main['Sortino']:.2f}",
                    "波動率": f"{res_main['Volatility (%)']:.1f}%"
                }
                
                for col in selected_cols:
                    row_data[col] = raw_data[col]
                
                results_list.append(row_data)
            my_bar.progress((idx + 1) / len(SCENARIOS))
        my_bar.empty()
        
        if results_list:
            df_res = pd.DataFrame(results_list)
            cols_order = ["劇本場景", "時間範圍", "RS 評分"] + [c for c in selected_cols]
            st.dataframe(df_res[cols_order], use_container_width=True, height=400)
    else:
        st.error(f"無法獲取 {symbol} 數據，請檢查代碼或網絡。")

# ==================================================
# TAB 2: 自定義詳細回測 (Deep Dive) - Pro Version
# ==================================================
with tab2:
    if full_df is not None:
        col_ctrl, col_chart = st.columns([1, 4])
        
        with col_ctrl:
            st.markdown("### ⏳ 時間與設置")
            min_date = full_df.index[0].date()
            max_date = full_df.index[-1].date()
            default_start = max(min_date, datetime.now().date() - timedelta(days=365*5))
            
            start_input = st.date_input("開始日期", value=default_start, min_value=min_date, max_value=max_date)
            end_input = st.date_input("結束日期", value=max_date, min_value=min_date, max_value=max_date)
            
            st.divider()
            st.markdown("### 🎨 圖表選項")
            show_cost = st.checkbox("顯示平均成本線", value=True)
            log_scale = st.checkbox("使用對數坐標 (Log)", value=False)
            
            sim_res = None
            bench_res = None
            
            # 使用 base_unit 回測
            if start_input < end_input:
                sim_res = run_simulation(full_df, start_input, end_input, unit_size=base_unit_size)
                if bench_df is not None:
                    bench_res = run_simulation(bench_df, start_input, end_input, unit_size=base_unit_size)

        with col_chart:
            if sim_res:
                df_period = sim_res['DataFrame']
                period_score = calculate_runsing_score(sim_res)

                # ==========================================
                # 1. 專業數據儀表板 (三層結構)
                # ==========================================
                st.markdown(f"#### 📊 策略深度面板 (本週期 RS 評分: :red[{period_score} 分])")
                
                # --- 第一層：資金績效 (Financials) ---
                st.caption("💰 資金績效")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("總投入本金", f"${sim_res['Invested']:,.0f}")
                m2.metric("最終資產市值", f"${sim_res['Final Value']:,.0f}", delta=f"${sim_res['Profit']:,.0f}")
                m3.metric("總回報率 (ROI)", f"{sim_res['ROI (%)']:.1f}%")
                cost_val = sim_res['Avg Cost Curve'][-1] if sim_res['Avg Cost Curve'][-1] else 0
                m4.metric("平均持倉成本", f"${cost_val:.2f}")

                st.divider()

                # --- 第二層：風險與效率 (Risk & Efficiency) ---
                st.caption("⚖️ 風險與效率 (硬核指標)")
                r1, r2, r3, r4 = st.columns(4)
                
                # 夏普比率 (Sharpe Ratio)
                sharpe_val = sim_res['Sharpe']
                r1.metric("夏普比率 (Sharpe)", f"{sharpe_val:.2f}", 
                          delta="優秀" if sharpe_val > 1 else ("普通" if sharpe_val > 0.5 else None))
                
                # 索提諾比率 (Sortino Ratio)
                sortino_val = sim_res['Sortino']
                r2.metric("索提諾比率 (Sortino)", f"{sortino_val:.2f}",
                          delta="極佳" if sortino_val > 1.5 else None)
                
                # 波動率 (Volatility)
                r3.metric("年化波動率", f"{sim_res['Volatility (%)']:.1f}%")
                
                # 最大回撤 (Max Drawdown)
                r4.metric("最大回撤 (MaxDD)", f"-{sim_res['Max DD (%)']:.2f}%", delta_color="inverse")

                st.divider()

                # --- 第三層：策略體質與對標 (Strategy Health) ---
                st.caption("📈 策略體質與對標")
                s1, s2, s3, s4 = st.columns(4)
                
                # XIRR
                s1.metric("XIRR (真實年化)", f"{sim_res['XIRR (%)']:.2f}%")
                
                # DCA 對比
                dca_gap = sim_res['ROI (%)'] - sim_res['DCA ROI (%)']
                s2.metric("vs 定投 (DCA)", f"{sim_res['DCA ROI (%)']:.1f}%", delta=f"{dca_gap:.1f}% (超額)")
                
                # 對標 Alpha
                if bench_res:
                    alpha = sim_res['ROI (%)'] - bench_res['ROI (%)']
                    s3.metric(f"vs {benchmark_symbol}", f"{bench_res['ROI (%)']:.1f}%", delta=f"{alpha:.1f}% (Alpha)")
                else:
                    s3.metric(f"vs {benchmark_symbol}", "N/A")
                
                # 交易頻率
                s4.metric("總交易次數", f"{len(sim_res['Buy Signals'])} 次")

                st.divider()

                # ==========================================
                # 2. 交互式圖表
                # ==========================================
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=df_period.index, y=df_period['Close'], mode='lines', name=f'{symbol} 股價', line=dict(color='#bdc3c7', width=1)), secondary_y=False)
                
                if show_cost:
                    fig.add_trace(go.Scatter(x=df_period.index[1:], y=sim_res['Avg Cost Curve'], mode='lines', name='持倉成本', line=dict(color='#f39c12', width=2, dash='dash')), secondary_y=False)
                
                buys = sim_res['Buy Signals']
                if buys:
                    b_dates = [b['Date'] for b in buys]
                    b_prices = [b['Price'] for b in buys]
                    fig.add_trace(go.Scatter(
                        x=b_dates, y=b_prices, mode='markers', name='策略買入',
                        marker=dict(symbol='triangle-up', size=10, color='#00CC96', line=dict(width=1, color='white')),
                        text=[f"買入{b['Units']}份 (${b['Cost']})" for b in buys]
                    ), secondary_y=False)

                if bench_res:
                    df_bench_p = bench_res['DataFrame']
                    fig.add_trace(go.Scatter(x=df_bench_p.index, y=df_bench_p['Close'], mode='lines', name=f'{benchmark_symbol} (同策略)', line=dict(color='#9b59b6', width=1.5, dash='dot')), secondary_y=True)

                fig.update_layout(title=f"策略可視化: {start_input} ~ {end_input}", height=500, template="plotly_dark", hovermode="x unified")
                y_type = "log" if log_scale else "linear"
                fig.update_yaxes(title_text=f"{symbol} 價格", type=y_type, secondary_y=False)
                fig.update_yaxes(title_text=f"{benchmark_symbol} 價格", type=y_type, secondary_y=True, showgrid=False)
                st.plotly_chart(fig, use_container_width=True)

                # ==========================================
                # 3. 交易詳細目錄 (Verification Log)
                # ==========================================
                st.markdown("### 🧾 交易詳細目錄 (Verification Log)")
                
                if buys:
                    df_trades = pd.DataFrame(buys)
                    # 處理 PENDING 信號的顯示
                    df_trades['Type'] = df_trades.apply(
                        lambda x: f"⚠️ 待執行 (PENDING)" if x.get('Status') == 'PENDING' 
                        else ("🔥 2x 強力" if x['Units']==2 else "✅ 1x 標準"), 
                        axis=1
                    )
                    
                    # 格式化日期
                    df_trades['Date'] = df_trades['Date'].dt.strftime('%Y-%m-%d')
                    
                    df_trades = df_trades[['Date', 'Type', 'Price', 'Units', 'Cost']]

                    st.dataframe(
                        df_trades,
                        use_container_width=True,
                        height=300,
                        hide_index=True,
                        column_config={
                            "Date": st.column_config.TextColumn("交易日期"),
                            "Type": st.column_config.TextColumn("信號級別"),
                            "Price": st.column_config.NumberColumn("成交價格", format="$%.2f"),
                            "Units": st.column_config.NumberColumn("買入份數"),
                            "Cost": st.column_config.NumberColumn("投入金額", format="$%.0f"),
                        }
                    )
                    
                    csv = df_trades.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 下載交易記錄 (CSV)",
                        data=csv,
                        file_name=f"{symbol}_trades.csv",
                        mime='text/csv',
                    )
                else:
                    st.info("在此回測區間內，策略未觸發任何交易。")

            else:
                st.warning("⚠️ 該區間內無數據或不足以進行回測，請調整日期。")

# ==================================================
# TAB 3: RunSing 全市場戰情儀表板 - v8.1 (Scanner Logic Fix)
# ==================================================
with tab3:
    st.header("📡 RunSing Universe 50+ 戰情室")
    st.caption("全市場掃描：捕捉「最新收盤確認」的買入信號 (Actionable Signals)。")
    
    # --- 0. 初始化 Session State ---
    if 'scan_results' not in st.session_state:
        st.session_state['scan_results'] = None
    if 'latest_scan_date' not in st.session_state:
        st.session_state['latest_scan_date'] = None

    # --- 1. 參數設定區 ---
    with st.expander("⚙️ 掃描參數設定", expanded=True):
        col_input, col_info = st.columns([2, 8])
        with col_input:
            lookback_years = st.number_input("歷史回溯年數", min_value=1, max_value=20, value=3)
        
        start_scan = st.button("🚀 啟動全市場掃描", type="primary")

    # --- 2. 數據處理邏輯 ---
    if start_scan:
        all_signals = []
        latest_date_seen = None
        
        progress_text = st.empty()
        my_bar = st.progress(0)
        
        total_tickers = len(TICKERS_LIST)
        scan_end = datetime.now()
        scan_start = scan_end - timedelta(days=365 * lookback_years)
        
        for i, ticker in enumerate(TICKERS_LIST):
            progress_text.text(f"正在掃描市場數據: {ticker} ({i+1}/{total_tickers})...")
            df_scan = get_data(ticker)
            
            if df_scan is not None and len(df_scan) > 10:
                last_data_date = df_scan.index[-1].date()
                if latest_date_seen is None or last_data_date > latest_date_seen:
                    latest_date_seen = last_data_date
                current_price = df_scan.iloc[-1]['Close']
                
                res_scan = run_simulation(df_scan, scan_start, scan_end, unit_size=1)
                
                if res_scan and res_scan['Buy Signals']:
                    for signal in res_scan['Buy Signals']:
                        
                        # [關鍵邏輯修正] 區分 "新信號" 和 "已執行"
                        raw_status = signal.get('Status', 'Executed')
                        
                        if raw_status == 'PENDING':
                            # 這是最新的，給你看的重點
                            sig_display = "2x 強力" if signal['Units'] == 2 else "1x 標準"
                            display_status = "NEW" # 標記為新
                        else:
                            # 這是歷史
                            sig_display = "2x 強力" if signal['Units'] == 2 else "1x 標準"
                            display_status = "DONE" # 標記為已完成
                        
                        buy_price = signal['Price']
                        pnl_pct = (current_price - buy_price) / buy_price * 100
                        
                        all_signals.append({
                            "Date": signal['Date'].strftime('%Y-%m-%d'),
                            "Asset": ticker,
                            "Signal_Level": sig_display,     # 顯示用的文字 (1x/2x)
                            "Raw_Status": raw_status,        # 邏輯判斷用的 (PENDING/Executed)
                            "Display_Status": display_status,# UI分組用的 (NEW/DONE)
                            "Buy_Price": buy_price,
                            "Current_Price": current_price,
                            "PnL": pnl_pct
                        })
            my_bar.progress((i + 1) / total_tickers)
            
        progress_text.empty()
        my_bar.empty()
        
        if all_signals:
            df_all = pd.DataFrame(all_signals)
            st.session_state['scan_results'] = df_all
            st.session_state['latest_scan_date'] = latest_date_seen
            st.success(f"掃描完成！")
        else:
            st.session_state['scan_results'] = pd.DataFrame()
            st.warning("在此期間內未發現任何信號。")

    # --- 3. 顯示邏輯 (Native Layout) ---
    if st.session_state['scan_results'] is not None and not st.session_state['scan_results'].empty:
        
        df_all = st.session_state['scan_results']
        latest_date_seen = st.session_state['latest_scan_date']
        today_str = latest_date_seen.strftime('%Y-%m-%d')
        
        # [核心過濾]
        # Actionable Signals (待執行/最新) = PENDING 狀態
        new_signals = df_all[df_all['Raw_Status'] == 'PENDING']
        
        # History Signals (已執行) = Executed 狀態
        history_signals = df_all[df_all['Raw_Status'] == 'Executed']

        # --- 頂部儀表板 ---
        # 這裡的計數只統計 "新信號"，因為這才是你關心的
        action_count = len(new_signals)
        market_sentiment = "🔥 機會湧現" if action_count > 3 else ("🍵 市場平靜" if action_count == 0 else "👀 局部機會")
        
        st.markdown("---")
        k1, k2, k3 = st.columns([1, 1, 2])
        k1.metric("數據截止日期", today_str)
        # 這裡用紅色強調新信號數量
        k2.metric("🚨 需操作信號 (Action)", f"{action_count} 個")
        k3.metric("市場狀態", market_sentiment)
        st.markdown("---")

        # --- 分欄佈局：左邊是重點 (新信號)，右邊是參考 (近期歷史) ---
        col_action, col_history = st.columns([1.2, 1.5])
        
        with col_action:
            st.subheader("🚨 待操作列表 (Buy at Open)")
            st.caption(f"以下信號於 {today_str} 收盤確認，請於下個開盤日執行。")
            
            if not new_signals.empty:
                for _, row in new_signals.iterrows():
                    # 視覺強調：新信號全部用醒目的卡片顯示
                    border_color = "#FF4B4B" if "2x" in row['Signal_Level'] else "#00CC96"
                    bg_color = "rgba(255, 75, 75, 0.1)" if "2x" in row['Signal_Level'] else "rgba(0, 204, 150, 0.1)"
                    
                    st.markdown(f"""
                    <div style="padding: 12px; border-radius: 8px; background-color: {bg_color}; border-left: 5px solid {border_color}; margin-bottom: 12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:1.4em; font-weight:bold; color:#EEE;">{row['Asset']}</span>
                            <span style="color:{border_color}; font-weight:bold; font-size:1.1em;">{row['Signal_Level']}</span>
                        </div>
                        <div style="margin-top:5px; font-size:0.9em; color:#CCC;">
                            收盤確認價: <strong>${row['Buy_Price']:.2f}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🍵 目前沒有新的買入信號，空倉或持倉觀望。")

        with col_history:
            st.subheader("✅ 近期已執行 (Executed)")
            st.caption("過去 7 天已經觸發並成交的信號 (僅供參考)")
            
            seven_days_ago = (datetime.strptime(today_str, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
            recent_executed = history_signals[(history_signals['Date'] > seven_days_ago) & (history_signals['Date'] <= today_str)].sort_values(by="Date", ascending=False)
            
            if not recent_executed.empty:
                # 簡易表格顯示
                st.dataframe(
                    recent_executed[['Date', 'Asset', 'Signal_Level', 'Buy_Price', 'PnL']],
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                    column_config={
                        "Date": st.column_config.TextColumn("執行日期"),
                        "Asset": st.column_config.TextColumn("資產"),
                        "Signal_Level": st.column_config.TextColumn("級別"),
                        "Buy_Price": st.column_config.NumberColumn("成本", format="$%.1f"),
                        "PnL": st.column_config.NumberColumn("當前盈虧", format="%.1f%%"),
                    }
                )
            else:
                st.caption("最近 7 天無交易紀錄。")

        st.divider()

        # --- 歷史流水帳 (保留完整的) ---
        st.subheader(f"📜 歷史交易總覽 (含已執行 & 待執行)")
        
        c_filter, c_sort, c_order = st.columns([2, 1, 1])
        with c_filter:
            all_tickers = sorted(df_all['Asset'].unique().tolist())
            selected_tickers = st.multiselect("🔍 篩選資產", all_tickers, key="filter_asset")
        with c_sort:
            sort_by = st.selectbox("排序依據", ["日期", "資產代號", "盈虧幅度 (PnL)"], index=0, key="sort_by")
        with c_order:
            sort_order = st.selectbox("順序", ["降序 (大到小)", "升序 (小到大)"], index=0, key="sort_order")

        df_display = df_all.copy()
        if selected_tickers:
            df_display = df_display[df_display['Asset'].isin(selected_tickers)]
        
        ascending = True if sort_order == "升序 (小到大)" else False
        if sort_by == "日期":
            df_display = df_display.sort_values(by="Date", ascending=ascending)
        elif sort_by == "資產代號":
            df_display = df_display.sort_values(by="Asset", ascending=ascending)
        elif sort_by == "盈虧幅度 (PnL)":
            df_display = df_display.sort_values(by="PnL", ascending=ascending)

        # 列表顯示
        h1, h2, h3, h4, h5 = st.columns([1.2, 0.8, 1.2, 1.5, 1])
        h1.markdown(":grey[**日期**]")
        h2.markdown(":grey[**資產**]")
        h3.markdown(":grey[**狀態/信號**]")
        h4.markdown(":grey[**價格**]")
        h5.markdown(":grey[**盈虧 %**]")
        st.divider()

        max_rows = 200
        for idx, row in df_display.head(max_rows).iterrows():
            c1, c2, c3, c4, c5 = st.columns([1.2, 0.8, 1.2, 1.5, 1])
            
            c1.write(row['Date'])
            c2.markdown(f"**{row['Asset']}**")
            
            # 狀態顯示邏輯
            if row['Raw_Status'] == 'PENDING':
                # 新信號：橙色閃亮
                c3.markdown(":orange[🔥 **NEW (待買入)**]")
                # 新信號沒有"買入價 -> 現價"的概念，因為還沒買，顯示參考價
                c4.caption(f"參考價: ${row['Buy_Price']:.2f}")
                c5.markdown("---") # 新信號沒有盈虧
            else:
                # 歷史信號
                if "2x" in row['Signal_Level']:
                    c3.markdown(":red[**2x 強力**]")
                else:
                    c3.markdown(":green[**1x 標準**]")
                
                c4.caption(f"${row['Buy_Price']:.1f} ➔ ${row['Current_Price']:.1f}")
                
                pnl = row['PnL']
                if pnl >= 0:
                    c5.markdown(f":green[**▲ {pnl:.1f}%**]")
                else:
                    c5.markdown(f":red[**▼ {abs(pnl):.1f}%**]")
            
            st.divider()

        if len(df_display) > max_rows:
            st.caption(f"⚠️ 僅顯示前 {max_rows} 筆數據 (共 {len(df_display)} 筆)")
