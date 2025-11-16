import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from scipy import stats
import gspread
from google.oauth2.service_account import Credentials

# ========================
# CẤU HÌNH TRANG
# ========================
st.set_page_config(page_title="SPC & MSA Dashboard", layout="wide")
st.title("SPC & MSA Real-time Dashboard")
st.markdown("**Theo tiêu chuẩn AIAG, ISO 22514, Western Electric Rules**")

# ========================
# HÀM KẾT NỐI GOOGLE SHEETS
# ========================
@st.cache_resource
def connect_to_gsheet():
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file("cred.json", scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Lỗi kết nối Google Sheets: {str(e)}")
        return None

@st.cache_data(ttl=300)
def load_all_data(_client, spreadsheet_id):
    try:
        spreadsheet = _client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.get_worksheet(0)
        all_values = worksheet.get_all_values()
        if len(all_values) < 2: return pd.DataFrame()
        headers = [col.strip() for col in all_values[0]]
        df = pd.DataFrame(all_values[1:], columns=headers)
        
        # Chuẩn hóa Measure
        if 'Measure' in df.columns:
            df['Measure'] = df['Measure'].astype(str).str.strip()\
                .str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            df['Measure'] = pd.to_numeric(df['Measure'], errors='coerce')
        
        # Chuẩn hóa các cột khác
        for col in ['Date', 'ProductItem', 'CheckerId', 'WorkerId', 'Trial']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        df = df[df['Measure'].notna() & (df['ProductItem'] != '') & (df['ProductItem'] != 'nan')].reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Lỗi load data: {str(e)}")
        return pd.DataFrame()

# ========================
# HÀM TÍNH CONTROL LIMITS (I-MR) - CHUẨN AIAG
# ========================
def calculate_imr_limits(values):
    if len(values) < 2:
        return 0, 0, 0, 0, 0, 0
    mean = np.mean(values)
    mr = np.abs(np.diff(values))
    mr_bar = np.mean(mr)
    d2 = 1.128  # AIAG constant for n=2
    sigma = mr_bar / d2
    ucl_i = mean + 3 * sigma
    lcl_i = max(0, mean - 3 * sigma)
    ucl_mr = 3.267 * mr_bar
    lcl_mr = 0
    return mean, ucl_i, lcl_i, mr_bar, ucl_mr, lcl_mr, sigma

# ========================
# PHÁT HIỆN SPC RULES (Western Electric)
# ========================
def detect_spc_violations(values, mean, ucl, lcl):
    violations = []
    n = len(values)
    sigma = (ucl - mean) / 3
    
    # Rule 1: 1 point beyond 3σ
    if np.any(values > ucl) or np.any(values < lcl):
        violations.append("Rule 1: 1 điểm ngoài 3σ")
    
    # Rule 2: 2/3 points > 2σ
    zone_a = mean + 2*sigma
    zone_b = mean - 2*sigma
    for i in range(n-2):
        side = 1 if values[i] > mean else -1
        if ((values[i] - mean) * side > 2*sigma and
            (values[i+1] - mean) * side > 2*sigma and
            (values[i+2] - mean) * side > 2*sigma):
            violations.append(f"Rule 2: 2/3 điểm > 2σ tại {i+1}-{i+3}")
            break
    
    # Rule 4: 8 points same side
    for i in range(n-7):
        if all(v > mean for v in values[i:i+8]) or all(v < mean for v in values[i:i+8]):
            violations.append(f"Rule 4: 8 điểm cùng phía tại {i+1}-{i+8}")
            break
    
    # Rule 5: 6 increasing/decreasing
    for i in range(n-5):
        if all(values[j] < values[j+1] for j in range(i, i+5)) or \
           all(values[j] > values[j+1] for j in range(i, i+5)):
            violations.append(f"Rule 5: 6 điểm tăng/giảm tại {i+1}-{i+6}")
            break
    
    return violations if violations else ["Không vi phạm quy tắc SPC"]

# ========================
# SIDEBAR
# ========================
st.sidebar.header("Cấu hình")
spreadsheet_id = st.sidebar.text_input("Google Sheet ID", value="1K41CnzAim6ZtEnvvjqpMfoWNjCmt_P_iFKlcAJzWtNQ", disabled=True)
usl = st.sidebar.number_input("USL", value=10.0, format="%.4f")
lsl = st.sidebar.number_input("LSL", value=0.0, format="%.4f")
target = st.sidebar.number_input("Target", value=5.0, format="%.4f")

# ========================
# LOAD DATA
# ========================
if spreadsheet_id:
    client = connect_to_gsheet()
    if client:
        df_all = load_all_data(client, spreadsheet_id)
        if not df_all.empty and 'ProductItem' in df_all.columns:
            items = sorted(df_all['ProductItem'].unique())
            selected = st.sidebar.selectbox("Product Item", items)
            df = df_all[df_all['ProductItem'] == selected].copy()
            df = df.sort_values('Date').reset_index(drop=True)
            
            if not df.empty:
                values = df['Measure'].values
                n = len(values)
                mean, ucl, lcl, mr_bar, ucl_mr, lcl_mr, sigma = calculate_imr_limits(values)
                
                # Sidebar metrics
                st.sidebar.markdown("---")
                st.sidebar.subheader("Control Limits")
                st.sidebar.metric("UCL (I)", f"{ucl:.4f}")
                st.sidebar.metric("Mean", f"{mean:.4f}")
                st.sidebar.metric("LCL (I)", f"{lcl:.4f}")
                st.sidebar.metric("MR̄", f"{mr_bar:.4f}")
                st.sidebar.metric("UCL (MR)", f"{ucl_mr:.4f}")

                # ========================
                # DỮ LIỆU & VIOLATIONS
                # ========================
                st.subheader(f"Dữ liệu: {selected}")
                display_df = df[['Date', 'Measure']].copy()
                display_df['Status'] = display_df['Measure'].apply(lambda x: 'Out' if x > ucl or x < lcl else 'In')
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Tổng mẫu", n)
                with col2: st.metric("In Control", f"{(np.logical_and(values >= lcl, values <= ucl)).sum()}/{n}")
                with col3: st.metric("Out Control", f"{(values > ucl).sum() + (values < lcl).sum()}/{n}")
                st.dataframe(display_df, use_container_width=True, height=300)

                # SPC Rules
                violations = detect_spc_violations(values, mean, ucl, lcl)
                st.markdown("### SPC Rule Violations")
                for v in violations:
                    if "Không" in v:
                        st.success(v)
                    else:
                        st.warning(v)

                # ========================
                # I-MR CHART
                # ========================
                st.markdown("### I-MR Control Chart")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**I-Chart**")
                    fig_i = go.Figure()
                    fig_i.add_scatter(x=list(range(1, n+1)), y=values, mode='lines+markers', name='Value')
                    fig_i.add_hline(mean, line_dash="solid", line_color="green", annotation_text="Mean")
                    fig_i.add_hline(ucl, line_dash="dash", line_color="red", annotation_text="UCL")
                    fig_i.add_hline(lcl, line_dash="dash", line_color="red", annotation_text="LCL")
                    fig_i.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_i, use_container_width=True)
                
                with col2:
                    st.markdown("**MR-Chart**")
                    if len(values) > 1:
                        mr = np.abs(np.diff(values))
                        fig_mr = go.Figure()
                        fig_mr.add_scatter(x=list(range(2, n+1)), y=mr, mode='lines+markers', name='MR')
                        fig_mr.add_hline(mr_bar, line_dash="solid", line_color="green")
                        fig_mr.add_hline(ucl_mr, line_dash="dash", line_color="red")
                        fig_mr.add_hline(lcl_mr, line_dash="dash", line_color="red")
                        fig_mr.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_mr, use_container_width=True)

               # ========================
                # HISTOGRAM + NORMAL (ĐÃ SỬA LỖI BINS)
                # ========================
                st.markdown("### Histogram & Normality")
                col1, col2 = st.columns(2)

                # --- Tính bins an toàn theo AIAG ---
                n_bins = max(5, min(20, int(np.sqrt(n) if n > 0 else 10)))

                # Xác định phạm vi thực tế
                data_min, data_max = values.min(), values.max()
                spec_min, spec_max = lsl, usl

                # Ưu tiên dùng USL/LSL nếu hợp lệ, nếu không thì dùng min/max dữ liệu
                if spec_min < spec_max:
                    plot_min = max(data_min, spec_min)
                    plot_max = min(data_max, spec_max)
                else:
                    plot_min = data_min
                    plot_max = data_max

                # Đảm bảo plot_min < plot_max
                if plot_min >= plot_max:
                    # Nếu dữ liệu chỉ có 1 giá trị hoặc bằng nhau
                    if n == 1:
                        plot_min = values[0] - 0.1
                        plot_max = values[0] + 0.1
                    else:
                        margin = (data_max - data_min) * 0.1 or 0.1
                        plot_min = data_min - margin
                        plot_max = data_max + margin

                # Tạo bins tăng dần
                bins = np.linspace(plot_min, plot_max, n_bins + 1)

                # Kiểm tra bins tăng dần (debug nếu cần)
                if not np.all(np.diff(bins) > 0):
                    st.error("Lỗi tạo bins: không tăng dần")
                    st.write(f"plot_min={plot_min}, plot_max={plot_max}, bins={bins}")
                else:
                    # ========================
                    # HISTOGRAM - TỰ ĐỘNG ZOOM VÀO VÙNG DỮ LIỆU
                    # ========================
                    with col1:
                        st.markdown("**Histogram (Tự động zoom vào dữ liệu)**")

                        # --- BƯỚC 1: Xác định vùng dữ liệu chính (loại bỏ outlier nhẹ) ---
                        if n <= 1:
                            center = values[0]
                            plot_min = center * 0.9
                            plot_max = center * 1.1
                        else:
                            # Dùng percentile 1% và 99% để loại outlier
                            p1, p99 = np.percentile(values, [1, 99])
                            iqr = p99 - p1
                            margin = max(iqr * 0.4, (values.max() - values.min()) * 0.1)  # ít nhất 10%
                            plot_min = p1 - margin
                            plot_max = p99 + margin

                            # Đảm bảo không âm nếu dữ liệu dương
                            if plot_min < 0 and values.min() >= 0:
                                plot_min = 0

                        # --- BƯỚC 2: Tạo bins TỰ ĐỘNG trong vùng zoom ---
                        n_bins = max(6, min(25, int(np.sqrt(n))))  # 6–25 bins
                        bin_edges = np.linspace(plot_min, plot_max, n_bins + 1)
                        bin_width = (plot_max - plot_min) / n_bins

                        # --- BƯỚC 3: Vẽ Histogram ---
                        fig_hist = go.Figure()
                        hist_counts, _ = np.histogram(values, bins=bin_edges)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                        fig_hist.add_trace(go.Bar(
                            x=bin_centers,
                            y=hist_counts,
                            name="Data",
                            marker_color="#1f77b4",
                            width=bin_width * 0.85,
                            hovertemplate="Giá trị: %{x:.3f}<br>Tần suất: %{y}<extra></extra>"
                        ))

                        # --- BƯỚC 4: Đường cong chuẩn (chỉ trong vùng zoom) ---
                        x_curve = np.linspace(plot_min, plot_max, 300)
                        pdf_scaled = stats.norm.pdf(x_curve, mean, sigma) * n * bin_width
                        fig_hist.add_trace(go.Scatter(
                            x=x_curve, y=pdf_scaled,
                            mode='lines', line=dict(color='red', width=3),
                            name='Phân phối chuẩn'
                        ))

                        # --- BƯỚC 5: Hiển thị LSL/USL làm tham chiếu (nếu có) ---
                        def add_spec_line(val, label, color="black"):
                            if val is not None and not np.isnan(val):
                                if plot_min <= val <= plot_max:
                                    fig_hist.add_vline(x=val, line_dash="dash", line_color=color,
                                                    annotation_text=label, annotation_position="top")
                                elif val < plot_min:
                                    fig_hist.add_annotation(x=plot_min, y=max(hist_counts)*0.95,
                                                        text=f"{label}←", showarrow=True, arrowhead=2,
                                                        ax=30, ay=0, font=dict(color=color))
                                elif val > plot_max:
                                    fig_hist.add_annotation(x=plot_max, y=max(hist_counts)*0.95,
                                                        text=f"→{label}", showarrow=True, arrowhead=2,
                                                        ax=-30, ay=0, font=dict(color=color))

                        add_spec_line(lsl, "LSL", "green")
                        add_spec_line(usl, "USL", "red")
                        if target and not np.isnan(target):
                            add_spec_line(target, "Target", "orange")

                        # --- BƯỚC 6: Layout đẹp, tự động zoom ---
                        fig_hist.update_layout(
                            height=450,
                            xaxis_title="Giá trị đo",
                            yaxis_title="Tần suất",
                            showlegend=False,
                            plot_bgcolor="white",
                            bargap=0.05,
                            xaxis=dict(
                                range=[plot_min, plot_max],
                                showgrid=True, gridcolor="lightgray",
                                tickformat=".3f"
                            ),
                            yaxis=dict(
                                showgrid=True, gridcolor="lightgray",
                                zeroline=False
                            ),
                            margin=dict(l=50, r=50, t=50, b=50),
                            hovermode="x unified"
                        )

                        st.plotly_chart(fig_hist, use_container_width=True)
                                
                    with col2:
                        st.markdown("**Normal Probability Plot**")
                        sorted_val = np.sort(values)
                        p = (np.arange(1, n+1) - 0.5) / n
                        theo = stats.norm.ppf(p, mean, sigma)
                        fig_pp = go.Figure()
                        fig_pp.add_scatter(x=sorted_val, y=p*100, mode='markers', name='Data')
                        slope, intercept = np.polyfit(theo, sorted_val, 1)
                        line = slope * theo + intercept
                        fig_pp.add_scatter(x=line, y=p*100, mode='lines', line=dict(color='red'), name='Fit')
                        fig_pp.update_layout(height=400, yaxis=dict(tickvals=[1,5,10,20,50,80,90,95,99]))
                        st.plotly_chart(fig_pp, use_container_width=True)

                # ========================
                # PROCESS CAPABILITY
                # ========================
                st.markdown("### Process Capability (Cp, Cpk, PPM, Sigma)")
                cp = (usl - lsl) / (6 * sigma) if sigma > 0 else 0
                cpu = (usl - mean) / (3 * sigma) if sigma > 0 else 0
                cpl = (mean - lsl) / (3 * sigma) if sigma > 0 else 0
                cpk = min(cpu, cpl)
                ppm = 1e6 * (len(values[values > usl]) + len(values[values < lsl])) / n
                yield_pct = 100 * (1 - ppm / 1e6)
                sigma_level = stats.norm.ppf(1 - ppm/2e6) if ppm > 0 else 6

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cp", f"{cp:.3f}")
                    st.metric("Cpk", f"{cpk:.3f}")
                with col2:
                    st.metric("PPM", f"{ppm:,.1f}")
                    st.metric("Yield", f"{yield_pct:.3f}%")
                with col3:
                    st.metric("Sigma Level", f"{sigma_level:.2f}")
                    if cpk >= 1.33: st.success("TỐT")
                    elif cpk >= 1.0: st.warning("CHẤP NHẬN")
                    else: st.error("CẢI THIỆN")

                # ========================
                # MSA (nếu có dữ liệu)
                # ========================
                if 'CheckerId' in df.columns and df['CheckerId'].nunique() > 1:
                    st.markdown("### MSA: Gage R&R (Simplified)")
                    # Giả sử có 3 người kiểm
                    grp = df.groupby('CheckerId')['Measure']
                    var_between = grp.var().mean()
                    var_within = df.groupby(['CheckerId', 'Trial'])['Measure'].var().mean() if 'Trial' in df.columns else 0
                    grr = np.sqrt(var_within + var_between)
                    pv = df['Measure'].std(ddof=1)
                    percent_grr = (grr / pv) * 100 if pv > 0 else 0
                    ndc = int(1.41 * (pv / grr)) if grr > 0 else 0
                    st.info(f"%GR&R: {percent_grr:.1f}% | ndc: {ndc}")

            else:
                st.warning("Không có dữ liệu.")
        else:
            st.error("Không tìm thấy ProductItem.")
    else:
        st.error("Không kết nối được Google Sheets.")
else:
    st.info("Nhập Sheet ID để bắt đầu.")
    with st.expander("Hướng dẫn"):
        st.markdown("- Tạo Service Account → `cred.json`\n- Share sheet với email service\n- Copy ID từ URL\n- Nhập ID + chọn item")