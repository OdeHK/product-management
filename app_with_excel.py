import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from scipy import stats

# ========================
# CẤU HÌNH TRANG
# ========================
st.set_page_config(page_title="SPC & MSA Dashboard", layout="wide")
st.title("SPC & MSA Real-time Dashboard")
st.markdown("**Theo tiêu chuẩn AIAG, ISO 22514, Western Electric Rules**")

# ========================
# HÀM LOAD DỮ LIỆU TỪ EXCEL
# ========================
@st.cache_data
def load_excel_data(uploaded_file):
    try:
        # Đọc file Excel
        df = pd.read_excel(uploaded_file, sheet_name=0)
        
        # Chuẩn hóa tên cột (loại bỏ khoảng trắng)
        df.columns = [col.strip() for col in df.columns]
        
        # Chuẩn hóa Measure
        if 'Measure' in df.columns:
            df['Measure'] = df['Measure'].astype(str).str.strip()
            df['Measure'] = pd.to_numeric(df['Measure'], errors='coerce')
        
        # Chuẩn hóa các cột khác
        for col in ['Date', 'ProductItem', 'CheckerId', 'WorkerId', 'Trial']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Lọc dữ liệu hợp lệ
        df = df[df['Measure'].notna() & (df['ProductItem'] != '') & (df['ProductItem'] != 'nan')].reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file Excel: {str(e)}")
        return pd.DataFrame()

# ========================
# HÀM TÍNH CONTROL LIMITS (I-MR) - CHUẨN AIAG
# ========================
def calculate_imr_limits(values):
    if len(values) < 2:
        return 0, 0, 0, 0, 0, 0, 0
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
# SIDEBAR - UPLOAD FILE
# ========================
st.sidebar.header("Tải file dữ liệu")
uploaded_file = st.sidebar.file_uploader(
    "Chọn file Excel (.xlsx, .xls)", 
    type=['xlsx', 'xls'],
    help="File Excel cần có các cột: Date, ProductItem, Measure, CheckerId (optional), WorkerId (optional), Trial (optional)"
)

st.sidebar.markdown("---")
st.sidebar.header("Cấu hình giới hạn")
usl = st.sidebar.number_input("USL (Upper Spec Limit)", value=10.0, format="%.4f")
lsl = st.sidebar.number_input("LSL (Lower Spec Limit)", value=0.0, format="%.4f")
target = st.sidebar.number_input("Target", value=5.0, format="%.4f")

# ========================
# XỬ LÝ DỮ LIỆU
# ========================
if uploaded_file is not None:
    df_all = load_excel_data(uploaded_file)
    
    if not df_all.empty and 'ProductItem' in df_all.columns:
        items = sorted(df_all['ProductItem'].unique())
        selected = st.sidebar.selectbox("Chọn Product Item", items)
        df = df_all[df_all['ProductItem'] == selected].copy()
        
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
            # HISTOGRAM + NORMAL
            # ========================
            st.markdown("### Histogram & Normality")
            col1, col2 = st.columns(2)

            # Tính bins an toàn
            n_bins = max(5, min(20, int(np.sqrt(n) if n > 0 else 10)))
            data_min, data_max = values.min(), values.max()

            # Xác định phạm vi
            if n <= 1:
                center = values[0]
                plot_min = center * 0.9
                plot_max = center * 1.1
            else:
                p1, p99 = np.percentile(values, [1, 99])
                iqr = p99 - p1
                margin = max(iqr * 0.4, (values.max() - values.min()) * 0.1)
                plot_min = p1 - margin
                plot_max = p99 + margin
                if plot_min < 0 and values.min() >= 0:
                    plot_min = 0

            # Đảm bảo plot_min < plot_max
            if plot_min >= plot_max:
                margin = 0.1
                plot_min = data_min - margin
                plot_max = data_max + margin

            bin_edges = np.linspace(plot_min, plot_max, n_bins + 1)
            bin_width = (plot_max - plot_min) / n_bins

            # HISTOGRAM
            with col1:
                st.markdown("**Histogram (Tự động zoom vào dữ liệu)**")
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

                # Đường cong chuẩn
                x_curve = np.linspace(plot_min, plot_max, 300)
                pdf_scaled = stats.norm.pdf(x_curve, mean, sigma) * n * bin_width
                fig_hist.add_trace(go.Scatter(
                    x=x_curve, y=pdf_scaled,
                    mode='lines', line=dict(color='red', width=3),
                    name='Phân phối chuẩn'
                ))

                # Hiển thị LSL/USL
                def add_spec_line(val, label, color="black"):
                    if val is not None and not np.isnan(val):
                        if plot_min <= val <= plot_max:
                            fig_hist.add_vline(x=val, line_dash="dash", line_color=color,
                                            annotation_text=label, annotation_position="top")

                add_spec_line(lsl, "LSL", "green")
                add_spec_line(usl, "USL", "red")
                if target and not np.isnan(target):
                    add_spec_line(target, "Target", "orange")

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
                st.markdown("**Q-Q Plot (Normal Probability)**")
                
                # Tính quantiles
                sorted_val = np.sort(values)
                p = (np.arange(1, n+1) - 0.5) / n
                theoretical_quantiles = stats.norm.ppf(p, 0, 1)  # Standard normal
                
                # Chuẩn hóa dữ liệu
                standardized_data = (sorted_val - mean) / sigma
                
                # Tính đường fit và R²
                slope, intercept = np.polyfit(theoretical_quantiles, standardized_data, 1)
                fitted_line = slope * theoretical_quantiles + intercept
                
                # Tính R² cho correlation
                correlation = np.corrcoef(theoretical_quantiles, standardized_data)[0, 1]
                r_squared = correlation ** 2
                
                # Tính confidence band 95%
                se = np.sqrt(np.sum((standardized_data - fitted_line)**2) / (n - 2))
                # Critical value for 95% confidence
                t_val = stats.t.ppf(0.975, n - 2)
                
                # Confidence interval
                x_mean = np.mean(theoretical_quantiles)
                sxx = np.sum((theoretical_quantiles - x_mean)**2)
                margin = t_val * se * np.sqrt(1/n + (theoretical_quantiles - x_mean)**2 / sxx)
                upper_band = fitted_line + margin
                lower_band = fitted_line - margin
                
                # Vẽ Q-Q Plot
                fig_qq = go.Figure()
                
                # Confidence bands (vùng tin cậy)
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=upper_band,
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'),
                    name='95% CI',
                    showlegend=False
                ))
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=lower_band,
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    name='95% CI',
                    showlegend=False
                ))
                
                # Đường fit chính
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=fitted_line,
                    mode='lines',
                    line=dict(color='gray', width=2),
                    name='Fit Line',
                    showlegend=False
                ))
                
                # Data points
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=standardized_data,
                    mode='markers',
                    marker=dict(color='blue', size=8),
                    name='Data',
                    showlegend=False
                ))
                
                # Thêm R² annotation
                fig_qq.add_annotation(
                    x=0.95, y=0.05,
                    xref='paper', yref='paper',
                    text=f'R² = {r_squared:.3f}',
                    showarrow=False,
                    font=dict(size=14),
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1
                )
                
                fig_qq.update_layout(
                    height=450,
                    xaxis_title="Theoretical quantiles",
                    yaxis_title="Ordered quantiles",
                    title="Q-Q Plot",
                    plot_bgcolor="white",
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="lightgray",
                        zeroline=True,
                        zerolinecolor='black',
                        zerolinewidth=1
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="lightgray",
                        zeroline=True,
                        zerolinecolor='black',
                        zerolinewidth=1
                    ),
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                st.plotly_chart(fig_qq, use_container_width=True)

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
                grp = df.groupby('CheckerId')['Measure']
                var_between = grp.var().mean()
                var_within = df.groupby(['CheckerId', 'Trial'])['Measure'].var().mean() if 'Trial' in df.columns else 0
                grr = np.sqrt(var_within + var_between)
                pv = df['Measure'].std(ddof=1)
                percent_grr = (grr / pv) * 100 if pv > 0 else 0
                ndc = int(1.41 * (pv / grr)) if grr > 0 else 0
                st.info(f"%GR&R: {percent_grr:.1f}% | ndc: {ndc}")

        else:
            st.warning("Không có dữ liệu cho Product Item được chọn.")
    else:
        st.error("Không tìm thấy cột 'ProductItem' trong file Excel.")
else:
    st.info("📤 Vui lòng tải file Excel lên để bắt đầu phân tích")
    with st.expander("ℹ️ Hướng dẫn sử dụng"):
        st.markdown("""
        ### Format file Excel yêu cầu:
        
        File Excel cần có các cột sau (sheet đầu tiên):
        - **Date**: Ngày đo (bắt buộc)
        - **ProductItem**: Tên sản phẩm/mã sản phẩm (bắt buộc)
        - **Measure**: Giá trị đo (bắt buộc)
        - **CheckerId**: Mã người kiểm tra (tùy chọn - cho MSA)
        - **WorkerId**: Mã người thực hiện (tùy chọn)
        - **Trial**: Số lần đo lặp (tùy chọn - cho MSA)
        
        ### Các bước sử dụng:
        1. Tải file Excel lên
        2. Chọn Product Item cần phân tích
        3. Điều chỉnh USL, LSL, Target theo yêu cầu
        4. Xem các biểu đồ và phân tích

        """)

