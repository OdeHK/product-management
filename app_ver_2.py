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
            df['Measure'] = df['Measure'].astype(str).str.strip()\
                .str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
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
    d2 = 1.128  # AIAG constant for n=2 (I-MR chart)
    sigma = mr_bar / d2
    
    # Theo công thức trong tài liệu:
    # I chart: UCL = X̄ + 2.66*MR̄ ; LCL = X̄ - 2.66*MR̄
    ucl_i = mean + 2.66 * mr_bar
    lcl_i = max(0, mean - 2.66 * mr_bar)
    
    # MR chart: UCL = 3.266*MR̄ ; LCL = 0
    ucl_mr = 3.266 * mr_bar
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
            # HISTOGRAM + NORMAL - THEO CHUẨN AIAG
            # ========================
            st.markdown("### Histogram & Normality")
            col1, col2 = st.columns(2)

            # Tính số bins theo bảng AIAG (trang 2 tài liệu)
            if n < 25:
                st.warning("⚠️ Dữ liệu < 25 mẫu, không nên vẽ histogram. Khuyến nghị dùng dot plot hoặc run chart.")
                n_bins = 6
            elif n < 50:
                n_bins = 6
            elif n < 100:
                n_bins = 8
            elif n < 250:
                n_bins = 10  # Chuẩn AIAG
            elif n < 500:
                n_bins = 12
            elif n < 1000:
                n_bins = 14
            else:
                n_bins = min(20, max(16, int(np.sqrt(n))))
            
            # Bin width theo công thức: (USL - LSL) / số bin
            if usl > lsl:
                bin_width_spec = (usl - lsl) / n_bins
                plot_min = lsl
                plot_max = usl
            else:
                # Nếu không có spec limits, dùng data range
                data_min, data_max = values.min(), values.max()
                data_range = data_max - data_min
                plot_min = data_min - 0.1 * data_range
                plot_max = data_max + 0.1 * data_range
                bin_width_spec = (plot_max - plot_min) / n_bins

            bin_edges = np.linspace(plot_min, plot_max, n_bins + 1)

            # HISTOGRAM
            with col1:
                st.markdown("**Histogram với Normal Curve**")
                fig_hist = go.Figure()
                hist_counts, _ = np.histogram(values, bins=bin_edges)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                fig_hist.add_trace(go.Bar(
                    x=bin_centers,
                    y=hist_counts,
                    name="Data",
                    marker_color="#1f77b4",
                    width=bin_width_spec * 0.9,
                    hovertemplate="Giá trị: %{x:.3f}<br>Tần suất: %{y}<extra></extra>"
                ))

                # Đường cong chuẩn theo công thức trong tài liệu:
                # y(x) = (1/σ√(2π)) * e^(-(x-μ)²/(2σ²))
                x_curve = np.linspace(plot_min, plot_max, 300)
                # Scale để khớp với histogram (nhân với tổng diện tích)
                total_area = n * bin_width_spec
                pdf_scaled = stats.norm.pdf(x_curve, mean, sigma) * total_area
                
                fig_hist.add_trace(go.Scatter(
                    x=x_curve, y=pdf_scaled,
                    mode='lines', 
                    line=dict(color='red', width=3),
                    name='Normal Curve'
                ))

                # Hiển thị LSL/USL/Target
                def add_spec_line(val, label, color="black"):
                    if val is not None and not np.isnan(val):
                        if plot_min <= val <= plot_max:
                            fig_hist.add_vline(
                                x=val, 
                                line_dash="dash", 
                                line_color=color,
                                annotation_text=label, 
                                annotation_position="top"
                            )

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
                        showgrid=True, 
                        gridcolor="lightgray",
                        tickformat=".3f"
                    ),
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor="lightgray",
                        zeroline=False
                    ),
                    margin=dict(l=50, r=50, t=50, b=50),
                    hovermode="x unified"
                )

                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Hiển thị thông tin histogram theo AIAG
                st.caption(f"📊 Bins: {n_bins} (AIAG standard for N={n}) | Bin width: {bin_width_spec:.4f}")
                        
            with col2:
                st.markdown("**Normal Probability Plot**")
                
                # Sắp xếp dữ liệu
                sorted_val = np.sort(values)
                
                # Tính plotting position theo công thức Blom: (i - 3/8) / (n + 1/4)
                # Đây là công thức chuẩn cho normal probability plot
                plotting_positions = (np.arange(1, n+1) - 0.375) / (n + 0.25)
                percentiles = plotting_positions * 100
                
                # Tính theoretical quantiles từ phân phối chuẩn
                z_scores = stats.norm.ppf(plotting_positions)
                theoretical_values = mean + sigma * z_scores
                
                # Tính R² (correlation coefficient)
                correlation = np.corrcoef(sorted_val, theoretical_values)[0, 1]
                r_squared = correlation ** 2
                
                # ===== CONFIDENCE BAND - Phương pháp chính xác =====
                # Sử dụng Lilliefors confidence envelope
                # Công thức: P_upper/lower = P ± c(α,n) × sqrt(P × (1-P))
                
                alpha = 0.05  # 95% confidence
                # Critical value cho confidence band (theo bảng Lilliefors)
                # Xấp xỉ: c ≈ sqrt(-ln(α/2) / 2) / sqrt(n)
                c_alpha = np.sqrt(-np.log(alpha/2) / 2)
                
                # Tính confidence band theo từng điểm
                # Band rộng hơn ở đuôi phân phối (P gần 0 hoặc 1)
                lower_band = []
                upper_band = []
                
                for i, p in enumerate(plotting_positions):
                    # Standard error tại điểm p
                    se = np.sqrt(p * (1 - p) / n)
                    
                    # Margin với hệ số điều chỉnh
                    # Sử dụng hệ số 2.5 để band rộng hơn và chứa hầu hết điểm
                    margin = 2.5 * c_alpha * se
                    
                    # Tính percentile boundaries
                    p_lower = max(0.0001, p - margin)  # Tránh 0
                    p_upper = min(0.9999, p + margin)  # Tránh 1
                    
                    # Chuyển về giá trị thực tế
                    z_lower = stats.norm.ppf(p_lower)
                    z_upper = stats.norm.ppf(p_upper)
                    
                    lower_band.append((p_lower * 100))
                    upper_band.append((p_upper * 100))
                
                lower_band = np.array(lower_band)
                upper_band = np.array(upper_band)
                
                # ===== VẼ BIỂU ĐỒ =====
                fig_prob = go.Figure()
                
                # Confidence bands (màu đỏ)
                fig_prob.add_trace(go.Scatter(
                    x=sorted_val,
                    y=upper_band,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='95% CI',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig_prob.add_trace(go.Scatter(
                    x=sorted_val,
                    y=lower_band,
                    mode='lines',
                    line=dict(color='red', width=2),
                    fill='tonexty',
                    fillcolor='rgba(255, 200, 200, 0.3)',
                    name='95% CI',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Đường lý thuyết (đen)
                fig_prob.add_trace(go.Scatter(
                    x=theoretical_values,
                    y=percentiles,
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Normal Line',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Data points (dấu + xanh lam)
                fig_prob.add_trace(go.Scatter(
                    x=sorted_val,
                    y=percentiles,
                    mode='markers',
                    marker=dict(
                        color='cyan',
                        size=10,
                        symbol='cross',
                        line=dict(width=2, color='cyan')
                    ),
                    name='Data',
                    showlegend=False,
                    hovertemplate='Value: %{x:.4f}<br>Percentile: %{y:.1f}%<extra></extra>'
                ))
                
                # Layout
                fig_prob.update_layout(
                    height=450,
                    xaxis_title="测试结果 Test Result",
                    yaxis_title="Percent",
                    plot_bgcolor="white",
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="lightgray",
                        zeroline=False
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="lightgray",
                        zeroline=False,
                        tickvals=[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99],
                        range=[0, 100]
                    ),
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Hiển thị thống kê
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("R² (Goodness of Fit)", f"{r_squared:.4f}")
                with col_b:
                    # Anderson-Darling test
                    ad_result = stats.anderson(values, dist='norm')
                    ad_stat = ad_result.statistic
                    critical_5pct = ad_result.critical_values[2]
                    is_normal = "✓ Normal" if ad_stat < critical_5pct else "✗ Not Normal"
                    st.metric("Anderson-Darling", f"{ad_stat:.3f} ({is_normal})")

            # ========================
            # PROCESS CAPABILITY - THEO CÔNG THỨC TÀI LIỆU
            # ========================
            st.markdown("### Process Capability (Cp, Cpk, PPM, Sigma)")
            
            # Cp = (USL - LSL) / (6σ)
            cp = (usl - lsl) / (6 * sigma) if sigma > 0 else 0
            
            # Cpk = min((USL - X̄)/(3σ), (X̄ - LSL)/(3σ))
            cpu = (usl - mean) / (3 * sigma) if sigma > 0 else 0
            cpl = (mean - lsl) / (3 * sigma) if sigma > 0 else 0
            cpk = min(cpu, cpl)
            
            # PPM = (1 - Yield) * 10^6
            # Tính số lượng ngoài spec
            out_of_spec = len(values[values > usl]) + len(values[values < lsl])
            yield_actual = (n - out_of_spec) / n
            ppm = (1 - yield_actual) * 1e6
            yield_pct = yield_actual * 100
            
            # Sigma Level: σ = Φ^(-1)(Yield)
            # Không cộng 1.5 shift (dùng công thức chuẩn)
            if yield_actual > 0.5:
                sigma_level = stats.norm.ppf(yield_actual)
            else:
                sigma_level = 0
            
            # Six-Sigma convention: σ = Φ^(-1)(Yield) + 1.5
            sigma_level_6s = sigma_level + 1.5 if sigma_level > 0 else 0
            
            # Cpm (nếu có Target)
            if target and not np.isnan(target) and sigma > 0:
                cpm = (usl - lsl) / (6 * np.sqrt(sigma**2 + (mean - target)**2))
            else:
                cpm = None

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cp (Potential)", f"{cp:.3f}")
                st.metric("Cpk (Actual)", f"{cpk:.3f}")
                if cpm is not None:
                    st.metric("Cpm (w/ Target)", f"{cpm:.3f}")
            with col2:
                st.metric("PPM (Defects)", f"{ppm:,.1f}")
                st.metric("Yield", f"{yield_pct:.2f}%")
            with col3:
                st.metric("Sigma Level", f"{sigma_level:.2f}σ")
                st.metric("6σ Convention", f"{sigma_level_6s:.2f}σ")
                
            # Đánh giá theo AIAG
            if cpk >= 1.33: 
                st.success("✓ TỐT - Quá trình có năng lực")
            elif cpk >= 1.0: 
                st.warning("⚠ CHẤP NHẬN - Cần theo dõi")
            else: 
                st.error("✗ CẢI THIỆN - Quá trình không đạt yêu cầu")

            # ========================
            # MSA: Gage R&R - THEO CÔNG THỨC TÀI LIỆU
            # ========================
            if 'CheckerId' in df.columns and df['CheckerId'].nunique() > 1:
                st.markdown("### MSA: Gage R&R Analysis")
                
                # Công thức theo tài liệu trang 3:
                # σ²_total = σ²_part + σ²_repeat + σ²_reprod
                # %GRR = √(σ²_repeat + σ²_reprod) / σ_total × 100%
                # ndc = 1.41 × (σ_part / σ_gauge)
                
                # Tính variance components
                grp = df.groupby('CheckerId')['Measure']
                var_reprod = grp.mean().var()  # Reproducibility variance (giữa người đo)
                
                if 'Trial' in df.columns:
                    var_repeat = df.groupby(['CheckerId', 'Trial'])['Measure'].var().mean()  # Repeatability
                else:
                    var_repeat = grp.var().mean()  # Repeatability (trong mỗi người đo)
                
                var_part = df.groupby('ProductItem')['Measure'].var().mean() if 'ProductItem' in df.columns else df['Measure'].var()
                
                # Total variation
                sigma_total = df['Measure'].std(ddof=1)
                
                # Gage R&R
                sigma_grr = np.sqrt(var_repeat + var_reprod)
                percent_grr = (sigma_grr / sigma_total) * 100 if sigma_total > 0 else 0
                
                # Number of Distinct Categories
                sigma_part = np.sqrt(var_part)
                ndc = int(1.41 * (sigma_part / sigma_grr)) if sigma_grr > 0 else 0
                
                # Hiển thị kết quả
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("%GR&R", f"{percent_grr:.1f}%")
                    if percent_grr < 10:
                        st.success("✓ Hệ thống đo TỐT")
                    elif percent_grr < 30:
                        st.warning("⚠ Chấp nhận có điều kiện")
                    else:
                        st.error("✗ Hệ thống đo KHÔNG ĐẠT")
                
                with col2:
                    st.metric("ndc", f"{ndc}")
                    if ndc >= 5:
                        st.success("✓ Đủ phân biệt")
                    else:
                        st.warning("⚠ Cần cải thiện")
                
                with col3:
                    st.metric("Repeatability", f"{np.sqrt(var_repeat):.4f}")
                    st.metric("Reproducibility", f"{np.sqrt(var_reprod):.4f}")
                
                st.caption("📌 AIAG standard: %GR&R < 10% (Good), < 30% (Acceptable), ≥30% (Unacceptable) | ndc ≥ 5 (Adequate)")

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
