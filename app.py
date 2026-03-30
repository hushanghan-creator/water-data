import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.font_manager as fm
import matplotlib
import os
import io

# ================= 1. 云端中文适配 (核心修复) =================
# 1. 明确指定字体文件路径 (确保文件名大小写与 GitHub 仓库完全一致)
font_file = 'myfont.ttc'
font_loaded = False
font_prop = None

# 2. 强力加载字体
if os.path.exists(font_file):
    try:
        # 注册字体
        fm.fontManager.addfont(font_file)
        # 获取字体属性
        font_prop = fm.FontProperties(fname=font_file)
        font_name = font_prop.get_name()
        
        # 强制设置全局字体
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        
        # Streamlit 云端强制补丁
        matplotlib.rc('font', family=font_name)
        
        st.success(f"✅ 成功加载字体: {font_name}")
        font_loaded = True
    except Exception as e:
        st.warning(f"加载字体失败: {str(e)}")
else:
    st.error(f"❌ 未找到字体文件: {font_file}，请确保已上传至 GitHub 根目录。")

# 设置默认样式
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# UI 汉化 CSS 补丁 (可选，解决 Browse files 按钮英文问题)
st.markdown("""
    <style>
    .stFileUploader section button span:last-child { display: none; }
    .stFileUploader section button::after { content: "点击选择文件"; }
    .stFileUploader section > div:first-child { display: none; }
    .stFileUploader section::before { content: "将数据文件拖拽到此处"; display: block; padding: 10px; color: #555; }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="水厂数据分析工具", layout="wide")
st.title('📊 水厂数据分析系统')

# ================= 2. 侧边栏配置 =================
st.sidebar.header('📂 操作配置')
uploaded_file = st.sidebar.file_uploader('第一步：上传数据', type=['csv', 'xlsx'])

if uploaded_file:
    # 智能表头读取
    try:
        # 读取前5行
        if uploaded_file.name.endswith('.csv'):
            preview_df = pd.read_csv(uploaded_file, nrows=5)
        else:
            preview_df = pd.read_excel(uploaded_file, nrows=5)
        
        # 关键字列表
        keywords = ['时间', '日期', '流量', '水量', '磷', 'TP', '浓度', '出水']
        header_row = 0
        
        # 寻找包含关键字的行作为header
        for i, row in preview_df.iterrows():
            row_str = ''.join(str(cell) for cell in row)
            if any(keyword in row_str for keyword in keywords):
                header_row = i
                break
        
        # 重新读取数据
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=header_row)
        else:
            df = pd.read_excel(uploaded_file, header=header_row)
        
        st.sidebar.success("文件读取成功！")
        
        # 极简操作
        st.sidebar.subheader('第二步：选择分析列')
        all_cols = df.columns.tolist()
        date_col = st.sidebar.selectbox('时间/日期列', all_cols)
        
        # 识别数值列
        numeric_cols = []
        for col in all_cols:
            if col != date_col:
                try:
                    # 尝试转换为数值
                    pd.to_numeric(df[col], errors='coerce')
                    numeric_cols.append(col)
                except:
                    pass
        
        # 确保目标分析列只能选择数值列
        if numeric_cols:
            # 允许选择多个目标分析列
            target_cols = st.sidebar.multiselect('目标分析列(数值)', numeric_cols, default=numeric_cols)
        else:
            st.error("未找到数值列，请检查数据文件")
            st.stop()
        
        # 数据预处理
        # 转换时间列
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        except:
            st.warning("时间列转换失败，使用原始值")
        
        # 转换数值列并处理缺失值
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除时间列或数值列为空的行
        df = df.dropna(subset=[date_col] + target_cols)
        
        # 按时间排序
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.sort_values(by=date_col)
        
        # ================= 3. 计算统计指标 =================
        stats_results = {}
        for col in target_cols:
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            robust_mean = data.median()
            robust_cv = (IQR / 1.349) / robust_mean if robust_mean > 0 else 0
            original_mean = data.mean()
            original_std = data.std()
            original_cv = original_std / original_mean if original_mean > 0 else 0
            
            stats_results[col] = {
                'data': data,
                'clean_series': data[(data >= lower_bound) & (data <= upper_bound)],
                'original_mean': original_mean,
                'original_std': original_std,
                'original_cv': original_cv,
                'robust_mean': robust_mean,
                'robust_cv': robust_cv,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'outliers': outliers,
                'outlier_count': len(outliers)
            }
        
        # 图表设置
        st.sidebar.subheader('第三步：图表设置')
        chart_title = st.sidebar.text_input('趋势图标题', '水质数据趋势分析')
        x_axis_label = st.sidebar.text_input('X轴名称', '日期')
        y_axis_label = st.sidebar.text_input('Y轴名称', '数值')
        
        use_secondary_y = st.sidebar.checkbox('使用双Y轴', value=False)
        if use_secondary_y:
            y2_axis_label = st.sidebar.text_input('右侧Y轴名称', '数值')
            secondary_y_cols = st.sidebar.multiselect('右侧Y轴数据列', target_cols, default=[])
        
        st.sidebar.subheader('数据标注设置')
        series_labels = {}
        series_colors = {}
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, col in enumerate(target_cols):
            series_labels[col] = st.sidebar.text_input(f'{col}的标注名', col)
            default_color = default_colors[i % len(default_colors)]
            series_colors[col] = st.sidebar.color_picker(f'{col}的颜色', default_color)
        
        st.sidebar.subheader('直方图设置')
        hist_title_template = st.sidebar.text_input('直方图标题模板', '{col}频率分布直方图')
        show_histogram = st.sidebar.checkbox('显示频次分布', value=True)
        show_mean = st.sidebar.checkbox('显示平均值', value=True)
        show_median = st.sidebar.checkbox('显示中位数', value=True)
        show_gaussian = st.sidebar.checkbox('显示高斯拟合曲线', value=True)
        bin_width = st.sidebar.number_input('柱状体代表的数值区间', min_value=0.001, max_value=1.0, value=0.01, step=0.001)
        
        st.subheader("统计指标")
        for col in target_cols:
            with st.expander(f"{col}统计指标"):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("原始均值", f"{stats_results[col]['original_mean']:.2f}")
                with col2: st.metric("原始CV", f"{stats_results[col]['original_cv']:.2%}")
                with col3: st.metric("稳健均值", f"{stats_results[col]['robust_mean']:.2f}")
                with col4: st.metric("稳健CV", f"{stats_results[col]['robust_cv']:.2%}")
                with col5: st.metric("异常值数量", stats_results[col]['outlier_count'])
        
        # ================= 4. 绘图显示 =================
        tab1, tab2, tab3 = st.tabs(["趋势折线图", "频率分布直方图", "异常值清单"])
        
        with tab1:
            st.subheader(chart_title)
            fig_line, ax_line = plt.subplots(figsize=(12, 7))
            ax2 = None
            if use_secondary_y and secondary_y_cols:
                ax2 = ax_line.twinx()
                ax2.set_ylabel(y2_axis_label, fontproperties=font_prop) # 显式设置
            
            for i, col in enumerate(target_cols):
                color = series_colors[col]
                label = series_labels[col]
                if use_secondary_y and col in secondary_y_cols:
                    if ax2: ax2.plot(df[date_col], df[col], color=color, linewidth=1.2, marker='.', markersize=4, alpha=0.8, label=label)
                else:
                    ax_line.plot(df[date_col], df[col], color=color, linewidth=1.2, marker='.', markersize=4, alpha=0.8, label=label)
            
            # 使用 font_prop 强行设置文本，防止全局失效
            ax_line.set_title(chart_title, pad=15, fontproperties=font_prop)
            ax_line.set_xlabel(x_axis_label, fontproperties=font_prop)
            ax_line.set_ylabel(y_axis_label, fontproperties=font_prop)
            
            if use_secondary_y and secondary_y_cols:
                lines1, labels1 = ax_line.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax_line.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop=font_prop)
            else:
                ax_line.legend(prop=font_prop)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_line)
            
            buf = io.BytesIO()
            fig_line.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(label="下载趋势图", data=buf, file_name=f"{chart_title}.png", mime="image/png")
        
        with tab2:
            for i, col in enumerate(target_cols):
                hist_title = hist_title_template.replace('{col}', col)
                st.subheader(hist_title)
                fig_hist, ax_hist = plt.subplots(figsize=(12, 7))
                clean_data = stats_results[col]['clean_series']
                
                if len(clean_data) > 0:
                    data_range = clean_data.max() - clean_data.min()
                    bins = int(data_range / bin_width) if data_range > 0 else 20
                    bins = max(5, min(bins, 50))
                else: bins = 20
                
                if show_histogram:
                    n, bins_edges, patches = ax_hist.hist(clean_data, bins=bins, density=False, color='skyblue', edgecolor='white', alpha=0.7, label='频次分布')
                
                mu, std = clean_data.mean(), clean_data.std()
                median = clean_data.median()
                
                if show_gaussian and len(clean_data) > 0:
                    x = np.linspace(clean_data.min(), clean_data.max(), 100)
                    bin_width_actual = bins_edges[1] - bins_edges[0] if len(bins_edges) > 1 else bin_width
                    p = stats.norm.pdf(x, mu, std) * len(clean_data) * bin_width_actual
                    ax_hist.plot(x, p, 'r-', linewidth=2, label='高斯拟合曲线')
                
                if show_mean: ax_hist.axvline(mu, color='red', linestyle='--', linewidth=1.5, label=f'平均值: {mu:.3f}')
                if show_median: ax_hist.axvline(median, color='green', linestyle=':', linewidth=1.5, label=f'中位数: {median:.3f}')
                
                ax_hist.set_title(hist_title, pad=15, fontsize=16, fontproperties=font_prop)
                ax_hist.legend(fontsize=10, prop=font_prop)
                ax_hist.set_xlabel(f'{col} (mg/L)', fontsize=12, fontproperties=font_prop)
                ax_hist.set_ylabel('出现频次', fontsize=12, fontproperties=font_prop)
                
                plt.tight_layout()
                st.pyplot(fig_hist)
        
        with tab3:
            for col in target_cols:
                st.subheader(f"{col}异常值清单")
                outliers_data = stats_results[col]['outliers']
                if len(outliers_data) > 0:
                    outlier_rows = df[df[col].isin(outliers_data)]
                    st.dataframe(outlier_rows[[date_col, col]])
                else: st.info(f"{col}无异常值")
        
    except Exception as e:
        st.error(f"数据处理错误: {str(e)}")
else:
    st.info("💡 请在左侧上传 Excel 或 CSV 数据文件开始分析。")