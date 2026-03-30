import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.font_manager as fm
import os
import io

# ================= 1. 云端中文适配 =================
# 检查并加载SimsunExtG.ttf字体
prop = None
font_path = os.path.join(os.path.dirname(__file__), 'SimsunExtG.ttf')
if os.path.exists(font_path):
    try:
        # 直接使用字体路径，不依赖字体管理器
        prop = fm.FontProperties(fname=font_path)
        # 设置字体
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        st.success("成功加载SimsunExtG.ttf字体")
    except Exception as e:
        st.warning(f"加载SimsunExtG.ttf字体失败: {str(e)}")
        # 使用默认字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning(f"未找到SimsunExtG.ttf字体文件，路径: {font_path}")
    # 使用默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

# 设置样式
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

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
        
        # 数据清洗
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 转换目标列为数值，并检查数据范围
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 显示数据预览，帮助调试
        st.subheader("数据预览")
        st.dataframe(df[[date_col] + target_cols].head(10))
        
        # 显示数据统计信息
        st.subheader("数据统计")
        st.write(f"数据量: {len(df)}")
        for col in target_cols:
            st.write(f"{col} 最小值: {df[col].min()}")
            st.write(f"{col} 最大值: {df[col].max()}")
            st.write(f"{col} 平均值: {df[col].mean()}")
        
        # 清洗数据
        df = df.dropna(subset=[date_col] + target_cols).sort_values(date_col)
        
        # 检查数据是否有效
        if len(df) == 0:
            st.error("数据清洗后无有效数据，请检查选择的列")
            st.stop()
        
        # ================= 3. 稳健统计算法 (IQR) =================
        def calculate_stats(series):
            raw_mean = series.mean()
            raw_cv = (series.std() / raw_mean) * 100 if raw_mean != 0 else 0
            
            # IQR 准则
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_b = max(0, q1 - 1.5 * iqr)
            upper_b = q3 + 1.5 * iqr
            
            clean_s = series[(series >= lower_b) & (series <= upper_b)]
            robust_mean = clean_s.mean()
            robust_cv = (clean_s.std() / robust_mean) * 100 if robust_mean != 0 else 0
            
            return {
                "raw_mean": raw_mean,
                "raw_cv": raw_cv,
                "robust_mean": robust_mean,
                "robust_cv": robust_cv,
                "bounds": [lower_b, upper_b],
                "outliers": len(series) - len(clean_s),
                "clean_series": clean_s
            }
        
        # 计算每个目标列的统计信息
        stats_results = {}
        for col in target_cols:
            stats_results[col] = calculate_stats(df[col])
        
        # ================= 4. 结果展示 =================
        # 添加图表设置功能
        st.sidebar.subheader('第三步：图表设置')
        chart_title = st.sidebar.text_input('趋势图标题', f'{"、".join(target_cols)}趋势分析')
        
        # X轴和Y轴名称编辑
        x_axis_label = st.sidebar.text_input('X轴名称', '日期')
        y1_axis_label = st.sidebar.text_input('左侧Y轴名称', '数值')
        
        # 双Y轴设置
        use_secondary_y = st.sidebar.checkbox('使用双Y轴', value=False)
        y2_axis_label = ''
        secondary_y_cols = []
        if use_secondary_y:
            y2_axis_label = st.sidebar.text_input('右侧Y轴名称', '数值')
            secondary_y_cols = st.sidebar.multiselect('右侧Y轴数据列', target_cols, default=[])
        
        # 数据标注名编辑
        st.sidebar.subheader('数据标注设置')
        series_labels = {}
        series_colors = {}
        # 预定义颜色列表
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, col in enumerate(target_cols):
            series_labels[col] = st.sidebar.text_input(f'{col}的标注名', col)
            # 为每个列添加颜色选择器
            default_color = default_colors[i % len(default_colors)]
            series_colors[col] = st.sidebar.color_picker(f'{col}的颜色', default_color)
        
        # 直方图设置
        st.sidebar.subheader('直方图设置')
        hist_title_template = st.sidebar.text_input('直方图标题模板', '{col}频率分布直方图')
        
        # 选择性显示元素
        show_histogram = st.sidebar.checkbox('显示频次分布', value=True)
        show_mean = st.sidebar.checkbox('显示平均值', value=True)
        show_median = st.sidebar.checkbox('显示中位数', value=True)
        show_gaussian = st.sidebar.checkbox('显示高斯拟合曲线', value=True)
        
        # 柱状体数值区间设置
        bin_width = st.sidebar.number_input('柱状体代表的数值区间', min_value=0.001, max_value=1.0, value=0.01, step=0.001, help='例如：设置为0.01表示每个柱体代表0.01mg/L的数据区间')
        
        # 顶层关键指标卡片 - 为每个目标列显示统计信息
        st.subheader("统计指标")
        for col in target_cols:
            stats_res = stats_results[col]
            st.write(f"**{col}**")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("原始均值", f"{stats_res['raw_mean']:.2f}")
            c2.metric("原始CV值", f"{stats_res['raw_cv']:.2f}%")
            c3.metric("稳健均值", f"{stats_res['robust_mean']:.2f}")
            c4.metric("稳健CV值", f"{stats_res['robust_cv']:.2f}%")
            c5.metric("异常点数量", f"{stats_res['outliers']}")
        
        # 图表区域
        tab1, tab2, tab3 = st.tabs(["📈 趋势折线图", "📊 频率分布直方图", "📋 异常值清单"])
        
        with tab1:
            fig_line, ax_line = plt.subplots(figsize=(12, 5))
            ax2 = None
            
            # 定义颜色列表
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # 如果使用双Y轴，创建第二个Y轴
            if use_secondary_y and secondary_y_cols:
                ax2 = ax_line.twinx()
                ax2.set_ylabel(y2_axis_label)
            
            # 为每个目标列绘制趋势线
            for i, col in enumerate(target_cols):
                color = series_colors[col]
                label = series_labels[col]
                
                # 决定使用哪个Y轴
                if use_secondary_y and col in secondary_y_cols:
                    if ax2:
                        ax2.plot(df[date_col], df[col], color=color, linewidth=1.2, marker='.', markersize=4, alpha=0.8, label=label)
                else:
                    ax_line.plot(df[date_col], df[col], color=color, linewidth=1.2, marker='.', markersize=4, alpha=0.8, label=label)
            
            # 设置标题和标签
            ax_line.set_title(chart_title, pad=15)
            ax_line.set_xlabel(x_axis_label)
            ax_line.set_ylabel(y1_axis_label)
            
            # 合并图例
            handles1, labels1 = ax_line.get_legend_handles_labels()
            handles2, labels2 = [], []
            if ax2:
                handles2, labels2 = ax2.get_legend_handles_labels()
            combined_handles = handles1 + handles2
            combined_labels = labels1 + labels2
            ax_line.legend(combined_handles, combined_labels, fontsize=10, loc='upper left')
            
            plt.xticks(rotation=45)
            ax_line.set_facecolor('white')
            
            # 调整布局
            plt.tight_layout()
            st.pyplot(fig_line)
            
            # 下载高清图表
            buf = io.BytesIO()
            fig_line.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="下载高清趋势图",
                data=buf,
                file_name=f"{chart_title}_趋势图.png",
                mime="image/png"
            )
        
        with tab2:
            # 为每个目标列创建单独的直方图
            for i, col in enumerate(target_cols):
                # 生成直方图标题
                hist_title = hist_title_template.replace('{col}', col)
                st.subheader(hist_title)
                
                fig_hist, ax_hist = plt.subplots(figsize=(12, 7))
                clean_data = stats_results[col]['clean_series']
                
                # 计算bins数量
                if len(clean_data) > 0:
                    data_range = clean_data.max() - clean_data.min()
                    bins = int(data_range / bin_width) if data_range > 0 else 20
                    bins = max(5, min(bins, 50))  # 限制bins数量在5-50之间
                else:
                    bins = 20
                
                # 直方图 - 使用实际频率而不是密度
                if show_histogram:
                    n, bins, patches = ax_hist.hist(clean_data, bins=bins, density=False, color='skyblue', edgecolor='white', alpha=0.7, label='频次分布')
                
                # 计算统计值
                mu, std = clean_data.mean(), clean_data.std()
                median = clean_data.median()
                
                # 高斯拟合 - 需要调整为与实际频率匹配
                if show_gaussian and len(clean_data) > 0:
                    x = np.linspace(clean_data.min(), clean_data.max(), 100)
                    # 计算高斯分布并缩放以匹配直方图高度
                    if show_histogram and len(n) > 0:
                        bin_width_actual = bins[1] - bins[0] if len(bins) > 1 else bin_width
                        p = stats.norm.pdf(x, mu, std) * len(clean_data) * bin_width_actual
                    else:
                        p = stats.norm.pdf(x, mu, std) * len(clean_data) * bin_width
                    ax_hist.plot(x, p, 'r-', linewidth=2, label='高斯拟合曲线')
                
                # 标注线
                if show_mean:
                    ax_hist.axvline(mu, color='red', linestyle='--', linewidth=1.5, label=f'平均值: {mu:.3f}')
                if show_median:
                    ax_hist.axvline(median, color='green', linestyle=':', linewidth=1.5, label=f'中位数: {median:.3f}')
                
                # 设置标题和标签
                ax_hist.set_title(hist_title, pad=15, fontsize=16)
                if show_histogram or show_gaussian or show_mean or show_median:
                    ax_hist.legend(fontsize=10)
                ax_hist.set_xlabel(f'{col} (mg/L)', fontsize=12)
                ax_hist.set_ylabel('出现频次', fontsize=12)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                
                # 调整布局
                plt.tight_layout()
                st.pyplot(fig_hist)
                
                # 下载高清图表
                buf = io.BytesIO()
                fig_hist.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label=f"下载{col}频率分布图",
                    data=buf,
                    file_name=f"{col}_频率分布图.png",
                    mime="image/png"
                )
            
        with tab3:
            for col in target_cols:
                st.subheader(f"{col}异常值清单")
                stats_res = stats_results[col]
                outliers_df = df[(df[col] < stats_res['bounds'][0]) | (df[col] > stats_res['bounds'][1])]
                st.write(f"正常运行区间设定为: **{stats_res['bounds'][0]:.2f} ~ {stats_res['bounds'][1]:.2f}**")
                st.dataframe(outliers_df[[date_col, col]])

    except Exception as e:
        st.error(f"解析出错：{str(e)}")
        st.info("提示：请确保选择了正确的列。")

else:
    st.info("💡 请在左侧上传 Excel 或 CSV 数据文件开始分析。")