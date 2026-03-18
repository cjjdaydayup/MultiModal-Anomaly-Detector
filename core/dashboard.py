import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def render_dashboard(record_dir="records"):
    """
    渲染工业数据大屏
    """
    st.header("📈 生产数据监控看板 (Dashboard)")

    record_file = os.path.join(record_dir, "detection_history.json")

    if not os.path.exists(record_file):
        st.info("📊 暂无检测记录。请先在检测面板运行几次检测后，再来查看数据分析。")
        return

    try:
        with open(record_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"读取数据失败: {e}")
        return

    if not data:
        st.info("📊 暂无检测记录。")
        return

    # 转换为 Pandas DataFrame 进行复杂的数据分析
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # ---- 1. 核心指标卡片 ----
    total_inspected = len(df)
    total_defects = len(df[df['result'] == 'Defective'])
    total_normal = total_inspected - total_defects
    defect_rate = (total_defects / total_inspected) * 100 if total_inspected > 0 else 0

    st.subheader("核心生产指标 (KPI)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("累计检测总数", f"{total_inspected} 件")
    col2.metric("发现良品", f"{total_normal} 件")
    col3.metric("发现不良品", f"{total_defects} 件", delta=f"{defect_rate:.2f}% 不良率", delta_color="inverse")
    col4.metric("平均缺陷置信度", f"{df['confidence_score'].mean():.4f}")

    st.divider()

    # ---- 2. 图表分析区 ----
    st.subheader("多维数据分析")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### 📌 各产品检测数量分布")
        product_counts = df['product'].value_counts()
        st.bar_chart(product_counts)

    with chart_col2:
        st.markdown("##### 📌 产品良率占比")
        # 统计结果类型
        result_counts = df['result'].value_counts()

        # 使用 matplotlib 画饼图
        fig, ax = plt.subplots(figsize=(5, 4))
        # 设定颜色：正常为绿色，缺陷为红色
        colors = ['#28a745' if idx == 'Normal' else '#dc3545' for idx in result_counts.index]
        ax.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        # 将 matplotlib 图表嵌入 streamlit
        st.pyplot(fig)

    # ---- 3. 原始数据表格 ----
    st.divider()
    st.subheader("📝 近期检测日志清单")
    # 倒序显示最近的 10 条记录
    st.dataframe(df.sort_values(by="timestamp", ascending=False).head(10), use_container_width=True)