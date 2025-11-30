import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 页面基础配置
# ==========================================
st.set_page_config(
    page_title="西安机场霜预测系统",
    page_icon="✈️",
    layout="centered"
)

# 标题和简介
st.title("✈️ 西安机场霜预测系统")
st.markdown("基于随机森林算法 | 阈值标准：0.5")
st.markdown("---")


# ==========================================
# 1. 加载模型 (核心步骤)
# ==========================================
@st.cache_resource
def load_model():
    # 这里的路径相对简单，只要和脚本放在一起即可
    model_path = "frost_prediction_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None


try:
    rf_model = load_model()
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

if rf_model is None:
    st.error("⚠️ 找不到模型文件！请将 frost_prediction_model.pkl 放入当前文件夹。")
    st.stop()

# ==========================================
# 2. 功能选择区
# ==========================================
tab1, tab2 = st.tabs(["📝 单样本预测", "📂 批量预测 (Excel)"])

# --- 功能 1：单样本预测 ---
with tab1:
    st.subheader("请输入气象数据")

    col1, col2 = st.columns(2)
    with col1:
        dewpoint = st.number_input("露点温度 (°C)", value=-5.0, step=0.1)
        temp = st.number_input("气温 (°C)", value=2.0, step=0.1)
        temp_dew_diff = st.number_input("温度露点差 (°C)", value=7.0, step=0.1)
    with col2:
        humidity = st.number_input("相对湿度 (%)", value=60.0, step=1.0)
        cloud = st.number_input("云量 (0-8)", value=2.0, step=1.0, min_value=0.0, max_value=8.0)

    if st.button("开始预测", type="primary"):
        # 准备数据
        features = np.array([[dewpoint, temp, temp_dew_diff, humidity, cloud]])

        # 预测
        probs = rf_model.predict_proba(features)[0]
        frost_prob = probs[1]

        # 判定逻辑 (严格 0.5)
        if frost_prob >= 0.5:
            result_text = "有霜"
            result_color = "red"
            icon = "❄️"
        else:
            result_text = "无霜"
            result_color = "green"
            icon = "☀️"

        # 展示结果
        st.markdown("### 预测结果")
        st.markdown(f":{result_color}[## {icon} {result_text}]")

        # 展示概率条
        st.progress(frost_prob, text=f"结霜概率: {frost_prob:.2%}")

        if frost_prob >= 0.5:
            st.warning("⚠️ 注意：概率超过 50%，建议防霜。")
        else:
            st.success("✅ 安全：概率低于 50%，气象条件良好。")

# --- 功能 2：批量预测 ---
with tab2:
    st.subheader("上传 Excel 文件")
    st.markdown("请确保文件包含列：`露点温度`, `气温`, `温度露点差`, `相对湿度`, `云量`")

    uploaded_file = st.file_uploader("点击上传", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            required_cols = ["露点温度", "气温", "温度露点差", "相对湿度", "云量"]

            # 检查列名
            if not all(col in data.columns for col in required_cols):
                st.error(f"文件缺少必要列，请检查！需要包含: {required_cols}")
            else:
                # 预测
                X_test = data[required_cols].values
                probs = rf_model.predict_proba(X_test)[:, 1]

                data['有霜概率'] = probs
                data['预测结果'] = ['有霜' if p >= 0.5 else '无霜' for p in probs]

                st.success(f"成功预测 {len(data)} 条数据！")

                # 简单的高亮显示
                st.dataframe(data)

                # 下载按钮
                csv = data.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载预测结果 (CSV)",
                    data=csv,
                    file_name="预测结果.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"文件解析错误: {e}")

# 页脚
st.markdown("---")
st.caption("技术支持：随机森林预测模型 v1.0")