import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import time

# 页面设置
st.set_page_config(page_title="KOA 患者衰弱风险预测", layout="centered")
st.title("🩺 膝骨关节炎患者衰弱风险预测系统")
st.markdown("根据输入的临床特征，预测膝关节骨关节炎（KOA）患者发生衰弱（Frailty）的概率，并可视化决策依据。")

# 自定义CSS：居中内容
st.markdown(
    """
    <style>
    .main > div {
        max-width: 800px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =======================
# 加载模型和特征名称
# =======================
@st.cache_resource
def load_model_and_features():
    try:
        base_path = Path(__file__).parent

        # 文件路径
        model_path = base_path / "frailty_xgb_model (1).pkl"
        feature_path = base_path / "frailty_feature_names.pkl"

        # 检查文件是否存在
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not feature_path.exists():
            raise FileNotFoundError(f"特征名称文件不存在: {feature_path}")

        # 加载模型（XGBoost 原生或 joblib）
        try:
            model = joblib.load(model_path)
        except Exception:
            model = xgb.Booster()
            model.load_model(str(model_path))

        # 加载特征名称
        feature_names = joblib.load(feature_path)

        return model, feature_names

    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.write("当前目录文件:", [f.name for f in Path('.').iterdir()])
        st.stop()

model, feature_names = load_model_and_features()

# 显示加载的特征名（调试用，上线可注释）
# st.write("✅ 加载的特征名:", feature_names)

# =======================
# 初始化 SHAP 解释器
# =======================
@st.cache_resource
def create_explainer(_model):
    # 如果是 xgb.Booster，需要包装成可调用模型
    if isinstance(_model, xgb.Booster):
        def predict_fn(x):
            dmat = xgb.DMatrix(x, feature_names=feature_names)
            return _model.predict(dmat)
        return shap.Explainer(predict_fn, pd.DataFrame(columns=feature_names))
    else:
        return shap.TreeExplainer(_model, model_output="raw")

explainer = create_explainer(model)

# =======================
# 用户输入表单
# =======================
with st.form("patient_input_form"):
    st.markdown("---")
    st.subheader("📋 请填写以下信息")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("性别", ["男", "女"], index=0)
        age = st.number_input("年龄（岁）", min_value=0, max_value=120, value=70)
        smoke = st.radio("是否吸烟", ["否", "是"], index=0)
        bmi = st.number_input("BMI（kg/m²）", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
        fall = st.radio("过去一年是否跌倒", ["否", "是"], index=0)
        pa = st.radio("体力活动水平", ["高水平", "中水平", "低水平"], index=0)
        complications = st.radio("并发症数量", ["没有", "1个", "至少2个"], index=0)

    with col2:
        adl = st.radio("日常生活能力", ["无限制", "有限制"], index=0)
        ftsst = st.radio("连续5次坐立时间", ["小于12秒", "大于等于12秒"], index=0)
        crp = st.number_input("C反应蛋白（mg/L）", min_value=0.0, value=5.0, step=0.1)
        hgb = st.number_input("血红蛋白（g/L）", min_value=0.0, value=130.0, step=0.1)

    submitted = st.form_submit_button("📊 开始预测")

# =======================
# 处理预测
# =======================
if submitted:
    with st.spinner('正在计算预测结果...'):
        time.sleep(0.3)

        # ======== 构建输入数据（严格按照模型期望的特征名）========
        input_dict = {
            'FTSST': 1 if ftsst == "大于等于12秒" else 0,
            'bmi': float(bmi),
            'age': int(age),
            'bl_crp': float(crp),
            'bl_hgb': float(hgb),
            'PA': 0 if pa == "高水平" else (1 if pa == "中水平" else 2),  # 0=高,1=中,2=低
            
            # One-Hot 编码 - 确保包含所有期望的特征
            'Complications_0': 1 if complications == "没有" else 0,
            'Complications_1': 1 if complications == "1个" else 0,
            'Complications_2': 1 if complications == "至少2个" else 0,
            
            'fall_0': 1 if fall == "否" else 0,
            'fall_1': 1 if fall == "是" else 0,
            'fall_2': 0,  # 添加缺失的 fall_2 特征，设置为 0
            
            'ADL_0': 1 if adl == "无限制" else 0,
            'ADL_1': 1 if adl == "有限制" else 0,
            
            'gender_0': 1 if gender == "男" else 0,
            'gender_1': 1 if gender == "女" else 0,
            
            'smoke_0': 1 if smoke == "否" else 0,
            'smoke_1': 1 if smoke == "是" else 0,
            'smoke_2': 0,  # smoke_2 恒为 0
        }

        # 转为 DataFrame
        input_df = pd.DataFrame([input_dict])

        # 确保包含所有期望的特征，缺失的特征填充为 0
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0

        # 确保列顺序与模型训练时一致
        try:
            input_df = input_df[feature_names]
        except KeyError as e:
            st.error(f"特征名不匹配: {e}")
            st.write("输入特征:", list(input_df.columns))
            st.write("期望特征:", feature_names)
            st.stop()

        # ======== 预测 ========
        try:
            # 转换为 DMatrix
            dmatrix = xgb.DMatrix(input_df.values, feature_names=feature_names)

            # 获取预测（原始 logit）
            if isinstance(model, xgb.Booster):
                raw_pred = model.predict(dmatrix)[0]
            else:
                raw_pred = model.predict(dmatrix, output_margin=True)[0]

            # 转为概率
            frail_prob = 1 / (1 + np.exp(-raw_pred))

            # ======== 显示结果 ========
            st.success(f"🎯 预测结果: 衰弱概率为 **{frail_prob * 100:.2f}%**")

            # 风险分级
            if frail_prob > 0.8:
                st.error("⚠️ **高风险：建议立即临床干预**")
                st.write("- 每周随访监测")
                st.write("- 必须物理治疗干预")
                st.write("- 全面评估并发症")
            elif frail_prob > 0.3:
                st.warning("⚠️ **中风险：建议定期监测**")
                st.write("- 每3-6个月评估一次")
                st.write("- 建议适度运动计划")
                st.write("- 基础营养评估")
            else:
                st.success("✅ **低风险：建议常规健康管理**")
                st.write("- 每年体检一次")
                st.write("- 保持健康生活方式")
                st.write("- 预防性健康指导")

            # ======== SHAP 可视化 ========
            try:
                # 获取 SHAP 值
                shap_values = explainer(input_df).values[0]

                # 特征中文标签映射（用于可视化）
                display_names = {
                    'FTSST': f'坐立时间={ftsst}',
                    'bmi': f'BMI={bmi:.1f}',
                    'age': f'年龄={age}',
                    'bl_crp': f'CRP={crp:.1f}',
                    'bl_hgb': f'Hb={hgb:.1f}',
                    'PA': f'活动={pa}',
                    'Complications_0': f'并发症=无' if complications=="没有" else '',
                    'Complications_1': f'并发症=1个' if complications=="1个" else '',
                    'Complications_2': f'并发症=≥2' if complications=="至少2个" else '',
                    'fall_0': f'跌倒=否' if fall=="否" else '',
                    'fall_1': f'跌倒=是' if fall=="是" else '',
                    'fall_2': '',  # 不显示
                    'ADL_0': f'ADL=正常' if adl=="无限制" else '',
                    'ADL_1': f'ADL=受限' if adl=="有限制" else '',
                    'gender_0': f'性别=男' if gender=="男" else '',
                    'gender_1': f'性别=女' if gender=="女" else '',
                    'smoke_0': f'吸烟=否' if smoke=="否" else '',
                    'smoke_1': f'吸烟=是' if smoke=="是" else '',
                    'smoke_2': '',  # 不显示
                }

                # 过滤空标签
                disp_names = [display_names.get(f, f) for f in feature_names]
                disp_names = [n if n.strip() else '_' for n in disp_names]  # 避免空

                # 绘制 SHAP 决策图
                st.subheader("🧠 决策依据分析")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(
                    shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=input_df.iloc[0], feature_names=disp_names),
                    max_display=10,
                    show=False
                )
                st.pyplot(fig)
                plt.close()

                st.markdown("""
                **颜色说明**:  
                🔴 红色 → 增加衰弱风险  
                🟢 绿色 → 降低衰弱风险
                """)

            except Exception as e:
                st.warning(f"⚠️ 解释功能暂时不可用: {str(e)}")

        except Exception as e:
            st.error(f"预测失败: {str(e)}")

# =======================
# 页脚
# =======================
st.markdown("---")
st.caption("© 2025 KOA 衰弱风险预测系统 | 仅供科研与临床参考")
