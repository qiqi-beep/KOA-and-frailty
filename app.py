import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import time
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# 页面设置
st.set_page_config(page_title="KOA 患者衰弱风险预测", layout="centered")
st.title("🩺 膝骨关节炎患者衰弱风险预测系统")
st.markdown("根据输入的临床特征，预测膝关节骨关节炎（KOA）患者发生衰弱（Frailty）的概率，并可视化决策依据。")

# 自定义CSS实现全页面居中
st.markdown(
    """
    <style>
    .main > div {
        max-width: 800px;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 加载模型和特征名称
@st.cache_resource
def load_model_and_features():
    try:
        import joblib
        from pathlib import Path
        
        base_path = Path(__file__).parent
        model_path = base_path / "frailty_xgb_model28.pkl"
        feature_path = base_path / "frailty_feature_names.pkl"
        
        # 验证文件
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在于: {model_path}")
        
        # 尝试多种加载方式
        try:
            # 方式1：优先尝试joblib加载
            model = joblib.load(model_path)
            if not hasattr(model, 'predict'):
                raise ValueError("加载的对象不是有效模型")
                
        except Exception as e:
            st.warning(f"Joblib加载失败，尝试XGBoost原生加载: {str(e)}")
            try:
                # 方式2：尝试XGBoost原生加载
                model = xgb.Booster()
                model.load_model(str(model_path))
            except Exception as e:
                raise ValueError(f"所有加载方式均失败: {str(e)}")
        
        # 加载特征名
        with open(feature_path, 'rb') as f:
            feature_names = pickle.load(f)
            
        return model, feature_names
        
    except Exception as e:
        st.error(f"加载失败: {str(e)}")
        st.write("""
        **故障排除步骤:**
        1. 确认模型文件完整
        2. 检查文件格式是否正确
        3. 尝试重新生成模型文件
        """)
        st.write("当前目录内容:", [f.name for f in Path('.').glob('*')])
        return None, None

model, feature_names = load_model_and_features()

# 如果模型加载失败，停止执行
if model is None or feature_names is None:
    st.stop()

# 初始化SHAP解释器
@st.cache_resource
def create_explainer(_model):
    try:
        # 尝试不同的方式创建解释器
        if hasattr(_model, 'predict_proba'):
            # 如果是scikit-learn接口的模型
            return shap.TreeExplainer(_model, model_output="probability")
        else:
            # 如果是原生XGBoost模型
            return shap.TreeExplainer(_model, model_output="margin")
    except:
        # 如果上述方法都失败，使用默认方式
        return shap.TreeExplainer(_model)

explainer = create_explainer(model)

# 创建输入表单
with st.form("patient_input_form"):
    st.markdown("---")
    st.subheader("📋 请填写以下信息") 
    
    # 表单字段
    gender = st.radio("您的性别", ["女", "男"])
    age = st.number_input("您的年龄（岁）", min_value=0, max_value=120, value=60)
    smoking = st.radio("您是否吸烟？", ["否", "是"])
    bmi = st.number_input("输入您的 BMI（体重指数，kg/m²）", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
    fall = st.radio("您过去一年是否发生过跌倒？", ["否", "是"])
    activity = st.radio("您觉得平时的体力活动水平", ["低水平", "中水平", "高水平"])
    complication = st.radio("您是否有并发症？", ["没有", "1个", "至少2个"])
    daily_activity = st.radio("您日常生活能力受限吗？", ["无限制", "有限制"])
    sit_stand = st.radio("输入您连续5次坐立的时间（s）", ["小于12s", "大于等于12s"])
    crp = st.number_input("输入您的C反应蛋白值（mg/L）", min_value=0, max_value=1000, value=200)
    hgb = st.number_input("输入您的血红蛋白含量（g/L）", min_value=0.0, max_value=1000.0, value=70.0, step=0.1)
        
    submitted = st.form_submit_button("开始评估")

# 处理输入数据并预测
if submitted:
    with st.spinner('正在计算...'):
        time.sleep(0.5)
        
        # 将输入转换为模型需要的格式
        input_data = {
            'gender': 1 if gender == "女" else 0,
            'age': age,
            'smoking': 1 if smoking == "是" else 0,
            'bmi': bmi,
            'fall': 1 if fall == "是" else 0,
            'PA_high': 1 if activity == "高水平" else 0,
            'PA_medium': 1 if activity == "中水平" else 0,
            'PA_low': 1 if activity == "低水平" else 0,
            'Complications_0': 1 if complication == "没有" else 0,
            'Complications_1': 1 if complication == "1个" else 0,
            'Complications_2': 1 if complication == "至少2个" else 0,
            'ADL': 1 if daily_activity == "有限制" else 0,
            'FTSST': 1 if sit_stand == "大于等于12s" else 0,
            'bl_crp': crp,
            'bl_hgb': hgb
        }
        
        # 创建DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 确保所有特征都存在
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # 重新排序列
        input_df = input_df[feature_names]
        
        try:
            # 根据模型类型进行预测
            if hasattr(model, 'predict_proba'):
                # scikit-learn接口的模型
                frail_prob = model.predict_proba(input_df)[0][1]
            else:
                # 原生XGBoost模型
                dmatrix = xgb.DMatrix(input_df, feature_names=feature_names)
                raw_pred = model.predict(dmatrix)[0]
                frail_prob = 1 / (1 + np.exp(-raw_pred))
            
            # 显示预测结果
            st.success(f"📊 预测结果: 患者衰弱概率为 {frail_prob*100:.2f}%")
            
            # 风险评估
            if frail_prob > 0.8:
                st.error("""⚠️ **高风险：建议立即临床干预**""")
                st.write("- 每周随访监测")
                st.write("- 必须物理治疗干预")
                st.write("- 全面评估并发症")
            elif frail_prob > 0.3:
                st.warning("""⚠️ **中风险：建议定期监测**""")
                st.write("- 每3-6个月评估一次")
                st.write("- 建议适度运动计划")
                st.write("- 基础营养评估")
            else:
                st.success("""✅ **低风险：建议常规健康管理**""")
                st.write("- 每年体检一次")
                st.write("- 保持健康生活方式")
                st.write("- 预防性健康指导")
            
            # SHAP可视化
            try:
                # 获取SHAP值
                if hasattr(model, 'predict_proba'):
                    # scikit-learn接口
                    shap_values = explainer.shap_values(input_df)
                    expected_value = explainer.expected_value[1]  # 取正类的期望值
                else:
                    # 原生XGBoost接口
                    shap_values = explainer.shap_values(dmatrix)
                    expected_value = explainer.expected_value
                
                # 特征名称映射
                feature_names_mapping = {
                    'age': f'年龄={int(age)}岁',
                    'bmi': f'BMI={bmi:.1f}',
                    'bl_crp': f'CRP={crp}mg/L',
                    'bl_hgb': f'血红蛋白={hgb:.1f}g/L',
                    'Complications_0': f'并发症={"无" if complication=="没有" else "有"}',
                    'Complications_1': f'并发症={"无" if complication=="没有" else "有"}',
                    'Complications_2': f'并发症={"无" if complication=="没有" else "有"}',
                    'FTSST': f'坐立测试={"慢(≥12s)" if sit_stand=="大于等于12s" else "快(<12s)"}',
                    'fall': f'跌倒史={"有" if fall=="是" else "无"}',
                    'ADL': f'日常活动={"受限" if daily_activity=="有限制" else "正常"}',
                    'gender': f'性别={"女" if gender=="女" else "男"}',
                    'PA_high': f'活动水平={"高" if activity=="高水平" else "中/低"}',
                    'PA_medium': f'活动水平={"中" if activity=="中水平" else "高/低"}',
                    'PA_low': f'活动水平={"低" if activity=="低水平" else "高/中"}',
                    'smoking': f'吸烟={"是" if smoking=="是" else "否"}'
                }

                # 创建SHAP决策图
                st.subheader(f"🧠 决策依据分析（{'衰弱' if frail_prob > 0.5 else '非衰弱'}类）")
                plt.figure(figsize=(14, 4))
                
                # 根据SHAP值的类型调整可视化
                if isinstance(shap_values, list):
                    # 如果是多类输出的列表，取第二类（正类）
                    shap_val = shap_values[1][0] if len(shap_values) > 1 else shap_values[0]
                else:
                    shap_val = shap_values[0]
                
                shap.force_plot(
                    base_value=expected_value,
                    shap_values=shap_val,
                    features=input_df.iloc[0],
                    feature_names=[feature_names_mapping.get(f, f) for f in input_df.columns],
                    matplotlib=True,
                    show=False,
                    plot_cmap="RdBu"
                )
                st.pyplot(plt.gcf(), clear_figure=True)
                plt.close()
                
                # 图例说明
                st.markdown("""
                **图例说明:**
                - 🔴 **红色**：增加衰弱风险的特征  
                - 🟢 **绿色**：降低衰弱风险的特征  
                - 📏 **长度**：特征对预测结果的影响程度
                """)
                
            except Exception as e:
                st.warning(f"SHAP可视化暂时不可用: {str(e)}")
                st.info("""
                **替代分析:**
                主要影响因素通常包括：
                - 年龄和BMI
                - 日常活动能力
                - 并发症数量
                - 体力活动水平
                """)
                
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")
            # 提供调试信息但不暴露敏感数据
            st.info("""
            **调试信息:**
            - 输入特征数量: {}
            - 模型所需特征数量: {}
            """.format(len(input_df.columns), len(feature_names)))

# 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")

