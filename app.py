import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import time
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# 加载模型和组件
@st.cache_resource
def load_components():
    try:
        base_path = Path(__file__).parent
        components = {}
        
        # 尝试加载预处理器
        try:
            preprocessor_path = base_path / "frailty_preprocessor28.pkl"
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    components['preprocessor'] = pickle.load(f)
                st.success("✅ 预处理器加载成功")
            else:
                st.warning("⚠️ 预处理器文件不存在，将使用内置预处理逻辑")
        except Exception as e:
            st.warning(f"⚠️ 预处理器加载失败: {str(e)}")
        
        # 尝试加载模型
        try:
            model_path = base_path / "frailty_xgb_model28.pkl"
            if model_path.exists():
                components['model'] = joblib.load(model_path)
                st.success("✅ 模型加载成功")
            else:
                st.error("❌ 模型文件不存在")
                return None
        except Exception as e:
            st.error(f"❌ 模型加载失败: {str(e)}")
            return None
        
        # 尝试加载预测器
        try:
            predictor_path = base_path / "frailty_predictor28.pkl"
            if predictor_path.exists():
                with open(predictor_path, 'rb') as f:
                    components['predictor'] = pickle.load(f)
                st.success("✅ 预测器加载成功")
            else:
                st.warning("⚠️ 预测器文件不存在，将使用默认阈值0.57")
        except Exception as e:
            st.warning(f"⚠️ 预测器加载失败: {str(e)}")
        
        return components
        
    except Exception as e:
        st.error(f"❌ 组件加载失败: {str(e)}")
        st.write("当前目录内容:", [f.name for f in base_path.glob('*')])
        return None

components = load_components()

if components is None or 'model' not in components:
    st.error("无法加载必要的模型组件，请检查文件是否存在")
    st.stop()

model = components['model']
preprocessor = components.get('preprocessor')
predictor = components.get('predictor')

# 如果没有预处理器，创建内置的预处理逻辑
if preprocessor is None:
    # 定义特征处理
    numeric_features = ['age', 'bmi', 'crp', 'hgb']
    categorical_features = ['gender', 'smoking', 'fall', 'activity', 'complication', 'daily_activity', 'sit_stand']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

# 初始化SHAP解释器
@st.cache_resource
def create_explainer(_model, _preprocessor):
    try:
        # 创建示例数据来拟合预处理器（如果没有已经拟合）
        if not hasattr(_preprocessor, 'transform'):
            example_data = pd.DataFrame({
                'gender': ['男'], 'age': [65], 'smoking': ['否'], 'bmi': [24.5],
                'fall': ['否'], 'activity': ['中'], 'complication': ['没有'],
                'daily_activity': ['无限制'], 'sit_stand': ['小于12s'],
                'crp': [3.2], 'hgb': [132.5]
            })
            _preprocessor.fit(example_data)
        
        # 创建解释器
        if hasattr(_model, 'predict_proba'):
            return shap.TreeExplainer(_model, model_output="probability")
        else:
            return shap.TreeExplainer(_model, model_output="margin")
    except Exception as e:
        st.warning(f"SHAP解释器创建失败: {str(e)}")
        return None

explainer = create_explainer(model, preprocessor)

# 创建输入表单
with st.form("patient_input_form"):
    st.markdown("---")
    st.subheader("📋 请填写以下信息") 
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.radio("您的性别", ["男", "女"])
        age = st.number_input("您的年龄（岁）", min_value=0, max_value=120, value=65)
        smoking = st.radio("您是否吸烟？", ["否", "是"])
        bmi = st.number_input("BMI（kg/m²）", min_value=10.0, max_value=50.0, value=24.5, step=0.1)
        fall = st.radio("过去一年是否跌倒？", ["否", "是"])
    
    with col2:
        activity = st.radio("体力活动水平", ["高", "中", "低"])
        complication = st.radio("并发症数量", ["没有", "1个", "至少2个"])
        daily_activity = st.radio("日常生活能力", ["无限制", "有限制"])
        sit_stand = st.radio("5次坐立时间", ["小于12s", "大于等于12s"])
        crp = st.number_input("C反应蛋白（mg/L）", min_value=0.0, max_value=100.0, value=3.2, step=0.1)
        hgb = st.number_input("血红蛋白（g/L）", min_value=0.0, max_value=200.0, value=132.5, step=0.1)
        
    submitted = st.form_submit_button("开始评估")

# 手动编码函数（如果预处理器不可用）
def manual_preprocess(input_df):
    # 数值特征标准化
    numeric_features = ['age', 'bmi', 'crp', 'hgb']
    for feature in numeric_features:
        input_df[feature] = (input_df[feature] - input_df[feature].mean()) / input_df[feature].std()
    
    # 分类特征one-hot编码
    categorical_mapping = {
        'gender': {'男': [1, 0], '女': [0, 1]},
        'smoking': {'否': [1, 0], '是': [0, 1]},
        'fall': {'否': [1, 0], '是': [0, 1]},
        'activity': {'高': [1, 0, 0], '中': [0, 1, 0], '低': [0, 0, 1]},
        'complication': {'没有': [1, 0, 0], '1个': [0, 1, 0], '至少2个': [0, 0, 1]},
        'daily_activity': {'无限制': [1, 0], '有限制': [0, 1]},
        'sit_stand': {'小于12s': [1, 0], '大于等于12s': [0, 1]}
    }
    
    processed_features = []
    for col in input_df.columns:
        if col in categorical_mapping:
            values = categorical_mapping[col][input_df[col].iloc[0]]
            for i, val in enumerate(values):
                processed_features.append(val)
        elif col in numeric_features:
            processed_features.append(input_df[col].iloc[0])
    
    return np.array([processed_features])

# 处理输入数据并预测
if submitted:
    with st.spinner('正在计算...'):
        time.sleep(0.5)
        
        # 创建原始输入数据
        input_data = {
            'gender': gender,
            'age': age,
            'smoking': smoking,
            'bmi': bmi,
            'fall': fall,
            'activity': activity,
            'complication': complication,
            'daily_activity': daily_activity,
            'sit_stand': sit_stand,
            'crp': crp,
            'hgb': hgb
        }
        
        # 转换为DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            # 预处理数据
            if hasattr(preprocessor, 'transform'):
                processed_data = preprocessor.transform(input_df)
            else:
                processed_data = manual_preprocess(input_df)
            
            # 进行预测
            if predictor is not None and hasattr(predictor, 'predict_proba'):
                proba = predictor.predict_proba(input_df)[0, 1]
                prediction = predictor.predict(input_df)[0]
            else:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(processed_data)[0, 1]
                    prediction = (proba >= 0.57).astype(int)
                else:
                    raw_pred = model.predict(processed_data)[0]
                    proba = 1 / (1 + np.exp(-raw_pred))
                    prediction = 1 if proba >= 0.57 else 0
            
            # 显示预测结果
            st.success(f"📊 预测结果: 患者衰弱概率为 {proba*100:.2f}%")
            
            # 风险评估
            if prediction == 1:
                st.error("""⚠️ **高风险：建议立即临床干预**""")
                st.write("- 每周随访监测")
                st.write("- 必须物理治疗干预")
                st.write("- 全面评估并发症")
            else:
                st.success("""✅ **低风险：建议常规健康管理**""")
                st.write("- 每年体检一次")
                st.write("- 保持健康生活方式")
                st.write("- 预防性健康指导")
            
            # SHAP可视化
            if explainer is not None:
                try:
                    shap_values = explainer.shap_values(processed_data)
                    expected_value = explainer.expected_value
                    
                    # 创建SHAP决策图
                    st.subheader(f"🧠 决策依据分析（{'衰弱' if prediction == 1 else '非衰弱'}类）")
                    
                    # 使用瀑布图
                    plt.figure(figsize=(12, 8))
                    shap.plots.waterfall(shap.Explanation(
                        values=shap_values[0], 
                        base_values=expected_value,
                        feature_names=[f'feature_{i}' for i in range(processed_data.shape[1])],
                        data=processed_data[0]
                    ))
                    st.pyplot(plt.gcf(), clear_figure=True)
                    plt.close()
                    
                    st.markdown("""
                    **图例说明:**
                    - 📊 **条形图**：显示每个特征对预测的影响程度
                    - ➕ **正值**：增加衰弱风险的特征
                    - ➖ **负值**：降低衰弱风险的特征
                    """)
                    
                except Exception as e:
                    st.warning(f"SHAP可视化失败: {str(e)}")
            
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")

# 显示系统状态
with st.expander("ℹ️ 系统状态"):
    st.write(f"**模型加载:** {'✅ 成功' if 'model' in components else '❌ 失败'}")
    st.write(f"**预处理器:** {'✅ 外部' if components.get('preprocessor') else '🔄 内置'}")
    st.write(f"**预测器:** {'✅ 外部' if components.get('predictor') else '🔄 默认阈值0.57'}")
    st.write(f"**SHAP解释器:** {'✅ 可用' if explainer else '❌ 不可用'}")

# 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")
