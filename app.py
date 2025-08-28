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

# 加载预处理器、模型和预测器
@st.cache_resource
def load_components():
    try:
        base_path = Path(__file__).parent
        
        # 加载预处理器
        try:
            with open(base_path / "frailty_preprocessor.pkl", 'rb') as f:
                preprocessor = pickle.load(f)
        except Exception as e:
            st.error(f"预处理器加载失败: {str(e)}")
            return None, None, None
        
        # 加载模型
        try:
            model = joblib.load(base_path / "frailty_xgb_model28.pkl")
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            return None, None, None
        
        # 加载预测器（包含优化阈值）
        try:
            with open(base_path / "frailty_predictor.pkl", 'rb') as f:
                predictor = pickle.load(f)
        except Exception as e:
            st.warning(f"预测器加载失败，将使用默认阈值: {str(e)}")
            predictor = None
        
        return preprocessor, model, predictor
        
    except Exception as e:
        st.error(f"组件加载失败: {str(e)}")
        st.write("当前目录内容:", [f.name for f in Path('.').glob('*')])
        return None, None, None

preprocessor, model, predictor = load_components()

if preprocessor is None or model is None:
    st.stop()

# 初始化SHAP解释器
@st.cache_resource
def create_explainer(_model, _preprocessor):
    try:
        # 创建一个示例数据来初始化解释器
        example_data = pd.DataFrame({
            'gender': ['男'],
            'age': [65],
            'smoking': ['否'],
            'bmi': [24.5],
            'fall': ['否'],
            'activity': ['中'],
            'complication': ['没有'],
            'daily_activity': ['无限制'],
            'sit_stand': ['小于12s'],
            'crp': [3.2],
            'hgb': [132.5]
        })
        
        # 预处理示例数据
        example_processed = _preprocessor.transform(example_data)
        
        # 创建解释器
        if hasattr(_model, 'predict_proba'):
            return shap.TreeExplainer(_model, example_processed, model_output="probability")
        else:
            return shap.TreeExplainer(_model, example_processed, model_output="margin")
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
            # 使用预处理器转换数据
            processed_data = preprocessor.transform(input_df)
            
            # 使用预测器或模型进行预测
            if predictor is not None:
                # 使用预测器（包含优化阈值）
                prediction = predictor.predict(input_df)
                proba = predictor.predict_proba(input_df)[0, 1]
            else:
                # 直接使用模型
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(processed_data)[0, 1]
                    prediction = (proba >= 0.57).astype(int)  # 使用优化阈值0.57
                else:
                    raw_pred = model.predict(processed_data)[0]
                    proba = 1 / (1 + np.exp(-raw_pred))
                    prediction = 1 if proba >= 0.57 else 0
            
            # 显示预测结果
            st.success(f"📊 预测结果: 患者衰弱概率为 {proba*100:.2f}%")
            
            # 根据优化阈值进行风险评估
            if prediction == 1:
                st.error("""⚠️ **高风险：建议立即临床干预**""")
                st.write("- 每周随访监测")
                st.write("- 必须物理治疗干预")
                st.write("- 全面评估并发症")
                st.write(f"- 使用优化阈值 {0.57 if predictor is None else '自定义'} 进行判断")
            else:
                st.success("""✅ **低风险：建议常规健康管理**""")
                st.write("- 每年体检一次")
                st.write("- 保持健康生活方式")
                st.write("- 预防性健康指导")
                st.write(f"- 使用优化阈值 {0.57 if predictor is None else '自定义'} 进行判断")
            
            # SHAP可视化
            if explainer is not None:
                try:
                    # 获取SHAP值
                    shap_values = explainer.shap_values(processed_data)
                    expected_value = explainer.expected_value
                    
                    # 获取特征名称（从预处理器中）
                    try:
                        # 尝试获取预处理后的特征名称
                        if hasattr(preprocessor, 'get_feature_names_out'):
                            feature_names = preprocessor.get_feature_names_out()
                        else:
                            # 使用默认特征名称
                            feature_names = [f'feature_{i}' for i in range(processed_data.shape[1])]
                    except:
                        feature_names = [f'feature_{i}' for i in range(processed_data.shape[1])]
                    
                    # 特征名称映射
                    feature_descriptions = {
                        'age': f'年龄={int(age)}岁',
                        'bmi': f'BMI={bmi:.1f}',
                        'crp': f'CRP={crp:.1f}mg/L',
                        'hgb': f'血红蛋白={hgb:.1f}g/L',
                        'gender': f'性别={gender}',
                        'smoking': f'吸烟={smoking}',
                        'fall': f'跌倒={fall}',
                        'activity': f'活动水平={activity}',
                        'complication': f'并发症={complication}',
                        'daily_activity': f'日常活动={daily_activity}',
                        'sit_stand': f'坐立测试={sit_stand}'
                    }
                    
                    # 创建SHAP决策图
                    st.subheader(f"🧠 决策依据分析（{'衰弱' if prediction == 1 else '非衰弱'}类）")
                    
                    # 使用瀑布图
                    plt.figure(figsize=(12, 8))
                    shap.plots.waterfall(shap.Explanation(
                        values=shap_values[0], 
                        base_values=expected_value,
                        feature_names=feature_names,
                        data=processed_data[0]
                    ))
                    st.pyplot(plt.gcf(), clear_figure=True)
                    plt.close()
                    
                    # 图例说明
                    st.markdown("""
                    **图例说明:**
                    - 📊 **条形图**：显示每个特征对预测的影响程度
                    - ➕ **正值**：增加衰弱风险的特征
                    - ➖ **负值**：降低衰弱风险的特征
                    - 📍 **基准值**：平均预测值
                    - 🎯 **最终值**：当前患者的预测值
                    """)
                    
                except Exception as e:
                    st.warning(f"SHAP可视化失败: {str(e)}")
            else:
                st.info("SHAP解释器不可用，无法显示决策依据分析")
                
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")
            st.write("调试信息:", {
                "输入数据形状": input_df.shape,
                "预处理后形状": processed_data.shape if 'processed_data' in locals() else "N/A",
                "模型类型": type(model)
            })

# 显示模型信息
with st.expander("ℹ️ 模型信息"):
    st.write(f"**模型类型:** {type(model).__name__}")
    st.write(f"**预处理器:** {type(preprocessor).__name__}")
    if predictor is not None:
        st.write(f"**预测器:** {type(predictor).__name__}")
        st.write(f"**优化阈值:** 0.57")
    else:
        st.write("**预测器:** 未使用")
        st.write("**阈值:** 默认0.57")
    st.write("**特征数量:**", processed_data.shape[1] if 'processed_data' in locals() else "未知")

# 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")
