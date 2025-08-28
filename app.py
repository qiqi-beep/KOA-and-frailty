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
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

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

# 加载模型
@st.cache_resource
def load_model():
    try:
        base_path = Path(__file__).parent
        model_path = base_path / "frailty_xgb_model28.pkl"
        
        if model_path.exists():
            model = joblib.load(model_path)
            st.success("✅ 模型加载成功")
            return model
        else:
            st.error("❌ 模型文件不存在")
            return None
            
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

model = load_model()

if model is None:
    st.error("无法加载模型，请检查文件是否存在")
    st.stop()

# 手动预处理函数
def manual_preprocess(input_df):
    # 数值特征标准化（使用训练数据的典型范围）
    numeric_stats = {
        'age': {'mean': 65, 'std': 10},
        'bmi': {'mean': 25, 'std': 4},
        'crp': {'mean': 5, 'std': 3},
        'hgb': {'mean': 130, 'std': 15}
    }
    
    # 分类特征one-hot编码（根据您的特征名称）
    categorical_features = {
        'gender': ['男', '女'],
        'smoking': ['否', '是'],
        'fall': ['否', '是'],
        'activity': ['高', '中', '低'],
        'complication': ['没有', '1个', '至少2个'],
        'daily_activity': ['无限制', '有限制'],
        'sit_stand': ['小于12s', '大于等于12s']
    }
    
    processed_data = []
    
    # 处理数值特征
    for feature in ['age', 'bmi', 'crp', 'hgb']:
        if feature in input_df.columns:
            value = input_df[feature].iloc[0]
            normalized = (value - numeric_stats[feature]['mean']) / numeric_stats[feature]['std']
            processed_data.append(normalized)
    
    # 处理分类特征 - 按照您提供的特征名称顺序
    # gender_0, gender_1
    gender_val = 0 if input_df['gender'].iloc[0] == '男' else 1
    processed_data.extend([1 - gender_val, gender_val])
    
    # smoke_0, smoke_1
    smoke_val = 0 if input_df['smoking'].iloc[0] == '否' else 1
    processed_data.extend([1 - smoke_val, smoke_val])
    
    # fall_0, fall_1
    fall_val = 0 if input_df['fall'].iloc[0] == '否' else 1
    processed_data.extend([1 - fall_val, fall_val])
    
    # PA_0, PA_1, PA_2
    activity_map = {'高': 0, '中': 1, '低': 2}
    activity_val = activity_map[input_df['activity'].iloc[0]]
    pa_features = [0, 0, 0]
    pa_features[activity_val] = 1
    processed_data.extend(pa_features)
    
    # Complications_0, Complications_1, Complications_2
    complication_map = {'没有': 0, '1个': 1, '至少2个': 2}
    complication_val = complication_map[input_df['complication'].iloc[0]]
    comp_features = [0, 0, 0]
    comp_features[complication_val] = 1
    processed_data.extend(comp_features)
    
    # ADL_0, ADL_1
    adl_val = 0 if input_df['daily_activity'].iloc[0] == '无限制' else 1
    processed_data.extend([1 - adl_val, adl_val])
    
    # FTSST (已经是数值特征，但需要放在正确位置)
    ftsst_val = 0 if input_df['sit_stand'].iloc[0] == '小于12s' else 1
    processed_data.append(ftsst_val)
    
    return np.array([processed_data])

# 获取特征名称（根据您的描述）
def get_feature_names():
    return [
        'age', 'bmi', 'crp', 'hgb',           # 数值特征
        'gender_0', 'gender_1',               # 性别
        'smoke_0', 'smoke_1',                 # 吸烟
        'fall_0', 'fall_1',                   # 跌倒
        'PA_0', 'PA_1', 'PA_2',               # 活动水平
        'Complications_0', 'Complications_1', 'Complications_2',  # 并发症
        'ADL_0', 'ADL_1',                     # 日常活动
        'FTSST'                               # 坐立测试
    ]

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
            # 手动预处理数据
            processed_data = manual_preprocess(input_df)
            
            # 进行预测
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(processed_data)[0, 1]
                prediction = 1 if proba >= 0.57 else 0
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
                st.write(f"- 使用优化阈值 0.57 进行判断")
            else:
                st.success("""✅ **低风险：建议常规健康管理**""")
                st.write("- 每年体检一次")
                st.write("- 保持健康生活方式")
                st.write("- 预防性健康指导")
                st.write(f"- 使用优化阈值 0.57 进行判断")
            
            # 简单特征重要性显示（替代SHAP）
            st.subheader("📈 特征重要性分析")
            
            # 获取特征名称
            feature_names = get_feature_names()
            
            # 如果是XGBoost模型，可以获取特征重要性
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    '特征': feature_names,
                    '重要性': model.feature_importances_
                }).sort_values('重要性', ascending=False).head(10)
                
                st.bar_chart(importance_df.set_index('特征')['重要性'])
                st.write("**Top 10 重要特征:**")
                for i, row in importance_df.iterrows():
                    st.write(f"- {row['特征']}: {row['重要性']:.3f}")
            
            # 显示处理后的特征值
            with st.expander("查看处理后的特征值"):
                st.write("**特征名称和值:**")
                for i, (name, value) in enumerate(zip(feature_names, processed_data[0])):
                    st.write(f"- {name}: {value:.3f}")
                
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")
            import traceback
            st.write("详细错误信息:", traceback.format_exc())

# 显示系统信息
with st.expander("ℹ️ 系统信息"):
    st.write(f"**模型类型:** {type(model).__name__}")
    st.write(f"**特征数量:** {len(get_feature_names())}")
    st.write(f"**预测阈值:** 0.57")
    st.write("**使用内置预处理:** ✅ 是")

# 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")
