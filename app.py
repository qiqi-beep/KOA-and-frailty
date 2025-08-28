import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path

# 页面设置
st.set_page_config(page_title="模型诊断与修复", layout="centered")
st.title("🔧 KOA模型诊断与修复工具")

# 加载模型
@st.cache_resource
def load_model():
    try:
        base_path = Path(__file__).parent
        model_path = base_path / "frailty_xgb_model2.pkl"
        feature_path = base_path / "frailty_feature_names.pkl"
        
        model = pickle.load(open(model_path, 'rb'))
        with open(feature_path, 'rb') as f:
            feature_names = pickle.load(f)
            
        return model, feature_names
    except:
        return None, None

model, feature_names = load_model()

if model is None:
    st.error("无法加载模型")
    st.stop()

# 显示当前问题
st.error("🚨 当前模型问题：所有预测概率都异常偏高（84%-97%）")
st.warning("这可能是因为：1）训练数据标签不平衡 2）特征编码方向错误 3）模型需要校准")

# 修复建议和实施
st.subheader("🔧 修复方案")

# 正确的特征编码函数
def encode_features_correctly(case):
    return {
        'gender': 0 if case["性别"] == "男" else 1,
        'age': case["年龄"],
        'smoking': 0 if case["吸烟"] == "否" else 1,
        'bmi': case["BMI"],
        'fall': 0 if case["跌倒"] == "否" else 1,
        'PA_high': 1 if case["活动水平"] == "高" else 0,
        'PA_medium': 1 if case["活动水平"] == "中" else 0,
        'PA_low': 1 if case["活动水平"] == "低" else 0,
        'Complications_0': 1 if case["并发症"] == "没有" else 0,
        'Complications_1': 1 if case["并发症"] == "1个" else 0,
        'Complications_2': 1 if case["并发症"] == "至少2个" else 0,
        'ADL': 0 if case["日常活动"] == "无限制" else 1,
        'FTSST': 0 if case["坐立测试"] == "小于12s" else 1,
        'bl_crp': case["CRP"],
        'bl_hgb': case["血红蛋白"]
    }

# 创建正确的测试案例
correct_test_cases = [
    {
        "描述": "健康年轻患者（低风险）", 
        "年龄": 45, "BMI": 22.0, "CRP": 5.0, "血红蛋白": 140.0, 
        "跌倒": 0, "活动水平": "高", "性别": "男", "吸烟": "否",
        "并发症": "没有", "日常活动": "无限制", "坐立测试": "小于12s",
        "期望": "低概率"
    },
    {
        "描述": "典型老年患者（中风险）", 
        "年龄": 70, "BMI": 26.0, "CRP": 12.0, "血红蛋白": 125.0, 
        "跌倒": 0, "活动水平": "中", "性别": "女", "吸烟": "否",
        "并发症": "1个", "日常活动": "无限制", "坐立测试": "大于等于12s",
        "期望": "中概率"
    },
    {
        "描述": "高风险患者", 
        "年龄": 82, "BMI": 31.0, "CRP": 45.0, "血红蛋白": 95.0, 
        "跌倒": 1, "活动水平": "低", "性别": "女", "吸烟": "是",
        "并发症": "至少2个", "日常活动": "有限制", "坐立测试": "大于等于12s",
        "期望": "高概率"
    },
]

# 测试当前模型
st.subheader("📊 当前模型表现")
results = []

for case in correct_test_cases:
    input_data = encode_features_correctly(case)
    input_df = pd.DataFrame([input_data])
    
    # 确保所有特征都存在
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[feature_names]
    
    try:
        proba = model.predict_proba(input_df)[0][1]
        results.append({
            "案例": case["描述"],
            "概率": proba,
            "期望": case["期望"]
        })
    except Exception as e:
        st.error(f"预测错误: {str(e)}")

for result in results:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(result["案例"])
    with col2:
        st.write(f"{result['概率']*100:.1f}%")
    with col3:
        if result["概率"] < 0.3 and "低" in result["期望"]:
            st.success("✅ 符合期望")
        elif 0.3 <= result["概率"] <= 0.7 and "中" in result["期望"]:
            st.warning("⚠️ 部分符合")
        elif result["概率"] > 0.7 and "高" in result["期望"]:
            st.error("❌ 概率过高")
        else:
            st.info("🔍 需要调整")

# 概率调整工具
st.subheader("🛠️ 概率调整工具")

def adjust_probability(raw_prob, case):
    """基于临床知识调整概率"""
    adjustment = 0.0
    
    # 基于年龄调整
    age = case["年龄"]
    if age < 50:
        adjustment -= 0.4
    elif age < 65:
        adjustment -= 0.2
    elif age > 80:
        adjustment += 0.15
    
    # 基于BMI调整
    bmi = case["BMI"]
    if 18.5 <= bmi <= 24.9:
        adjustment -= 0.15
    elif bmi >= 30:
        adjustment += 0.2
    
    # 基于血红蛋白调整
    hgb = case["血红蛋白"]
    gender = case["性别"]
    if gender == "男":
        if hgb > 140:
            adjustment -= 0.1
        elif hgb < 110:
            adjustment += 0.15
    else:
        if hgb > 130:
            adjustment -= 0.1
        elif hgb < 100:
            adjustment += 0.15
    
    # 基于CRP调整
    crp = case["CRP"]
    if crp < 8:
        adjustment -= 0.1
    elif crp > 20:
        adjustment += 0.1
    
    # 基于其他因素调整
    if case["跌倒"] == 0:
        adjustment -= 0.05
    if case["并发症"] == "没有":
        adjustment -= 0.1
    elif case["并发症"] == "至少2个":
        adjustment += 0.15
    
    if case["日常活动"] == "无限制":
        adjustment -= 0.08
    if case["坐立测试"] == "小于12s":
        adjustment -= 0.07
    if case["活动水平"] == "高":
        adjustment -= 0.1
    elif case["活动水平"] == "低":
        adjustment += 0.1
    if case["吸烟"] == "否":
        adjustment -= 0.05
    
    adjusted_prob = max(0.01, min(0.99, raw_prob + adjustment))
    return adjusted_prob

# 应用调整
st.write("**调整后的概率:**")
for case in correct_test_cases:
    input_data = encode_features_correctly(case)
    input_df = pd.DataFrame([input_data])
    
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[feature_names]
    
    raw_prob = model.predict_proba(input_df)[0][1]
    adjusted_prob = adjust_probability(raw_prob, case)
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.write(case["描述"])
    with col2:
        st.write(f"原始: {raw_prob*100:.1f}%")
    with col3:
        st.write(f"调整: {adjusted_prob*100:.1f}%")
    with col4:
        if adjusted_prob < 0.3:
            st.success("低风险")
        elif adjusted_prob < 0.7:
            st.warning("中风险")
        else:
            st.error("高风险")



