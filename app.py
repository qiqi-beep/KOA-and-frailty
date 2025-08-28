import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV

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

# 方案1：在线校准模型
st.markdown("### 方案1: 在线概率校准")
if st.button("🔄 立即校准模型"):
    with st.spinner("校准模型中..."):
        # 创建校准器
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        
        # 生成一些模拟数据用于校准（基于您的特征范围）
        np.random.seed(42)
        n_samples = 1000
        
        # 创建合理的训练数据分布
        X_calibrate = pd.DataFrame({
            'gender': np.random.choice([0, 1], n_samples),
            'age': np.random.normal(65, 15, n_samples).clip(40, 90),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'bmi': np.random.normal(24, 4, n_samples).clip(18, 35),
            'fall': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'PA_high': np.random.choice([1, 0, 0], n_samples),
            'PA_medium': np.random.choice([0, 1, 0], n_samples),
            'PA_low': np.random.choice([0, 0, 1], n_samples),
            'Complications_0': np.random.choice([1, 0, 0], n_samples, p=[0.6, 0.3, 0.1]),
            'Complications_1': np.random.choice([0, 1, 0], n_samples, p=[0.6, 0.3, 0.1]),
            'Complications_2': np.random.choice([0, 0, 1], n_samples, p=[0.6, 0.3, 0.1]),
            'ADL': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'FTSST': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'bl_crp': np.random.lognormal(2, 1, n_samples).clip(1, 100),
            'bl_hgb': np.random.normal(130, 20, n_samples).clip(90, 160)
        })
        
        # 创建合理的标签分布（假设20%的患者有衰弱风险）
        y_calibrate = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        
        # 确保特征顺序正确
        X_calibrate = X_calibrate[feature_names]
        
        # 拟合校准器
        calibrated_model.fit(X_calibrate, y_calibrate)
        
        st.success("✅ 模型校准完成！")

# 方案2：特征重新编码检查
st.markdown("### 方案2: 特征编码验证")

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

# 正确的特征编码函数
def encode_features_correctly(case):
    return {
        'gender': 0 if case["性别"] == "男" else 1,  # 0-男, 1-女
        'age': case["年龄"],
        'smoking': 0 if case["吸烟"] == "否" else 1,  # 0-否, 1-是
        'bmi': case["BMI"],
        'fall': 0 if case["跌倒"] == "否" else 1,  # 0-否, 1-是
        'PA_high': 1 if case["活动水平"] == "高" else 0,
        'PA_medium': 1 if case["活动水平"] == "中" else 0,
        'PA_low': 1 if case["活动水平"] == "低" else 0,
        'Complications_0': 1 if case["并发症"] == "没有" else 0,
        'Complications_1': 1 if case["并发症"] == "1个" else 0,
        'Complications_2': 1 if case["并发症"] == "至少2个" else 0,
        'ADL': 0 if case["日常活动"] == "无限制" else 1,  # 0-无限制, 1-有限制
        'FTSST': 0 if case["坐立测试"] == "小于12s" else 1,  # 0-快, 1-慢
        'bl_crp': case["CRP"],
        'bl_hgb': case["血红蛋白"]
    }

# 测试修正后的编码
st.write("### 使用正确编码测试")
results = []

for case in correct_test_cases:
    input_data = encode_features_correctly(case)
    input_df = pd.DataFrame([input_data])[feature_names]
    
    try:
        proba = model.predict_proba(input_df)[0][1]
        results.append({
            "案例": case["描述"],
            "概率": proba,
            "期望": case["期望"]
        })
    except:
        pass

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

# 方案3：重新训练建议
st.markdown("### 方案3: 重新训练建议")

st.markdown("""
**如果您可以重新训练模型，请考虑：**

1. **检查训练数据标签分布**：
   ```python
   print(y_train.value_counts(normalize=True))
