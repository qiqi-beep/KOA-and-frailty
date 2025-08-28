import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path

# 页面设置
st.set_page_config(page_title="模型诊断工具", layout="centered")
st.title("🔍 XGBoost 模型诊断工具")
st.markdown("用于分析和诊断 `frailty_xgb_model2.pkl` 模型")

# 加载模型
@st.cache_resource
def load_model_for_diagnosis():
    try:
        base_path = Path(__file__).parent
        model_path = base_path / "frailty_xgb_model2.pkl"
        
        if not model_path.exists():
            st.error(f"模型文件不存在: {model_path}")
            return None
            
        # 尝试多种加载方式
        try:
            import joblib
            model = joblib.load(model_path)
            st.success("使用 joblib 成功加载模型")
            return model
        except:
            try:
                model = xgb.Booster()
                model.load_model(str(model_path))
                st.success("使用 XGBoost 原生加载成功")
                return model
            except Exception as e:
                st.error(f"所有加载方式均失败: {str(e)}")
                return None
                
    except Exception as e:
        st.error(f"加载失败: {str(e)}")
        return None

model = load_model_for_diagnosis()

if model is None:
    st.stop()

# 模型基本信息
st.subheader("📊 模型基本信息")

if hasattr(model, 'get_params'):
    # scikit-learn 接口的模型
    st.write("**模型类型:** Scikit-learn 接口的 XGBoost")
    params = model.get_params()
    st.write("**模型参数:**")
    st.json(params)
    
    # 获取特征重要性
    try:
        importance = model.feature_importances_
        st.write("**特征重要性:**", importance)
    except:
        st.write("无法获取特征重要性")
        
elif hasattr(model, 'save_config'):
    # 原生 XGBoost 模型
    st.write("**模型类型:** 原生 XGBoost")
    try:
        config = model.save_config()
        st.write("**模型配置:**")
        st.text(config[:1000] + "..." if len(config) > 1000 else config)
    except:
        st.write("无法获取模型配置")

# 创建测试数据来验证模型行为
st.subheader("🧪 模型行为测试")

# 创建一些典型的测试案例
test_cases = [
    {"描述": "健康年轻患者", "年龄": 40, "BMI": 22, "CRP": 5, "血红蛋白": 130, "跌倒": 0, "活动水平": "高"},
    {"描述": "典型老年患者", "年龄": 70, "BMI": 26, "CRP": 15, "血红蛋白": 110, "跌倒": 0, "活动水平": "中"},
    {"描述": "高风险患者", "年龄": 80, "BMI": 30, "CRP": 50, "血红蛋白": 90, "跌倒": 1, "活动水平": "低"},
]

results = []

for i, case in enumerate(test_cases):
    # 构建完整的输入数据（需要与您的特征名称匹配）
    input_data = {
        'gender': 0,  # 男
        'age': case["年龄"],
        'smoking': 0,  # 不吸烟
        'bmi': case["BMI"],
        'fall': case["跌倒"],
        'PA_high': 1 if case["活动水平"] == "高" else 0,
        'PA_medium': 1 if case["活动水平"] == "中" else 0,
        'PA_low': 1 if case["活动水平"] == "低" else 0,
        'Complications_0': 1,  # 无并发症
        'Complications_1': 0,
        'Complications_2': 0,
        'ADL': 0,  # 无日常活动限制
        'FTSST': 0,  # 坐立测试快
        'bl_crp': case["CRP"],
        'bl_hgb': case["血红蛋白"]
    }
    
    # 创建 DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 进行预测
    try:
        if hasattr(model, 'predict_proba'):
            # scikit-learn 接口
            proba = model.predict_proba(input_df)[0][1]
            raw_pred = model.predict(input_df)[0]
        else:
            # 原生 XGBoost
            dmatrix = xgb.DMatrix(input_df)
            raw_pred = model.predict(dmatrix)[0]
            proba = 1 / (1 + np.exp(-raw_pred))
        
        results.append({
            "案例": case["描述"],
            "原始预测值": raw_pred,
            "概率": proba,
            **case
        })
        
    except Exception as e:
        st.error(f"测试案例 {i+1} 预测失败: {str(e)}")

# 显示结果
if results:
    st.write("**测试结果:**")
    for result in results:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("案例", result["案例"])
        with col2:
            st.metric("原始预测值", f"{result['原始预测值']:.4f}")
        with col3:
            st.metric("衰弱概率", f"{result['概率']*100:.2f}%")
        
        # 显示详细参数
        with st.expander(f"查看 {result['案例']} 的详细参数"):
            st.write(f"年龄: {result['年龄']}岁")
            st.write(f"BMI: {result['BMI']}")
            st.write(f"CRP: {result['CRP']} mg/L")
            st.write(f"血红蛋白: {result['血红蛋白']} g/L")
            st.write(f"跌倒史: {'有' if result['跌倒'] else '无'}")
            st.write(f"活动水平: {result['活动水平']}")

# 模型校准检查
st.subheader("⚖️ 模型校准检查")

# 检查概率分布是否合理
if results:
    probabilities = [r['概率'] for r in results]
    avg_prob = np.mean(probabilities)
    
    st.write(f"**平均预测概率:** {avg_prob:.4f}")
    st.write(f"**概率范围:** {min(probabilities):.4f} - {max(probabilities):.4f}")
    
    if avg_prob > 0.8:
        st.warning("⚠️ 模型可能过于悲观，平均预测概率偏高")
    elif avg_prob < 0.2:
        st.warning("⚠️ 模型可能过于乐观，平均预测概率偏低")
    else:
        st.success("✅ 平均预测概率在合理范围内")

# 建议的修复步骤
st.subheader("🔧 建议的修复步骤")

st.markdown("""
如果模型概率不符合常理，可以尝试：

1. **检查数据预处理**：确保训练和预测时的特征处理一致
2. **重新校准模型**：使用 Platt scaling 或 isotonic regression
3. **调整模型参数**：特别是 `scale_pos_weight` 和 `max_delta_step`
4. **检查类别平衡**：训练数据中正负样本的比例
5. **验证特征工程**：确保所有特征都有合理的数值范围

**立即检查:**
- 训练数据的标签分布
- 特征的标准缩放是否正确
- 模型是否过拟合
""")

# 提供调试信息
st.subheader("📋 调试信息")
st.write("**模型对象类型:**", type(model))
st.write("**模型方法:**", [method for method in dir(model) if not method.startswith('_')])
