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
            return None, None
            
        # 尝试多种加载方式
        try:
            import joblib
            model = joblib.load(model_path)
            st.success("使用 joblib 成功加载模型")
            model_type = "sklearn"
        except Exception as e:
            try:
                model = xgb.Booster()
                model.load_model(str(model_path))
                st.success("使用 XGBoost 原生加载成功")
                model_type = "native"
            except Exception as e:
                st.error(f"所有加载方式均失败: {str(e)}")
                return None, None
                
        # 尝试加载特征名称
        feature_path = base_path / "frailty_feature_names.pkl"
        if feature_path.exists():
            with open(feature_path, 'rb') as f:
                feature_names = pickle.load(f)
            st.success("成功加载特征名称")
        else:
            st.warning("特征名称文件不存在，将使用默认特征顺序")
            feature_names = None
            
        return model, model_type, feature_names
                
    except Exception as e:
        st.error(f"加载失败: {str(e)}")
        return None, None, None

model, model_type, feature_names = load_model_for_diagnosis()

if model is None:
    st.stop()

# 模型基本信息
st.subheader("📊 模型基本信息")

st.write(f"**模型类型:** {'Scikit-learn 接口' if model_type == 'sklearn' else '原生 XGBoost'}")

if model_type == "sklearn":
    # scikit-learn 接口的模型
    try:
        params = model.get_params()
        st.write("**模型参数:**")
        st.json(params)
    except:
        st.write("无法获取模型参数")
    
    # 获取特征重要性
    try:
        importance = model.feature_importances_
        st.write("**特征重要性:**", importance)
    except:
        st.write("无法获取特征重要性")
        
else:
    # 原生 XGBoost 模型
    st.write("**原生 XGBoost 模型**")
    try:
        # 获取模型信息
        num_features = model.num_features()
        num_trees = model.num_trees()
        st.write(f"**特征数量:** {num_features}")
        st.write(f"**树的数量:** {num_trees}")
        
        # 获取一些配置信息
        config = model.save_config()
        # 提取一些关键信息
        if "objective" in config:
            objective_start = config.find("objective") + 12
            objective_end = config.find('"', objective_start)
            objective = config[objective_start:objective_end]
            st.write(f"**目标函数:** {objective}")
            
    except Exception as e:
        st.write(f"无法获取完整模型信息: {str(e)}")

# 创建测试数据来验证模型行为
st.subheader("🧪 模型行为测试")

# 定义特征名称（如果文件不存在）
if feature_names is None:
    feature_names = [
        'gender', 'age', 'smoking', 'bmi', 'fall', 'PA_high', 'PA_medium', 'PA_low',
        'Complications_0', 'Complications_1', 'Complications_2', 'ADL', 'FTSST',
        'bl_crp', 'bl_hgb'
    ]
    st.warning("使用默认特征名称，可能与实际模型不匹配")

# 创建一些典型的测试案例
test_cases = [
    {
        "描述": "健康年轻患者", 
        "年龄": 40, "BMI": 22.0, "CRP": 5.0, "血红蛋白": 130.0, 
        "跌倒": 0, "活动水平": "高", "性别": "男", "吸烟": "否",
        "并发症": "没有", "日常活动": "无限制", "坐立测试": "小于12s"
    },
    {
        "描述": "典型老年患者", 
        "年龄": 70, "BMI": 26.0, "CRP": 15.0, "血红蛋白": 110.0, 
        "跌倒": 0, "活动水平": "中", "性别": "女", "吸烟": "否",
        "并发症": "1个", "日常活动": "无限制", "坐立测试": "大于等于12s"
    },
    {
        "描述": "高风险患者", 
        "年龄": 80, "BMI": 30.0, "CRP": 50.0, "血红蛋白": 90.0, 
        "跌倒": 1, "活动水平": "低", "性别": "女", "吸烟": "是",
        "并发症": "至少2个", "日常活动": "有限制", "坐立测试": "大于等于12s"
    },
]

results = []

for case in test_cases:
    # 构建完整的输入数据
    input_data = {
        'gender': 1 if case["性别"] == "女" else 0,
        'age': case["年龄"],
        'smoking': 1 if case["吸烟"] == "是" else 0,
        'bmi': case["BMI"],
        'fall': case["跌倒"],
        'PA_high': 1 if case["活动水平"] == "高" else 0,
        'PA_medium': 1 if case["活动水平"] == "中" else 0,
        'PA_low': 1 if case["活动水平"] == "低" else 0,
        'Complications_0': 1 if case["并发症"] == "没有" else 0,
        'Complications_1': 1 if case["并发症"] == "1个" else 0,
        'Complications_2': 1 if case["并发症"] == "至少2个" else 0,
        'ADL': 1 if case["日常活动"] == "有限制" else 0,
        'FTSST': 1 if case["坐立测试"] == "大于等于12s" else 0,
        'bl_crp': case["CRP"],
        'bl_hgb': case["血红蛋白"]
    }
    
    # 创建 DataFrame 并确保特征顺序正确
    input_df = pd.DataFrame([input_data])
    
    # 确保所有特征都存在
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[feature_names]
    
    # 进行预测
    try:
        if model_type == "sklearn":
            # scikit-learn 接口
            proba = model.predict_proba(input_df)[0][1]
            raw_pred = model.predict(input_df)[0]
        else:
            # 原生 XGBoost
            dmatrix = xgb.DMatrix(input_df, feature_names=feature_names)
            raw_pred = model.predict(dmatrix)[0]
            proba = 1 / (1 + np.exp(-raw_pred))
        
        results.append({
            "案例": case["描述"],
            "原始预测值": float(raw_pred),
            "概率": float(proba),
            **case
        })
        
    except Exception as e:
        st.error(f"测试案例 '{case['描述']}' 预测失败: {str(e)}")

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
            st.write(f"性别: {result['性别']}")
            st.write(f"吸烟: {result['吸烟']}")
            st.write(f"并发症: {result['并发症']}")

    # 模型校准检查
    st.subheader("⚖️ 模型校准检查")
    
    probabilities = [r['概率'] for r in results]
    avg_prob = np.mean(probabilities)
    
    st.write(f"**平均预测概率:** {avg_prob:.4f}")
    st.write(f"**概率范围:** {min(probabilities):.4f} - {max(probabilities):.4f}")
    
    if avg_prob > 0.8:
        st.error("⚠️ 警告：模型可能过于悲观，平均预测概率偏高")
    elif avg_prob < 0.2:
        st.error("⚠️ 警告：模型可能过于乐观，平均预测概率偏低")
    else:
        st.success("✅ 平均预测概率在合理范围内")

# 调试信息
st.subheader("🐛 调试信息")
st.write("请分享以下信息以便进一步分析:")
st.write(f"- 模型类型: {model_type}")
st.write(f"- 测试案例数量: {len(results)}")
if results:
    st.write(f"- 概率范围: {min([r['概率'] for r in results]):.4f} - {max([r['概率'] for r in results]):.4f}")

# 建议
st.subheader("💡 建议")
st.markdown("""
如果概率不符合常理，可能的原因：
1. **训练数据问题**：正负样本极度不平衡
2. **特征缩放**：数值特征未进行标准化
3. **模型过拟合**：在训练集上表现太好，测试集表现差
4. **目标函数问题**：可能需要调整模型参数

**请检查:**
- 训练数据的标签分布比例
- 数值特征（年龄、BMI、CRP、血红蛋白）的分布范围
- 模型是否使用了正确的目标函数（如 binary:logistic）
""")
