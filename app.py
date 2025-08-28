import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import time
from pathlib import Path
import joblib

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

# 手动预处理函数 - 调整特征方向
def manual_preprocess(input_df):
    # 数值特征标准化（使用训练数据的典型范围）
    # 注意：对于负面影响的特征（如年龄、CRP），我们保持正值
    numeric_stats = {
        'age': {'mean': 65, 'std': 10},
        'bmi': {'mean': 25, 'std': 4},
        'crp': {'mean': 5, 'std': 3},
        'hgb': {'mean': 130, 'std': 15}
    }
    
    processed_data = []
    
    # 处理数值特征 - 增加衰弱风险的特征
    # 年龄越大，风险越高
    age_val = input_df['age'].iloc[0]
    age_norm = (age_val - numeric_stats['age']['mean']) / numeric_stats['age']['std']
    processed_data.append(age_norm)
    
    # BMI越高，风险越高
    bmi_val = input_df['bmi'].iloc[0]
    bmi_norm = (bmi_val - numeric_stats['bmi']['mean']) / numeric_stats['bmi']['std']
    processed_data.append(bmi_norm)
    
    # CRP越高，风险越高
    crp_val = input_df['crp'].iloc[0]
    crp_norm = (crp_val - numeric_stats['crp']['mean']) / numeric_stats['crp']['std']
    processed_data.append(crp_norm)
    
    # 血红蛋白越低，风险越高（所以取负值）
    hgb_val = input_df['hgb'].iloc[0]
    hgb_norm = -(hgb_val - numeric_stats['hgb']['mean']) / numeric_stats['hgb']['std']
    processed_data.append(hgb_norm)
    
    # 处理分类特征 - 调整方向使负面情况对应更高风险
    
    # gender_0, gender_1 (女性风险通常更高)
    gender_val = 0 if input_df['gender'].iloc[0] == '男' else 1
    processed_data.extend([1 - gender_val, gender_val])  # 女性=1，风险更高
    
    # smoke_0, smoke_1 (吸烟风险更高)
    smoke_val = 0 if input_df['smoking'].iloc[0] == '否' else 1
    processed_data.extend([1 - smoke_val, smoke_val])  # 吸烟=1，风险更高
    
    # fall_0, fall_1 (跌倒风险更高)
    fall_val = 0 if input_df['fall'].iloc[0] == '否' else 1
    processed_data.extend([1 - fall_val, fall_val])  # 跌倒=1，风险更高
    
    # PA_0, PA_1, PA_2 (活动水平越低风险越高)
    activity_map = {'高': 0, '中': 1, '低': 2}
    activity_val = activity_map[input_df['activity'].iloc[0]]
    pa_features = [0, 0, 0]
    pa_features[activity_val] = 1
    processed_data.extend(pa_features)  # 低活动水平=1，风险更高
    
    # Complications_0, Complications_1, Complications_2 (并发症越多风险越高)
    complication_map = {'没有': 0, '1个': 1, '至少2个': 2}
    complication_val = complication_map[input_df['complication'].iloc[0]]
    comp_features = [0, 0, 0]
    comp_features[complication_val] = 1
    processed_data.extend(comp_features)  # 更多并发症=1，风险更高
    
    # ADL_0, ADL_1 (日常活动受限风险更高)
    adl_val = 0 if input_df['daily_activity'].iloc[0] == '无限制' else 1
    processed_data.extend([1 - adl_val, adl_val])  # 受限=1，风险更高
    
    # FTSST (坐立时间越长风险越高)
    ftsst_val = 0 if input_df['sit_stand'].iloc[0] == '小于12s' else 1
    processed_data.append(ftsst_val)  # 时间越长=1，风险更高
    
    return np.array([processed_data])

# 获取特征名称
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
        age = st.number_input("您的年龄（岁）", min_value=0, max_value=120, value=75)
        smoking = st.radio("您是否吸烟？", ["否", "是"])
        bmi = st.number_input("BMI（kg/m²）", min_value=10.0, max_value=50.0, value=28.0, step=0.1)
        fall = st.radio("过去一年是否跌倒？", ["否", "是"])
    
    with col2:
        activity = st.radio("体力活动水平", ["高", "中", "低"])
        complication = st.radio("并发症数量", ["没有", "1个", "至少2个"])
        daily_activity = st.radio("日常生活能力", ["无限制", "有限制"])
        sit_stand = st.radio("5次坐立时间", ["小于12s", "大于等于12s"])
        crp = st.number_input("C反应蛋白（mg/L）", min_value=0.0, max_value=100.0, value=8.0, step=0.1)
        hgb = st.number_input("血红蛋白（g/L）", min_value=0.0, max_value=200.0, value=115.0, step=0.1)
        
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
            
            # 进行预测 - 反转概率方向
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(processed_data)[0, 1]
                # 反转概率：1 - 原始概率
                frail_prob = 1 - proba
                prediction = 1 if frail_prob >= 0.57 else 0
            else:
                raw_pred = model.predict(processed_data)[0]
                proba = 1 / (1 + np.exp(-raw_pred))
                # 反转概率：1 - 原始概率
                frail_prob = 1 - proba
                prediction = 1 if frail_prob >= 0.57 else 0
            
            # 显示预测结果
            st.success(f"📊 预测结果: 患者衰弱概率为 {frail_prob*100:.2f}%")
            
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
            
            # 显示临床解释
            st.subheader("🧪 临床特征分析")
            
            risk_factors = []
            if age > 70:
                risk_factors.append(f"👴 高龄 ({age}岁)")
            if bmi > 28:
                risk_factors.append(f"⚖️ 高BMI ({bmi:.1f})")
            if crp > 5:
                risk_factors.append(f"🔥 高炎症指标CRP ({crp:.1f}mg/L)")
            if hgb < 120:
                risk_factors.append(f"🩸 低血红蛋白 ({hgb:.1f}g/L)")
            if smoking == "是":
                risk_factors.append("🚬 吸烟")
            if fall == "是":
                risk_factors.append("⚠️ 近期跌倒史")
            if activity == "低":
                risk_factors.append("🏃 低体力活动")
            if complication != "没有":
                risk_factors.append(f"🩺 {complication}并发症")
            if daily_activity == "有限制":
                risk_factors.append("🧓 日常活动受限")
            if sit_stand == "大于等于12s":
                risk_factors.append("⏱️ 坐立测试时间较长")
            
            if risk_factors:
                st.write("**识别到的风险因素:**")
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.write("**未识别到明显风险因素** ✅")
                
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")
            import traceback
            st.write("详细错误信息:", traceback.format_exc())

# 显示系统信息
with st.expander("ℹ️ 系统信息"):
    st.write(f"**模型类型:** {type(model).__name__}")
    st.write(f"**特征数量:** {len(get_feature_names())}")
    st.write(f"**预测阈值:** 0.57")
    st.write(f"**概率方向:** 已调整（高风险对应高概率）")
    st.write("**使用内置预处理:** ✅ 是")

# 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")
