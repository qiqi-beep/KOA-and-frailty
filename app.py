import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import joblib
import sys

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

# 加载模型（跳过预处理器）
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
            st.write("当前目录文件:", [f.name for f in base_path.glob('*')])
            return None
            
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

model = load_model()

if model is None:
    st.error("无法加载模型，请检查文件是否存在")
    st.stop()

# 手动预处理函数 - 按照您提供的特征顺序
def manual_preprocess(input_df):
    """
    手动预处理函数，按照模型训练时的特征顺序
    特征顺序: 'FTSST', 'bmi', 'age', 'bl_crp', 'bl_hgb', 'PA_0', 'PA_1', 'PA_2', 
             'Complications_0', 'Complications_1', 'Complications_2', 'fall_0', 'fall_1', 
             'ADL_0', 'ADL_1', 'gender_0', 'gender_1', 'smoke_0', 'smoke_1'
    """
    processed_data = []
    
    # 1. FTSST (坐立测试时间)
    ftsst_val = 1 if input_df['sit_stand'].iloc[0] == "大于等于12s" else 0
    processed_data.append(ftsst_val)
    
    # 2. bmi (直接使用)
    processed_data.append(input_df['bmi'].iloc[0])
    
    # 3. age (直接使用)
    processed_data.append(input_df['age'].iloc[0])
    
    # 4. bl_crp (CRP值)
    processed_data.append(input_df['crp'].iloc[0])
    
    # 5. bl_hgb (血红蛋白值)
    processed_data.append(input_df['hgb'].iloc[0])
    
    # 6-8. PA_0, PA_1, PA_2 (体力活动水平)
    activity = input_df['activity'].iloc[0]
    pa_0 = 1 if activity == "高" else 0
    pa_1 = 1 if activity == "中" else 0
    pa_2 = 1 if activity == "低" else 0
    processed_data.extend([pa_0, pa_1, pa_2])
    
    # 9-11. Complications_0, Complications_1, Complications_2 (并发症)
    complication = input_df['complication'].iloc[0]
    comp_0 = 1 if complication == "没有" else 0
    comp_1 = 1 if complication == "1个" else 0
    comp_2 = 1 if complication == "至少2个" else 0
    processed_data.extend([comp_0, comp_1, comp_2])
    
    # 12-13. fall_0, fall_1 (跌倒史)
    fall = input_df['fall'].iloc[0]
    fall_0 = 1 if fall == "否" else 0
    fall_1 = 1 if fall == "是" else 0
    processed_data.extend([fall_0, fall_1])
    
    # 14-15. ADL_0, ADL_1 (日常活动能力)
    daily_activity = input_df['daily_activity'].iloc[0]
    adl_0 = 1 if daily_activity == "无限制" else 0
    adl_1 = 1 if daily_activity == "有限制" else 0
    processed_data.extend([adl_0, adl_1])
    
    # 16-17. gender_0, gender_1 (性别)
    gender = input_df['gender'].iloc[0]
    gender_0 = 1 if gender == "男" else 0
    gender_1 = 1 if gender == "女" else 0
    processed_data.extend([gender_0, gender_1])
    
    # 18-19. smoke_0, smoke_1 (吸烟)
    smoking = input_df['smoking'].iloc[0]
    smoke_0 = 1 if smoking == "否" else 0
    smoke_1 = 1 if smoking == "是" else 0
    processed_data.extend([smoke_0, smoke_1])
    
    return np.array([processed_data])

# 获取特征名称
def get_feature_names():
    return [
        'FTSST', 'bmi', 'age', 'bl_crp', 'bl_hgb',
        'PA_0', 'PA_1', 'PA_2',
        'Complications_0', 'Complications_1', 'Complications_2',
        'fall_0', 'fall_1',
        'ADL_0', 'ADL_1',
        'gender_0', 'gender_1',
        'smoke_0', 'smoke_1'
    ]

# 简单的特征重要性计算（替代SHAP）
def calculate_feature_importance(model, processed_data, feature_names):
    """计算简单的特征重要性"""
    try:
        # 如果是XGBoost模型，使用内置的特征重要性
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            # 使用随机扰动法计算重要性
            base_pred = model.predict_proba(processed_data)[0, 1] if hasattr(model, 'predict_proba') else model.predict(processed_data)[0]
            importances = []
            
            for i in range(len(feature_names)):
                perturbed_data = processed_data.copy()
                perturbed_data[0, i] = 0  # 将该特征设为0
                perturbed_pred = model.predict_proba(perturbed_data)[0, 1] if hasattr(model, 'predict_proba') else model.predict(perturbed_data)[0]
                importance = abs(base_pred - perturbed_pred)
                importances.append(importance)
            
            return np.array(importances)
    except:
        # 如果都失败，返回均匀分布的重要性
        return np.ones(len(feature_names)) / len(feature_names)

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
            feature_names = get_feature_names()
            
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
            else:
                st.success("""✅ **低风险：建议常规健康管理**""")
                st.write("- 每年体检一次")
                st.write("- 保持健康生活方式")
                st.write("- 预防性健康指导")
            
            # 计算特征重要性
            st.subheader("📈 特征影响分析")
            
            feature_importance = calculate_feature_importance(model, processed_data, feature_names)
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                '特征': feature_names,
                '重要性': feature_importance,
                '原始值': processed_data[0],
                '特征描述': [
                    '坐立测试时间(≥12s=1)', 'BMI', '年龄', 'C反应蛋白', '血红蛋白',
                    '活动水平-高', '活动水平-中', '活动水平-低',
                    '无并发症', '1个并发症', '≥2个并发症',
                    '无跌倒史', '有跌倒史',
                    '日常活动无限制', '日常活动受限',
                    '男性', '女性',
                    '不吸烟', '吸烟'
                ]
            }).sort_values('重要性', ascending=False)
            
            # 显示前10个最重要特征
            top_features = importance_df.head(10)
            
            # 创建水平条形图
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['red' if val > 0 else 'green' for val in top_features['原始值']]
            bars = ax.barh(top_features['特征描述'], top_features['重要性'], color=colors)
            ax.set_xlabel('相对重要性')
            ax.set_title('Top 10 特征影响程度')
            plt.gca().invert_yaxis()
            st.pyplot(fig)
            
            # 显示详细表格
            with st.expander("查看详细特征信息"):
                st.dataframe(importance_df, use_container_width=True)
            
            # 显示临床解读
            st.subheader("🧪 临床解读")
            
            # 根据特征值给出解读
            risk_factors = []
            if processed_data[0, 0] == 1:  # FTSST
                risk_factors.append("坐立测试时间较长(≥12s)")
            if processed_data[0, 1] > 28:  # BMI
                risk_factors.append(f"BMI较高({processed_data[0, 1]:.1f})")
            if processed_data[0, 2] > 70:  # 年龄
                risk_factors.append(f"高龄({int(processed_data[0, 2])}岁)")
            if processed_data[0, 3] > 5:   # CRP
                risk_factors.append(f"炎症指标较高({processed_data[0, 3]:.1f}mg/L)")
            if processed_data[0, 4] < 120: # 血红蛋白
                risk_factors.append(f"血红蛋白较低({processed_data[0, 4]:.1f}g/L)")
            if processed_data[0, 8] == 0:  # 有并发症
                risk_factors.append("存在并发症")
            if processed_data[0, 12] == 0: # 有跌倒史
                risk_factors.append("近期有跌倒史")
            if processed_data[0, 14] == 1: # 日常活动受限
                risk_factors.append("日常活动受限")
            if processed_data[0, 19] == 1: # 吸烟
                risk_factors.append("吸烟")
            
            if risk_factors:
                st.write("**识别到的风险因素:**")
                for factor in risk_factors:
                    st.write(f"⚠️ {factor}")
            else:
                st.write("✅ 未识别到明显风险因素")
                
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")
            import traceback
            st.write(traceback.format_exc())

# 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")
