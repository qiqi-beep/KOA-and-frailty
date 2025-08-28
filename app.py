import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import joblib

# 页面设置
st.set_page_config(page_title="KOA 患者衰弱风险预测", layout="centered")
st.title("🩺 膝骨关节炎患者衰弱风险预测系统")
st.markdown("根据输入的临床特征，预测膝关节骨关节炎（KOA）患者发生衰弱（Frailty）的概率。")

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
            st.write("当前目录文件:", [f.name for f in base_path.glob('*')])
            return None
            
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

model = load_model()

if model is None:
    st.error("无法加载模型，请检查文件是否存在")
    st.stop()

# 直接构建模型所需的特征数组
def build_feature_array(input_data):
    """
    直接按照模型训练时的特征顺序构建数组：
    'FTSST', 'bmi', 'age', 'bl_crp', 'bl_hgb', 
    'PA_0', 'PA_1', 'PA_2', 'Complications_0', 'Complications_1', 'Complications_2', 
    'fall_0', 'fall_1', 'ADL_0', 'ADL_1', 'gender_0', 'gender_1', 'smoke_0', 'smoke_1'
    """
    features = []
    
    # 1. FTSST (坐立测试时间)
    features.append(1 if input_data['sit_stand'] == "大于等于12s" else 0)
    
    # 2-5. 数值特征
    features.append(input_data['bmi'])
    features.append(input_data['age'])
    features.append(input_data['crp'])
    features.append(input_data['hgb'])
    
    # 6-8. PA (体力活动水平)
    activity = input_data['activity']
    features.append(1 if activity == "高" else 0)
    features.append(1 if activity == "中" else 0)
    features.append(1 if activity == "低" else 0)
    
    # 9-11. Complications (并发症)
    complication = input_data['complication']
    features.append(1 if complication == "没有" else 0)
    features.append(1 if complication == "1个" else 0)
    features.append(1 if complication == "至少2个" else 0)
    
    # 12-13. fall (跌倒史)
    fall = input_data['fall']
    features.append(1 if fall == "否" else 0)
    features.append(1 if fall == "是" else 0)
    
    # 14-15. ADL (日常活动能力)
    daily_activity = input_data['daily_activity']
    features.append(1 if daily_activity == "无限制" else 0)
    features.append(1 if daily_activity == "有限制" else 0)
    
    # 16-17. gender (性别)
    gender = input_data['gender']
    features.append(1 if gender == "男" else 0)
    features.append(1 if gender == "女" else 0)
    
    # 18-19. smoke (吸烟)
    smoking = input_data['smoking']
    features.append(1 if smoking == "否" else 0)
    features.append(1 if smoking == "是" else 0)
    
    return np.array([features])

# 特征名称
feature_names = [
    'FTSST', 'bmi', 'age', 'bl_crp', 'bl_hgb',
    'PA_0', 'PA_1', 'PA_2',
    'Complications_0', 'Complications_1', 'Complications_2',
    'fall_0', 'fall_1',
    'ADL_0', 'ADL_1',
    'gender_0', 'gender_1',
    'smoke_0', 'smoke_1'
]

# 特征描述（用于显示）
feature_descriptions = [
    '坐立测试时间(≥12s=1)', 'BMI值', '年龄', 'C反应蛋白值', '血红蛋白值',
    '活动水平-高', '活动水平-中', '活动水平-低',
    '无并发症', '1个并发症', '≥2个并发症',
    '无跌倒史', '有跌倒史',
    '日常活动无限制', '日常活动受限',
    '男性', '女性',
    '不吸烟', '吸烟'
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
        
        # 构建输入数据
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
        
        try:
            # 构建特征数组
            feature_array = build_feature_array(input_data)
            
            # 进行预测
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_array)[0, 1]
            else:
                raw_pred = model.predict(feature_array)[0]
                proba = 1 / (1 + np.exp(-raw_pred))
            
            # 使用0.57作为阈值
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
            
            # 显示特征值
            st.subheader("📋 输入特征值")
            feature_values = feature_array[0]
            
            # 创建特征表格
            feature_df = pd.DataFrame({
                '特征': feature_descriptions,
                '值': feature_values,
                '类型': ['分类'] * 5 + ['数值'] * 14  # 前5个是数值，后面是分类
            })
            
            st.dataframe(feature_df, use_container_width=True)
            
            # 临床风险因素分析
            st.subheader("🧪 临床风险因素分析")
            
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
            st.write(traceback.format_exc())

# 特征说明
with st.expander("ℹ️ 特征说明"):
    st.write("""
    **模型使用的19个特征:**
    1. **FTSST**: 坐立测试时间 (≥12s=1, <12s=0)
    2. **bmi**: 体重指数
    3. **age**: 年龄
    4. **bl_crp**: C反应蛋白值
    5. **bl_hgb**: 血红蛋白值
    6. **PA_0**: 活动水平-高
    7. **PA_1**: 活动水平-中  
    8. **PA_2**: 活动水平-低
    9. **Complications_0**: 无并发症
    10. **Complications_1**: 1个并发症
    11. **Complications_2**: ≥2个并发症
    12. **fall_0**: 无跌倒史
    13. **fall_1**: 有跌倒史
    14. **ADL_0**: 日常活动无限制
    15. **ADL_1**: 日常活动受限
    16. **gender_0**: 男性
    17. **gender_1**: 女性
    18. **smoke_0**: 不吸烟
    19. **smoke_1**: 吸烟
    """)

# 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")
