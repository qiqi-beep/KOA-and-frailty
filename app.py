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

# 加载所有组件
@st.cache_resource
def load_components():
    try:
        base_path = Path(__file__).parent
        components = {}
        
        # 加载预处理器
        try:
            preprocessor_path = base_path / "frailty_preprocessor28.pkl"
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    components['preprocessor'] = pickle.load(f)
                st.success("✅ 预处理器加载成功")
            else:
                st.error("❌ 预处理器文件不存在")
                return None
        except Exception as e:
            st.error(f"❌ 预处理器加载失败: {str(e)}")
            return None
        
        # 加载模型
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
        
        # 加载预测器
        try:
            predictor_path = base_path / "frailty_predictor28.pkl"
            if predictor_path.exists():
                with open(predictor_path, 'rb') as f:
                    components['predictor'] = pickle.load(f)
                st.success("✅ 预测器加载成功")
            else:
                st.warning("⚠️ 预测器文件不存在，使用默认阈值")
        except Exception as e:
            st.warning(f"⚠️ 预测器加载失败: {str(e)}")
        
        return components
        
    except Exception as e:
        st.error(f"❌ 组件加载失败: {str(e)}")
        return None

components = load_components()

if components is None:
    st.error("无法加载必要的组件，请检查文件是否存在")
    st.stop()

preprocessor = components['preprocessor']
model = components['model']
predictor = components.get('predictor')

# 初始化SHAP解释器
@st.cache_resource
def create_explainer(_model):
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception as e:
        st.warning(f"SHAP解释器创建失败: {str(e)}")
        return None

explainer = create_explainer(model)

# 获取特征名称
def get_feature_names(preprocessor):
    try:
        if hasattr(preprocessor, 'get_feature_names_out'):
            return preprocessor.get_feature_names_out()
        else:
            # 根据您的特征结构生成名称
            return [
                'FTSST', 'bmi', 'age', 'bl_crp', 'bl_hgb',
                'PA_0', 'PA_1', 'PA_2',
                'Complications_0', 'Complications_1', 'Complications_2',
                'fall_0', 'fall_1',
                'ADL_0', 'ADL_1',
                'gender_0', 'gender_1',
                'smoke_0', 'smoke_1'
            ]
    except:
        return [f'feature_{i}' for i in range(19)]  # 假设有19个特征

feature_names = get_feature_names(preprocessor)

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
            
            # 进行预测
            if predictor is not None and hasattr(predictor, 'predict_proba'):
                proba = predictor.predict_proba(input_df)[0, 1]
                prediction = predictor.predict(input_df)[0]
            else:
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
            
            # SHAP可视化
            if explainer is not None:
                try:
                    # 获取SHAP值
                    shap_values = explainer.shap_values(processed_data)
                    
                    # 创建SHAP水平图（force plot）
                    st.subheader("🧠 SHAP决策依据分析")
                    
                    # 设置matplotlib后端为Agg以避免GUI问题
                    plt.switch_backend('Agg')
                    
                    # 创建水平方向的force plot
                    plt.figure(figsize=(12, 4))
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[0],
                        processed_data[0],
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False,
                        plot_cmap="RdBu"
                    )
                    plt.tight_layout()
                    st.pyplot(plt.gcf(), clear_figure=True)
                    plt.close()
                    
                    # 图例说明
                    st.markdown("""
                    **SHAP图例说明:**
                    - 🔴 **红色特征**：增加衰弱风险的因素
                    - 🟢 **绿色特征**：降低衰弱风险的因素  
                    - 📏 **条形长度**：影响程度大小
                    - 📍 **基准值**：平均预测水平
                    - 🎯 **最终值**：当前患者的预测值
                    """)
                    
                    # 显示特征重要性排序
                    st.subheader("📊 特征影响程度排序")
                    
                    # 创建特征重要性DataFrame
                    importance_df = pd.DataFrame({
                        '特征': feature_names,
                        'SHAP值': np.abs(shap_values[0]),
                        '原始值': processed_data[0],
                        '影响方向': ['增加风险' if val > 0 else '降低风险' for val in shap_values[0]]
                    }).sort_values('SHAP值', ascending=False).head(10)
                    
                    # 显示表格
                    st.dataframe(importance_df, use_container_width=True)
                    
                    # 显示条形图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['red' if dir == '增加风险' else 'green' for dir in importance_df['影响方向']]
                    bars = ax.barh(importance_df['特征'], importance_df['SHAP值'], color=colors)
                    ax.set_xlabel('影响程度（绝对值）')
                    ax.set_title('Top 10 特征影响程度')
                    plt.gca().invert_yaxis()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"SHAP可视化失败: {str(e)}")
                    import traceback
                    st.write(traceback.format_exc())
            else:
                st.warning("SHAP解释器不可用，无法显示决策依据分析")
                
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")
            import traceback
            st.write(traceback.format_exc())

# 显示特征映射说明
with st.expander("ℹ️ 特征编码说明"):
    st.write("""
    **特征编码规则:**
    - **性别**: 男=gender_0, 女=gender_1
    - **吸烟**: 否=smoke_0, 是=smoke_1  
    - **跌倒**: 否=fall_0, 是=fall_1
    - **活动水平**: 高=PA_0, 中=PA_1, 低=PA_2
    - **并发症**: 没有=Complications_0, 1个=Complications_1, ≥2个=Complications_2
    - **日常活动**: 无限制=ADL_0, 有限制=ADL_1
    - **坐立测试**: <12s=FTSST=0, ≥12s=FTSST=1
    - **数值特征**: age, bmi, bl_crp, bl_hgb 保持原始值
    """)

# 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")
