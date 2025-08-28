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

# 加载模型和特征名称
@st.cache_resource
def load_model_and_features():
    try:
        base_path = Path(__file__).parent
        model_path = base_path / "frailty_predictor28.pkl"
        feature_path = base_path / "frailty_feature_names.pkl"
        
        # 验证文件
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在于: {model_path}")
        
        # 尝试多种加载方式
        try:
            # 方式1：优先尝试joblib加载
            model = joblib.load(model_path)
        except Exception as e:
            st.warning(f"Joblib加载失败，尝试其他方式: {str(e)}")
            try:
                # 方式2：尝试pickle加载
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                st.warning(f"Pickle加载失败，尝试XGBoost原生加载: {str(e)}")
                try:
                    # 方式3：尝试XGBoost原生加载
                    model = xgb.Booster()
                    model.load_model(str(model_path))
                except Exception as e:
                    raise ValueError(f"所有加载方式均失败: {str(e)}")
        
        # 加载特征名
        with open(feature_path, 'rb') as f:
            feature_names = pickle.load(f)
            
        return model, feature_names
        
    except Exception as e:
        st.error(f"加载失败: {str(e)}")
        st.write("""
        **故障排除步骤:**
        1. 确认模型文件完整
        2. 检查文件格式是否正确
        3. 尝试重新生成模型文件
        """)
        st.write("当前目录内容:", [f.name for f in Path('.').glob('*')])
        return None, None

model, feature_names = load_model_and_features()

if model is None:
    st.stop()

# 初始化SHAP解释器
@st.cache_resource
def create_explainer(_model):
    try:
        return shap.TreeExplainer(_model, model_output="margin")
    except:
        try:
            return shap.TreeExplainer(_model, model_output="raw")
        except:
            return shap.TreeExplainer(_model)

explainer = create_explainer(model)

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
        
        # 根据实际特征名称创建输入数据
        input_data = {
            'FTSST': 1 if sit_stand == "大于等于12s" else 0,
            'bmi': bmi,
            'age': age,
            'bl_crp': crp,
            'bl_hgb': hgb,
            'PA_0': 1 if activity == "高" else 0,
            'PA_1': 1 if activity == "中" else 0,
            'PA_2': 1 if activity == "低" else 0,
            'Complications_0': 1 if complication == "没有" else 0,
            'Complications_1': 1 if complication == "1个" else 0,
            'Complications_2': 1 if complication == "至少2个" else 0,
            'fall_0': 1 if fall == "否" else 0,
            'fall_1': 1 if fall == "是" else 0,
            'ADL_0': 1 if daily_activity == "无限制" else 0,
            'ADL_1': 1 if daily_activity == "有限制" else 0,
            'gender_0': 1 if gender == "男" else 0,
            'gender_1': 1 if gender == "女" else 0,
            'smoke_0': 1 if smoking == "否" else 0,
            'smoke_1': 1 if smoking == "是" else 0
        }
        
        # 创建DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 确保所有特征都存在
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # 重新排序列
        input_df = input_df[feature_names]
        
        try:
            # 兼容不同版本的预测方式
            if hasattr(model, 'predict_proba'):
                # scikit-learn接口的模型
                frail_prob = model.predict_proba(input_df)[0, 1]
            elif hasattr(model, 'predict'):
                # 尝试直接使用numpy数组
                try:
                    raw_pred = model.predict(input_df.values)[0]
                    frail_prob = 1 / (1 + np.exp(-raw_pred))
                except Exception as e:
                    st.warning(f"直接预测失败，尝试其他方法: {str(e)}")
                    # 如果是XGBoost booster，使用DMatrix
                    if hasattr(model, 'feature_names'):
                        dmatrix = xgb.DMatrix(input_df, feature_names=feature_names)
                        raw_pred = model.predict(dmatrix)[0]
                        frail_prob = 1 / (1 + np.exp(-raw_pred))
                    else:
                        raise e
            else:
                st.error("无法识别模型类型")
                st.stop()
            
            # 显示预测结果
            st.success(f"📊 预测结果: 患者衰弱概率为 {frail_prob*100:.2f}%")
            
            # 风险评估
            if frail_prob > 0.8:
                st.error("""⚠️ **高风险：建议立即临床干预**""")
                st.write("- 每周随访监测")
                st.write("- 必须物理治疗干预")
                st.write("- 全面评估并发症")
            elif frail_prob > 0.3:
                st.warning("""⚠️ **中风险：建议定期监测**""")
                st.write("- 每3-6个月评估一次")
                st.write("- 建议适度运动计划")
                st.write("- 基础营养评估")
            else:
                st.success("""✅ **低风险：建议常规健康管理**""")
                st.write("- 每年体检一次")
                st.write("- 保持健康生活方式")
                st.write("- 预防性健康指导")
            
            # SHAP可视化
            try:
                # 获取SHAP值
                if hasattr(explainer, 'shap_values'):
                    shap_values = explainer.shap_values(input_df)
                else:
                    shap_values = explainer(input_df).values
                
                expected_value = explainer.expected_value
                
                # 特征名称映射
                feature_names_mapping = {
                    'age': f'年龄={int(age)}岁',
                    'bmi': f'BMI={bmi:.1f}',
                    'bl_crp': f'CRP={crp:.1f}mg/L',
                    'bl_hgb': f'血红蛋白={hgb:.1f}g/L',
                    'Complications_0': f'并发症={"无" if complication=="没有" else "有"}',
                    'Complications_1': f'并发症={"1个" if complication=="1个" else "其他"}',
                    'Complications_2': f'并发症={"≥2个" if complication=="至少2个" else "其他"}',
                    'FTSST': f'坐立测试={"≥12s" if sit_stand=="大于等于12s" else "<12s"}',
                    'fall_0': f'跌倒={"否" if fall=="否" else "是"}',
                    'fall_1': f'跌倒={"是" if fall=="是" else "否"}',
                    'ADL_0': f'日常活动={"无限制" if daily_activity=="无限制" else "受限"}',
                    'ADL_1': f'日常活动={"受限" if daily_activity=="有限制" else "无限制"}',
                    'gender_0': f'性别={"男" if gender=="男" else "女"}',
                    'gender_1': f'性别={"女" if gender=="女" else "男"}',
                    'PA_0': f'活动水平={"高" if activity=="高" else "中/低"}',
                    'PA_1': f'活动水平={"中" if activity=="中" else "高/低"}',
                    'PA_2': f'活动水平={"低" if activity=="低" else "高/中"}',
                    'smoke_0': f'吸烟={"否" if smoking=="否" else "是"}',
                    'smoke_1': f'吸烟={"是" if smoking=="是" else "否"}'
                }

                # 创建SHAP决策图
                st.subheader(f"🧠 决策依据分析（{'衰弱' if frail_prob > 0.5 else '非衰弱'}类）")
                
                # 使用瀑布图
                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(shap.Explanation(
                    values=shap_values[0], 
                    base_values=expected_value,
                    feature_names=[feature_names_mapping.get(f, f) for f in input_df.columns],
                    data=input_df.iloc[0].values
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
                st.error(f"SHAP可视化失败: {str(e)}")
                
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")
            st.write("调试信息:", {
                "输入数据形状": input_df.shape,
                "特征数量": len(feature_names),
                "输入数据列": list(input_df.columns),
                "模型类型": type(model)
            })

# 页脚
st.markdown("---")
st.caption("©2025 KOA预测系统 | 仅供临床参考")

