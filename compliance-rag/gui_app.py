# web_app.py
import streamlit as st
import pandas as pd
from src.rag_engine import ComplianceRAGEngine
import tempfile
import os
import sys
# === 1. 安全添加项目根目录 ===
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# 设置页面配置
st.set_page_config(
    page_title="金融合规审查系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 RAG 引擎
@st.cache_resource
def load_engine():
    try:
        # 设置 DashScope API Key
        os.environ["DASHSCOPE_API_KEY"] = "sk-a677631fd47a4e2184b6836f6097f0b5"
        engine = ComplianceRAGEngine()
        return engine
    except Exception as e:
        st.error(f"引擎初始化失败: {str(e)}")
        return None

def main():
    st.title("🔍 金融合规审查系统")
    st.markdown("基于AI的金融营销话术合规性自动审查")
    st.markdown("---")
    
    # 加载引擎
    engine = load_engine()
    if engine is None:
        st.stop()
    
    # 侧边栏
    st.sidebar.title("导航")
    app_mode = st.sidebar.selectbox(
        "选择功能",
        ["单条文本分析", "批量文件分析", "测试用例演示", "误判案例验证"]
    )
    
    if app_mode == "单条文本分析":
        single_text_analysis(engine)
    elif app_mode == "批量文件分析":
        batch_file_analysis(engine)
    elif app_mode == "测试用例演示":
        demo_analysis(engine)
    elif app_mode == "误判案例验证":
        false_positive_validation(engine)

def single_text_analysis(engine):
    st.header("单条文本分析")
    
    # 文本输入区域
    text_input = st.text_area(
        "输入待审查文本:",
        placeholder="请输入需要合规审查的文本内容...",
        height=150
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_btn = st.button("开始分析", type="primary", use_container_width=True)
    
    if analyze_btn and text_input.strip():
        with st.spinner("正在分析中..."):
            result = engine.predict(text_input.strip())
            
        # 显示结果
        st.markdown("### 📊 分析结果")
        
        # 使用列布局显示主要结果
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if result["violation"]:
                st.error("❌ 违规")
            else:
                st.success("✅ 合规")
                
        with col2:
            st.metric("触发事件", result["triggered_event"])
            
        with col3:
            if result.get('pre_check_used', False):
                st.metric("分析方式", "预检查")
            else:
                st.metric("分析方式", "深度分析")
                
        with col4:
            confidence = result.get('confidence', '中')
            if confidence == '高':
                st.metric("置信度", "🔴 高")
            elif confidence == '中':
                st.metric("置信度", "🟡 中")
            else:
                st.metric("置信度", "🟢 低")
        
        # 详细理由
        st.markdown("### 📝 分析理由")
        if result["violation"]:
            st.error(result["reason"])
        else:
            st.success(result["reason"])
        
        # 上下文分析（如果可用）
        if result.get('context_analysis'):
            st.markdown("### 🔍 上下文分析")
            context = result['context_analysis']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("服务方发言", "是" if context.get('is_service_party') else "否")
            with col2:
                st.metric("营销语境", "是" if context.get('is_marketing_context') else "否")
            with col3:
                st.metric("历史业绩", "是" if context.get('contains_historical_performance') else "否")
            with col4:
                st.metric("风险提示", "是" if context.get('contains_risk_disclaimer') else "否")
        
        # 原始响应
        with st.expander("查看原始响应"):
            st.text(result["raw_response"])

def batch_file_analysis(engine):
    st.header("批量文件分析")
    
    uploaded_file = st.file_uploader(
        "上传文本文件",
        type=['txt', 'csv'],
        help="支持TXT和CSV文件，TXT文件每行一条，CSV文件需包含'text'列"
    )
    
    if uploaded_file is not None:
        # 读取文件内容
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("CSV文件必须包含'text'列")
                return
            lines = df['text'].dropna().tolist()
        else:
            content = uploaded_file.read().decode('utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        st.success(f"成功读取文件，共 {len(lines)} 条内容")
        
        # 显示前几条内容预览
        with st.expander("预览前5条内容"):
            for i, line in enumerate(lines[:5]):
                st.write(f"{i+1}. {line}")
        
        if st.button("开始批量分析", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, line in enumerate(lines):
                status_text.text(f"正在分析第 {i+1}/{len(lines)} 条: {line[:50]}...")
                result = engine.predict(line)
                results.append({
                    '内容': line,
                    '合规状态': '违规' if result['violation'] else '合规',
                    '触发事件': result["triggered_event"],
                    '分析方式': '预检查' if result.get('pre_check_used', False) else '深度分析',
                    '理由': result["reason"]
                })
                progress_bar.progress((i + 1) / len(lines))
            
            status_text.text("分析完成！")
            
            # 显示结果表格
            df_results = pd.DataFrame(results)
            st.markdown("### 📋 分析结果汇总")
            st.dataframe(df_results, use_container_width=True)
            
            # 统计信息
            st.markdown("### 📈 统计分析")
            col1, col2, col3, col4, col5 = st.columns(5)
            total_count = len(results)
            violation_count = sum(1 for r in results if r['合规状态'] == '违规')
            compliant_count = total_count - violation_count
            pre_check_count = sum(1 for r in results if r['分析方式'] == '预检查')
            
            col1.metric("总条目", total_count)
            col2.metric("违规数量", violation_count)
            col3.metric("合规数量", compliant_count)
            col4.metric("预检查数量", pre_check_count)
            col5.metric("违规率", f"{(violation_count/total_count*100):.1f}%")
            
            # 可视化
            col1, col2 = st.columns(2)
            with col1:
                status_counts = df_results['合规状态'].value_counts()
                st.bar_chart(status_counts)
                
            with col2:
                method_counts = df_results['分析方式'].value_counts()
                st.bar_chart(method_counts)
            
            # 提供下载
            csv = df_results.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载分析结果 (CSV)",
                data=csv,
                file_name="合规分析结果.csv",
                mime="text/csv"
            )

def demo_analysis(engine):
    st.header("测试用例演示")
    
    # 预定义的测试用例 - 包含已知的误判案例
    test_cases = [
        {
            "text": "这款产品稳赚不赔，年化收益保底8%！",
            "expected": "违规",
            "description": "明确收益承诺 - 应该违规"
        },
        {
            "text": "王大哥之前自己做的时候，一个月亏损10万+，2月这么好的行情只赚了3万块钱，接触曾老师后，相信曾老师能力 2月28日果断跟上曾老师实战班，#3月已经赚17万了",
            "expected": "合规", 
            "description": "历史业绩展示 - 应该合规"
        },
        {
            "text": "客户问：能保证赚钱吗？我们回答：不能，历史业绩不代表未来收益。",
            "expected": "合规",
            "description": "客户询问回应 - 应该合规"
        },
        {
            "text": "在吗？今日福利股请查收?首席福利内参好票额外申请，是您购买服务以外的额外福利",
            "expected": "合规",
            "description": "服务邀约 - 应该合规"
        },
        {
            "text": "你微信号多少？我加你个人微信发策略。",
            "expected": "违规",
            "description": "私下联系 - 应该违规"
        },
        {
            "text": "策略会通过官方微信服务号推送，请关注",
            "expected": "合规",
            "description": "官方渠道 - 应该合规"
        },
        {
            "text": "10万本金，轻松赚7万！马上加入！",
            "expected": "违规",
            "description": "低投入高回报 - 应该违规"
        },
        {
            "text": "学员张大哥10万本金赚了7万，投资有风险入市需谨慎",
            "expected": "合规",
            "description": "客户案例含风险提示 - 应该合规"
        }
    ]
    
    st.info("以下是一些预定义的测试用例，包含已知的误判案例验证")
    
    if st.button("运行测试用例", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, case in enumerate(test_cases):
            status_text.text(f"正在分析第 {i+1}/{len(test_cases)} 条...")
            result = engine.predict(case["text"])
            
            # 检查预测是否正确
            predicted_status = "违规" if result['violation'] else "合规"
            is_correct = predicted_status == case["expected"]
            
            results.append({
                '测试用例': case["text"][:50] + "...",
                '描述': case["description"],
                '预期结果': case["expected"],
                '实际结果': predicted_status,
                '是否正确': "✅" if is_correct else "❌",
                '触发事件': result["triggered_event"],
                '分析方式': '预检查' if result.get('pre_check_used', False) else '深度分析',
                '理由': result["reason"]
            })
            progress_bar.progress((i + 1) / len(test_cases))
        
        status_text.text("分析完成！")
        
        # 显示结果
        df = pd.DataFrame(results)
        st.markdown("### 🧪 测试用例分析结果")
        st.dataframe(df, use_container_width=True)
        
        # 统计准确率
        correct_count = sum(1 for r in results if r['是否正确'] == '✅')
        accuracy = correct_count / len(test_cases) * 100
        
        st.markdown("### 📊 性能统计")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总测试用例", len(test_cases))
        col2.metric("正确预测", correct_count)
        col3.metric("错误预测", len(test_cases) - correct_count)
        col4.metric("准确率", f"{accuracy:.1f}%")
        
        # 错误分析
        incorrect_cases = [r for r in results if r['是否正确'] == '❌']
        if incorrect_cases:
            st.markdown("### ⚠️ 错误案例分析")
            for case in incorrect_cases:
                st.error(f"**用例**: {case['测试用例']}")
                st.error(f"**预期**: {case['预期结果']}, **实际**: {case['实际结果']}")
                st.error(f"**理由**: {case['理由']}")
                st.markdown("---")

def false_positive_validation(engine):
    st.header("误判案例验证")
    
    st.info("""
    此功能专门用于验证历史上容易误判的案例。
    系统会重点测试那些之前被错误标记为违规的合规案例。
    """)
    
    # 已知的误判案例
    false_positive_cases = [
        {
            "text": "王大哥之前自己做的时候，一个月亏损10万+，2月这么好的行情只赚了3万块钱，接触曾老师后，相信曾老师能力 2月28日果断跟上曾老师实战班，#3月已经赚17万了",
            "correct_reason": "历史业绩展示，非服务方主动承诺"
        },
        {
            "text": "在吗？今日福利股请查收?首席福利内参好票额外申请，是您购买服务以外的额外福利",
            "correct_reason": "仅为服务邀约，无收益承诺"
        },
        {
            "text": "大盘如期反弹，昨晚提醒今天反弹，#你手中有被套十个点的股嘛？有的话把代码发到我微信上，我找老师帮您看看，需不需对冲解套",
            "correct_reason": "官方服务微信，非个人私下联系"
        },
        {
            "text": "对你好的事情小王一直在做的，感受到老师的实力了吧[红包]你看3月13日单独提醒您的【300430诚益通】今天再次大涨10%",
            "correct_reason": "历史业绩回顾，非未来收益承诺"
        },
        {
            "text": "重要的转折窗口期，明天你一定要注意节奏[太阳]华为大会在即，新风口机会绝佳配置时机不容你忽视",
            "correct_reason": "市场分析展望，非个股确定性判断"
        }
    ]
    
    if st.button("验证误判案例", type="primary"):
        results = []
        
        for i, case in enumerate(false_positive_cases):
            with st.expander(f"案例 {i+1}: {case['text'][:50]}...", expanded=True):
                result = engine.predict(case["text"])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if result["violation"]:
                        st.error("❌ 系统判断: 违规")
                    else:
                        st.success("✅ 系统判断: 合规")
                
                with col2:
                    st.metric("预期结果", "合规")
                
                with col3:
                    is_correct = not result["violation"]
                    if is_correct:
                        st.success("✅ 判断正确")
                    else:
                        st.error("❌ 判断错误")
                
                st.markdown(f"**系统理由**: {result['reason']}")
                st.markdown(f"**正确原因**: {case['correct_reason']}")
                
                if result.get('pre_check_used', False):
                    st.info("🔍 使用了预检查机制")
                
                results.append({
                    '案例': case['text'][:50] + "...",
                    '系统判断': '违规' if result['violation'] else '合规',
                    '是否正确': is_correct,
                    '分析方式': '预检查' if result.get('pre_check_used', False) else '深度分析'
                })
        
        # 总体统计
        st.markdown("### 📈 误判验证总体结果")
        correct_count = sum(1 for r in results if r['是否正确'])
        total_count = len(results)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("总案例数", total_count)
        col2.metric("正确判断", correct_count)
        col3.metric("准确率", f"{(correct_count/total_count*100):.1f}%")
        
        if correct_count == total_count:
            st.balloons()
            st.success("🎉 所有误判案例验证通过！系统改进有效。")

if __name__ == "__main__":
    main()
