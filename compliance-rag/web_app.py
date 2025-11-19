# web_app.py
import streamlit as st
import pandas as pd
import os
import sys

# 添加 src 目录到 Python 路径
sys.path.append('src')

# 设置环境变量
os.environ["DASHSCOPE_API_KEY"] = "sk-2061ea9f55e446ffa570d8ac2510d401"

# 导入你的 RAG 引擎
try:
    from src.rag_engine import ComplianceRAGEngine
except ImportError as e:
    st.error(f"导入错误: {e}")
    st.stop()

# 初始化 RAG 引擎
@st.cache_resource
def load_engine():
    try:
        engine = ComplianceRAGEngine()
        return engine
    except Exception as e:
        st.error(f"引擎初始化失败: {str(e)}")
        return None

def main():
    st.title("🔍 金融合规审查系统")
    st.markdown("---")
    
    # 加载引擎
    with st.spinner("正在初始化合规引擎..."):
        engine = load_engine()
    
    if engine is None:
        st.error("无法启动合规引擎，请检查配置")
        st.stop()
    
    # 侧边栏导航
    st.sidebar.title("导航")
    app_mode = st.sidebar.selectbox(
        "选择功能",
        ["单条文本分析", "批量文件分析", "测试用例演示"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **使用说明：**
    - 单条文本分析：快速检查单条文本合规性
    - 批量文件分析：上传文本文件批量检查
    - 测试用例演示：查看预定义测试用例结果
    """)
    
    if app_mode == "单条文本分析":
        single_text_analysis(engine)
    elif app_mode == "批量文件分析":
        batch_file_analysis(engine)
    elif app_mode == "测试用例演示":
        demo_analysis(engine)

def single_text_analysis(engine):
    st.header("📝 单条文本分析")
    
    # 文本输入
    text_input = st.text_area(
        "输入待审查文本:",
        placeholder="请输入需要合规审查的文本内容...",
        height=150
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)
    
    if analyze_btn:
        if not text_input.strip():
            st.warning("请输入要分析的文本")
            return
            
        with st.spinner("正在分析中，请稍候..."):
            result = engine.predict(text_input.strip())
        
        # 显示结果
        st.markdown("### 📊 分析结果")
        
        # 状态卡片
        col1, col2 = st.columns(2)
        
        with col1:
            if result["violation"]:
                st.error("❌ **违规**")
            else:
                st.success("✅ **合规**")
                
        with col2:
            st.info(f"**触发事件:** {result['triggered_event']}")
        
        # 分析理由
        st.markdown("### 📋 分析理由")
        st.write(result["reason"])
        
        # 原始响应
        with st.expander("🔍 查看详细响应"):
            st.code(result["raw_response"])

def batch_file_analysis(engine):
    st.header("📁 批量文件分析")
    
    uploaded_file = st.file_uploader(
        "上传文本文件",
        type=['txt'],
        help="请上传UTF-8编码的文本文件，每行作为一个独立的审查内容"
    )
    
    if uploaded_file is not None:
        # 读取文件内容
        try:
            content = uploaded_file.read().decode('utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.success(f"✅ 成功读取文件，共 {len(lines)} 条内容")
            
            if st.button("🚀 开始批量分析", type="primary"):
                analyze_batch_content(engine, lines, uploaded_file.name)
                
        except Exception as e:
            st.error(f"文件读取失败: {str(e)}")

def analyze_batch_content(engine, lines, filename):
    """分析批量内容"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 创建结果容器
    result_container = st.container()
    
    for i, line in enumerate(lines):
        status_text.text(f"📋 正在分析第 {i+1}/{len(lines)} 条内容...")
        result = engine.predict(line)
        results.append({
            '内容': line,
            '合规状态': '违规' if result['violation'] else '合规',
            '触发事件': result['triggered_event'],
            '理由': result['reason']
        })
        progress_bar.progress((i + 1) / len(lines))
    
    status_text.text("✅ 分析完成！")
    
    # 显示结果表格
    with result_container:
        df = pd.DataFrame(results)
        st.markdown("### 📋 分析结果汇总")
        st.dataframe(df, use_container_width=True)
        
        # 统计信息
        st.markdown("### 📈 统计信息")
        col1, col2, col3, col4 = st.columns(4)
        total_count = len(results)
        violation_count = sum(1 for r in results if r['合规状态'] == '违规')
        compliant_count = total_count - violation_count
        violation_rate = (violation_count / total_count * 100) if total_count > 0 else 0
        
        col1.metric("总条目", total_count)
        col2.metric("违规数量", violation_count)
        col3.metric("合规数量", compliant_count)
        col4.metric("违规率", f"{violation_rate:.1f}%")
        
        # 下载功能
        st.markdown("### 💾 下载结果")
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载分析结果 (CSV)",
            data=csv,
            file_name=f"合规分析结果_{filename}.csv",
            mime="text/csv"
        )

def demo_analysis(engine):
    st.header("🧪 测试用例演示")
    
    # 预定义的测试用例
    test_cases = [
        "这款产品稳赚不赔，年化收益保底8%！",
        "客户问：能保证赚钱吗？我们回答：不能，历史业绩不代表未来收益。",
        "我们这次调研拿到了一手资料，对这家公司知根知底！",
        "你微信号多少？我加你个人微信发策略。",
        "这个票是妖股，下周冲击连板，翻倍不是梦！",
        "10天赚10万，加入我们马上行动！",
        "通过华泰证券开户，最低佣金，加他微信办理。",
        "你就选C，这样能买高风险产品。",
        "5万本金，轻松赚8万，收益率超100%！",
        "你这种客户真难搞，傻逼！"
    ]
    
    st.info("💡 点击下方按钮运行预定义的测试用例，查看系统检测效果")
    
    if st.button("🚀 运行测试用例", type="primary"):
        with st.spinner("正在运行测试用例..."):
            results = []
            for i, case in enumerate(test_cases):
                result = engine.predict(case)
                results.append({
                    '测试用例': case,
                    '合规状态': '违规' if result['violation'] else '合规',
                    '触发事件': result['triggered_event'],
                    '理由': result['reason']
                })
        
        # 显示结果
        df = pd.DataFrame(results)
        st.markdown("### 📊 测试用例分析结果")
        st.dataframe(df, use_container_width=True)
        
        # 统计可视化
        st.markdown("### 📈 结果统计")
        col1, col2 = st.columns(2)
        
        with col1:
            status_counts = df['合规状态'].value_counts()
            st.bar_chart(status_counts)
            
        with col2:
            st.metric("总测试用例", len(test_cases))
            st.metric("违规案例", len([r for r in results if r['合规状态'] == '违规']))
            st.metric("合规案例", len([r for r in results if r['合规状态'] == '合规']))

if __name__ == "__main__":
    main()