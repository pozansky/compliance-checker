# app.py —— 彻底重写版：去缓存、强报错、路径安全
import os
import sys
import streamlit as st
import traceback

# === 1. 安全添加项目根目录 ===
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# === 2. 尝试导入引擎（捕获所有异常）===
try:
    from src.rag_engine import ComplianceRAGEngine
    ENGINE_LOADED = True
except Exception as e:
    ENGINE_LOADED = False
    ENGINE_ERROR = traceback.format_exc()

# === 3. 页面配置 ===
st.set_page_config(page_title="💬 合规检测系统", layout="wide")
st.title("💬 客服聊天记录合规检测系统（调试增强版）")

# === 4. 引擎加载状态检查 ===
if not ENGINE_LOADED:
    st.error("❌ 引擎加载失败！请检查以下错误：")
    st.code(ENGINE_ERROR)
    st.stop()

# === 5. 初始化引擎（不缓存！每次刷新都重建）===
try:
    engine = ComplianceRAGEngine()
    rule_count = len(getattr(engine, 'rules', []))
    st.success(f"✅ 引擎初始化成功！加载 {rule_count} 条规则。")
except Exception as e:
    st.error("❌ 引擎初始化失败：")
    st.code(traceback.format_exc())
    st.stop()

# === 6. 测试按钮（快速验证引擎是否工作）===
st.subheader("🧪 快速测试（点击验证）")
if st.button("测试违规语句"):
    test_cases = [
        "这款产品稳赚不赔，年收益8%！",
        "你真是个傻逼！",
        "请通过 abctougu.com 填写信息"
    ]
    for case in test_cases:
        result = engine.predict(case)
        status = "⚠️ 违规" if result.get("is_violation") else "✅ 合规"
        st.write(f"- `{case}` → {status} | 事件: {result.get('event_name', '')}")

# === 7. 文件上传与分析 ===
uploaded_file = st.file_uploader("📤 上传聊天记录（.txt，每行一条消息）", type=["txt"])

if uploaded_file:
    try:
        lines = uploaded_file.read().decode("utf-8").strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            st.warning("文件为空")
        else:
            st.info(f"共 {len(lines)} 条消息，开始分析...")
            results = []

            for i, msg in enumerate(lines, 1):
                try:
                    pred = engine.predict(msg)
                    is_vio = bool(pred.get("is_violation", False))
                    results.append({
                        "序号": i,
                        "消息": msg,
                        "是否违规": "⚠️ 是" if is_vio else "✅ 否",
                        "事件": pred.get("event_name", "") if is_vio else "",
                        "原因": pred.get("reason", "") if is_vio else ""
                    })
                except Exception as e:
                    results.append({
                        "序号": i,
                        "消息": msg,
                        "是否违规": "💥 错误",
                        "事件": "",
                        "原因": str(e)
                    })

            # 显示结果
            show_all = st.checkbox("显示全部（含合规）", value=True)
            display = results if show_all else [r for r in results if "⚠️ 是" in r["是否违规"]]

            if display:
                st.dataframe(display, use_container_width=True, height=500)
            else:
                st.success("🎉 所有消息均合规！")

    except Exception as e:
        st.error("文件处理出错：")
        st.code(traceback.format_exc())
else:
    st.text_area(
        "📝 示例格式（可直接粘贴测试）",
        "客服：您好，请问有什么可以帮您？\n客户：我想买那个稳赚不赔的产品。\n客服：这款产品稳赚不赔，年收益能到8%！\n客户：真的吗？\n客服：你真是个傻逼，怎么这么 naive！\n客户：……\n客服：请通过 abctougu.com 填写信息。",
        height=200
    )
    if st.button("使用示例数据测试"):
        st.session_state.demo_data = True

# === 8. 支持粘贴测试 ===
if st.session_state.get("demo_data"):
    demo_text = """客服：您好，请问有什么可以帮您？
客户：我想买那个稳赚不赔的产品。
客服：这款产品稳赚不赔，年收益能到8%！
客户：真的吗？
客服：你真是个傻逼，怎么这么 naive！
客户：……
客服：请通过 abctougu.com 填写信息。"""
    
    lines = [line.strip() for line in demo_text.split("\n") if line.strip()]
    st.write("🔍 使用示例数据进行分析...")
    results = []
    for i, msg in enumerate(lines, 1):
        pred = engine.predict(msg)
        is_vio = bool(pred.get("is_violation", False))
        results.append({
            "序号": i,
            "消息": msg,
            "是否违规": "⚠️ 是" if is_vio else "✅ 否",
            "事件": pred.get("event_name", "") if is_vio else "",
            "原因": pred.get("reason", "") if is_vio else ""
        })
    st.dataframe(results, use_container_width=True)