import os
import yaml
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any

# 设置 DashScope API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-a677631fd47a4e2184b6836f6097f0b5"

class ComplianceRAGEngine:
    def __init__(self, rules_file: "compliance_rag/compliance_rules.yaml"):
        # 如果未提供规则文件路径，使用硬编码的绝对路径
        if rules_file is None:
            # 使用项目中的实际路径
            rules_file = "/mount/compliance-checker/src/compliance_rules.yaml"
        
        print(f"使用规则文件: {rules_file}")
        
        # 检查文件是否存在
        if not os.path.exists(rules_file):
            raise FileNotFoundError(f"规则文件未找到: {rules_file}。请确保文件存在且路径正确。")
        
        # 动态导入
        try:
            from rule_loader import load_all_rules
            from document_builder import build_rule_documents
        except ImportError as e:
            raise ImportError(f"导入依赖模块失败: {e}")
        
        # 加载规则和文档
        rules = load_all_rules(rules_file)
        documents = build_rule_documents(rules)
        
        # 使用 HuggingFace 本地嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 使用 DashScope 的 Qwen 模型
        self.llm = ChatOpenAI(
            model="qwen-max",
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
            max_tokens=500,
        )
        
        # 改进的 Prompt，增加业务上下文理解
        prompt = ChatPromptTemplate.from_template("""
你是一名金融合规审核员。请严格根据以下规则判断用户聊天内容是否违规，同时要考虑金融业务的实际场景。

相关规则：
{rules}

**重要业务上下文说明：**
1. 使用企业微信进行服务通知是允许的正常业务流程
2. 回顾历史个股表现和绩效展示是允许的，只要不是对未来收益的承诺
3. 产品介绍和邀约是正常营销行为，只要不涉及具体投资指导
4. 使用"有望"、"可能"等非确定性词汇描述潜力是允许的
5. 服务提醒和内容推送是正常客户服务

**需要特别注意的违规行为：**
- 明确承诺或保证收益
- 提供具体的买卖点指导
- 引导使用私人联系方式（非企微）
- 使用绝对化词汇如"肯定"、"必然"、"稳赚"
- 怂恿使用他人身份办理业务

聊天内容：
{input}

你必须且只能按以下格式输出，不要任何其他文字：

是否违规：是/否
触发事件：[事件名称，若不违规则写"无"]
理由：[简明理由，引用规则中的关键词或逻辑，区分正常业务行为和违规行为]
""")
        
        self.chain = (
            {"rules": self.retriever, "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _preprocess_text(self, text: str) -> str:
        """预处理文本，识别正常业务模式"""
        # 识别企微服务模式
        if any(pattern in text for pattern in ['企业微信', '企微', '服务号', 'APP']):
            text += " [此内容涉及企业微信服务流程]"
        
        # 识别历史业绩回顾
        if any(pattern in text for pattern in ['回顾', '上周', '上月', '历史', '过去', '此前']):
            text += " [此内容为历史业绩回顾]"
        
        # 识别产品邀约模式
        if any(pattern in text for pattern in ['邀请', '邀约', '查看', '关注', '回复']):
            text += " [此内容为正常产品邀约]"
            
        return text

    def predict(self, text: str) -> Dict[str, Any]:
        # 预处理文本，增加业务上下文信息
        processed_text = self._preprocess_text(text)
        
        raw_response = self.chain.invoke(processed_text).strip()

        violation = False
        triggered_event = "无"
        reason = "未能解析模型响应"

        try:
            if "是否违规：" in raw_response:
                violation_line = raw_response.split("是否违规：")[1].split("\n")[0].strip()
                violation = "是" in violation_line

            if "触发事件：" in raw_response:
                triggered_event = raw_response.split("触发事件：")[1].split("\n")[0].strip()

            if "理由：" in raw_response:
                reason = raw_response.split("理由：")[1].strip()
        except Exception as e:
            reason = f"解析错误: {str(e)}"

        return {
            "raw_response": raw_response,
            "violation": violation,
            "triggered_event": triggered_event,
            "reason": reason
        }

# 使用示例
if __name__ == "__main__":
    # 测试引擎
    try:
        # 可以在这里指定不同的路径进行测试
        engine = ComplianceRAGEngine()
        print("合规引擎初始化成功！")
        
        # 测试预测
        test_text = "加我私人微信，告诉你明天必涨的股票"
        result = engine.predict(test_text)
        print(f"测试结果: {result}")
        
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
