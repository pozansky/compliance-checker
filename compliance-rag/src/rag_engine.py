# src/rag_engine.py
import os
import yaml
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any

# 设置 DashScope API Key
# os.environ["DASHSCOPE_API_KEY"] = "sk-2061ea9f55e446ffa570d8ac2510d401"
os.environ["DASHSCOPE_API_KEY"] = "sk-a677631fd47a4e2184b6836f6097f0b5"
class ComplianceRAGEngine:
    def __init__(self, rules_file: str = None):
        from .rule_loader import load_all_rules
        from .document_builder import build_rule_documents
        
        # 自动查找或创建规则文件
        if rules_file is None:
            rules_file = self._find_or_create_rules_file()
        
        print(f"使用规则文件: {rules_file}")
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
        
        # 定义带结构化输出的 Prompt
        prompt = ChatPromptTemplate.from_template("""
你是一名金融合规审核员。请严格根据以下规则判断用户聊天内容是否违规。

相关规则：
{rules}

聊天内容：
{input}

你必须且只能按以下格式输出，不要任何其他文字：

是否违规：是/否
触发事件：[事件名称，若不违规则写"无"]
理由：[简明理由，引用规则中的关键词或逻辑]
""")
        
        self.chain = (
            {"rules": self.retriever, "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _find_or_create_rules_file(self):
        """查找或创建规则文件"""
        # 获取当前文件所在目录（src目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 规则文件就在当前src目录下
        rules_path = os.path.join(current_dir, "compliance_rules.yaml")
        
        if os.path.exists(rules_path):
            print(f"找到规则文件: {rules_path}")
            return rules_path
        
        # 如果找不到，在当前src目录创建规则文件
        print(f"未找到规则文件，创建在src目录: {rules_path}")
       
        return rules_path


    def predict(self, text: str) -> Dict[str, Any]:
        raw_response = self.chain.invoke(text).strip()

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
