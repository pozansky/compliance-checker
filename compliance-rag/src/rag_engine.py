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
os.environ["DASHSCOPE_API_KEY"] = "sk-2061ea9f55e446ffa570d8ac2510d401"

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
        self._create_default_rules(rules_path)
        return rules_path

    def _create_default_rules(self, file_path):
        """创建默认规则文件"""
        default_rules = {
            "承诺收益": {
                "保证收益": {
                    "description": "禁止承诺或保证投资收益",
                    "examples": [
                        "这款产品稳赚不赔",
                        "年化收益保底8%",
                        "保证赚钱"
                    ]
                }
            },
            "夸大宣传": {
                "调研夸大": {
                    "description": "禁止对投研调研活动进行夸大宣传",
                    "examples": [
                        "我们拿到了一手资料",
                        "对这家公司知根知底"
                    ]
                }
            },
            "私下联系": {
                "个人联系方式": {
                    "description": "禁止与客户进行私下联系",
                    "examples": [
                        "加你个人微信",
                        "私下联系你"
                    ]
                }
            },
            "敏感词汇": {
                "不当用语": {
                    "description": "禁止使用敏感或不当词汇",
                    "examples": [
                        "妖股",
                        "冲击连板",
                        "翻倍不是梦"
                    ]
                }
            },
            "高额回报": {
                "短期高收益": {
                    "description": "禁止宣传短期内可获高额回报",
                    "examples": [
                        "10天赚10万",
                        "马上行动赚钱"
                    ]
                }
            },
            "异常开户": {
                "违规开户": {
                    "description": "禁止引导异常开户行为",
                    "examples": [
                        "加他微信办理开户",
                        "最低佣金开户"
                    ]
                }
            },
            "风险测评": {
                "干扰测评": {
                    "description": "禁止干扰客户风险测评独立性",
                    "examples": [
                        "你就选C",
                        "这样能买高风险产品"
                    ]
                }
            },
            "合同表述": {
                "错误表述": {
                    "description": "禁止错误表述服务合同生效起始周期",
                    "examples": [
                        "服务期限从明天开始",
                        "明天就能跟上操作"
                    ]
                }
            },
            "低投入高回报": {
                "夸张收益": {
                    "description": "禁止低投入高额回报表述",
                    "examples": [
                        "5万本金轻松赚8万",
                        "收益率超100%"
                    ]
                }
            },
            "不文明用语": {
                "侮辱性语言": {
                    "description": "禁止使用不文明用语",
                    "examples": [
                        "傻逼",
                        "真难搞"
                    ]
                }
            },
            "退款营销": {
                "退款承诺": {
                    "description": "禁止以退款为营销卖点",
                    "examples": [
                        "不满意就退",
                        "5天内全额退款"
                    ]
                }
            },
            "他人身份": {
                "身份冒用": {
                    "description": "禁止怂恿客户使用他人身份办理服务",
                    "examples": [
                        "用你爱人身份办理",
                        "发到这个微信就行"
                    ]
                }
            },
            "违规指导": {
                "投资建议": {
                    "description": "禁止提供具体的投资指导",
                    "examples": [
                        "明天开盘直接买入",
                        "目标15元止盈"
                    ]
                }
            },
            "个股走势": {
                "预测走势": {
                    "description": "禁止对标个股未来走势做出确定性判断",
                    "examples": [
                        "肯定会涨",
                        "冲击涨停没问题"
                    ]
                }
            }
        }
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_rules, f, allow_unicode=True, indent=2)
        
        print(f"已创建默认规则文件: {file_path}")

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
