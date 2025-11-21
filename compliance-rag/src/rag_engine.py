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
    def __init__(self, rules_file: str = None):
        # 使用绝对路径
        if rules_file is None:
            # 获取当前文件的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 直接指定规则文件在 src 目录下
            rules_file = os.path.join(current_dir, "compliance_rules.yaml")
            
            # 如果不存在就创建
            if not os.path.exists(rules_file):
                print(f"规则文件不存在，创建: {rules_file}")
                self._create_default_rules(rules_file)
        
        print(f"使用规则文件: {rules_file}")
        
        # 确保文件存在
        if not os.path.exists(rules_file):
            raise FileNotFoundError(f"规则文件不存在: {rules_file}")
        
        # 动态导入
        from rule_loader import load_all_rules
        from document_builder import build_rule_documents
        
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

    def _create_default_rules(self, file_path):
        """创建默认规则文件"""
        default_rules = {
            "承诺收益": {
                "保证收益": {
                    "description": "禁止明确承诺或保证投资收益，使用'稳赚不赔'、'保底收益'、'保证赚钱'等绝对化词汇",
                    "examples": [
                        "这款产品稳赚不赔",
                        "年化收益保底8%",
                        "保证赚钱",
                        "100%盈利"
                    ]
                }
            },
            "夸大宣传": {
                "调研夸大": {
                    "description": "禁止对投研调研活动进行夸大宣传，如声称有内幕消息或特殊渠道",
                    "examples": [
                        "我们拿到了一手内幕资料",
                        "有内部消息源",
                        "这家公司的董事长是我亲戚"
                    ]
                }
            },
            "私下联系": {
                "个人联系方式": {
                    "description": "禁止引导客户使用企业微信以外的私人联系方式，如个人微信、手机号、QQ等",
                    "examples": [
                        "加我私人微信聊",
                        "这是我的手机号",
                        "用QQ联系更方便"
                    ]
                },
                "注意": {
                    "description": "使用企业微信进行服务通知和提醒是允许的正常业务流程",
                    "examples": [
                        "企业微信通知是正常的",
                        "通过企微服务号发送是允许的"
                    ]
                }
            },
            "敏感词汇": {
                "不当用语": {
                    "description": "禁止使用敏感或不当词汇，如'妖股'、'冲击连板'等暗示投机行为的词汇",
                    "examples": [
                        "妖股",
                        "冲击连板",
                        "翻倍不是梦",
                        "马上涨停"
                    ]
                }
            },
            "高额回报": {
                "短期高收益": {
                    "description": "禁止宣传短期内可获得确定性的高额回报",
                    "examples": [
                        "10天保证赚10万",
                        "一个月资金翻倍",
                        "下周就能赚50%"
                    ]
                },
                "允许行为": {
                    "description": "展示历史业绩、回顾过往表现是允许的，只要不是对未来收益的承诺",
                    "examples": [
                        "回顾上周涨幅是允许的",
                        "展示历史绩效可以",
                        "过去的表现数据可以分享"
                    ]
                }
            },
            "异常开户": {
                "违规开户": {
                    "description": "禁止引导异常开户行为，如使用他人身份或非正规渠道",
                    "examples": [
                        "用你爱人的身份开户",
                        "找中介代办开户",
                        "虚假信息开户"
                    ]
                }
            },
            "风险测评": {
                "干扰测评": {
                    "description": "禁止干扰客户风险测评独立性，指导客户选择特定答案",
                    "examples": [
                        "你就选C选项",
                        "这样测评才能通过",
                        "按我说的选就能买高风险产品"
                    ]
                }
            },
            "合同表述": {
                "错误表述": {
                    "description": "禁止错误表述服务合同生效起始周期",
                    "examples": [
                        "服务期限从明天开始（实际应从签约日开始）",
                        "明天就能跟上操作（但合同尚未生效）"
                    ]
                }
            },
            "低投入高回报": {
                "夸张收益": {
                    "description": "禁止宣传低投入就能获得确定性的高额回报",
                    "examples": [
                        "5万本金保证赚8万",
                        "收益率肯定超100%",
                        "投入少回报高是确定的"
                    ]
                }
            },
            "不文明用语": {
                "侮辱性语言": {
                    "description": "禁止使用不文明用语或侮辱性语言",
                    "examples": [
                        "傻逼",
                        "真难搞",
                        "你这都不懂"
                    ]
                }
            },
            "退款营销": {
                "退款承诺": {
                    "description": "禁止以退款为营销卖点进行不当宣传",
                    "examples": [
                        "不满意就全额退款",
                        "5天内无理由退款保证"
                    ]
                }
            },
            "他人身份": {
                "身份冒用": {
                    "description": "禁止怂恿客户使用他人身份办理服务",
                    "examples": [
                        "用你爱人身份办理",
                        "发到这个非本人微信就行"
                    ]
                }
            },
            "违规指导": {
                "投资建议": {
                    "description": "禁止提供具体的投资买卖指导，包括具体的买入卖出价格和时间",
                    "examples": [
                        "明天开盘直接买入XX股票",
                        "目标15元必须止盈",
                        "现在立即卖出"
                    ]
                },
                "允许行为": {
                    "description": "产品介绍、服务邀约、内容推送是正常营销行为",
                    "examples": [
                        "邀请查看研报是允许的",
                        "推送分析内容可以",
                        "介绍产品特点没问题"
                    ]
                }
            },
            "个股走势": {
                "预测走势": {
                    "description": "禁止对标个股未来走势做出确定性判断，使用'肯定'、'必然'等绝对化词汇",
                    "examples": [
                        "肯定会涨",
                        "必然涨停",
                        "绝对会赚钱"
                    ]
                },
                "允许行为": {
                    "description": "使用'有望'、'可能'、'潜力'等非确定性词汇描述是允许的",
                    "examples": [
                        "有望上涨是允许表述",
                        "可能有潜力没问题",
                        "具备上涨空间可以"
                    ]
                }
            }
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_rules, f, allow_unicode=True, indent=2)
        
        print(f"已创建默认规则文件: {file_path}")

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
