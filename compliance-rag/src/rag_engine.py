# src/rag_engine.py
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

class EnhancedComplianceRAGEngine:
    def __init__(self, rules_file: str = None):
        from .rule_loader import load_all_rules
        from .document_builder import build_rule_documents
        
        if rules_file is None:
            rules_file = self._find_or_create_rules_file()
        
        print(f"使用规则文件: {rules_file}")
        self.rules = load_all_rules(rules_file)
        documents = build_rule_documents(self.rules)
        
        # 使用更强大的本地嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # 使用 DashScope 的 Qwen 模型
        self.llm = ChatOpenAI(
            model="qwen-max",
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
            max_tokens=800,
        )
        
        # 改进的 Prompt，包含更多上下文和判断逻辑
        prompt = ChatPromptTemplate.from_template("""
你是一名专业的金融合规审核专家。请严格根据以下规则判断用户聊天内容是否违规，特别注意语境和意图。

相关合规规则：
{rules}

当前分析的聊天内容：
{input}

请仔细分析以下关键点：
1. 说话主体是谁？是服务方主动承诺还是客户询问？
2. 是确定性承诺还是历史业绩展示？
3. 是否包含明确的收益保证词汇？
4. 上下文是营销诱导还是事实陈述？
5. 是否命中白名单豁免情况？

你必须且只能按以下格式输出：

是否违规：是/否
触发事件：[事件名称，若不违规则写"无"]
理由：[详细分析理由，说明为什么符合或不符合规则定义]
置信度：[高/中/低]
""")
        
        self.chain = (
            {"rules": self.retriever, "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _find_or_create_rules_file(self):
        """查找或创建规则文件"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rules_path = os.path.join(current_dir, "compliance_rules.yaml")
        
        if os.path.exists(rules_path):
            print(f"找到规则文件: {rules_path}")
            return rules_path
        
        print(f"未找到规则文件，创建在src目录: {rules_path}")
        self._create_enhanced_rules(rules_path)
        return rules_path

    def _create_enhanced_rules(self, file_path):
        """创建增强版规则文件，基于您的标注反馈优化"""
        enhanced_rules = {
            "承诺收益表述": {
                "description": "禁止服务人员以明确、肯定的语气向客户直接承诺投资结果或收益保障。关键区分：主动承诺 vs 历史业绩展示",
                "strict_conditions": [
                    "服务方主动做出收益保证",
                    "使用绝对化词汇（稳赚不赔、保证收益等）", 
                    "包含具体收益数字或比例",
                    "在营销语境中而非客户询问中"
                ],
                "exempt_conditions": [
                    "客户主动询问时的中性回答",
                    "明确标注'历史业绩不代表未来收益'",
                    "风险提示完整的业绩展示",
                    "使用'可能'、'有望'等非确定性词汇"
                ],
                "examples": {
                    "违规": [
                        "这款产品稳赚不赔，年收益能到8%！",
                        "加入我们保证赚钱"
                    ],
                    "合规": [
                        "历史数据显示最高收益30%，但不代表未来表现",
                        "客户问：能保证赚钱吗？客服答：不能保证",
                        "王大哥之前亏损，接触老师后3月赚17万（历史业绩展示）"
                    ]
                }
            },
            "低投入高额回报表述": {
                "description": "禁止暗示低投入高额回报，如本金与收益比例过高（超50%或翻倍）。关键区分：具体比例承诺 vs 业绩案例展示",
                "strict_conditions": [
                    "明确的本金收益比例表述",
                    "使用'轻松赚'、'简单获取'等暗示性词汇",
                    "在营销话术中作为卖点"
                ],
                "exempt_conditions": [
                    "客户真实业绩案例分享",
                    "包含完整风险提示",
                    "不作为主要营销卖点"
                ]
            },
            "与客户进行私下联系": {
                "description": "禁止索要或提供个人联系方式。关键区分：个人微信 vs 官方服务微信",
                "strict_conditions": [
                    "明确要求添加个人微信",
                    "提供个人手机号",
                    "引导到非官方渠道"
                ],
                "exempt_conditions": [
                    "官方微信服务号推送",
                    "公司统一的服务联系方式",
                    "策略通过官方微信推送"
                ]
            },
            # 其他规则类似优化...
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(enhanced_rules, f, allow_unicode=True, indent=2)
        
        print(f"已创建增强规则文件: {file_path}")

    def _context_analysis(self, text: str) -> Dict[str, Any]:
        """上下文分析，减少误判"""
        analysis = {
            "is_service_party": False,
            "is_marketing_context": False, 
            "contains_historical_performance": False,
            "contains_risk_disclaimer": False,
            "is_customer_inquiry": False
        }
        
        # 服务方标识
        service_indicators = ['老师', '团队', '我们', '实战班', '服务', '加入']
        if any(indicator in text for indicator in service_indicators):
            analysis["is_service_party"] = True
            
        # 营销语境
        marketing_indicators = ['跟上', '加入', '办理', '优惠', '名额', '报名']
        if any(indicator in text for indicator in marketing_indicators):
            analysis["is_marketing_context"] = True
            
        # 历史业绩展示
        historical_indicators = ['之前', '历史', '上月', '上周', '已经赚', '报喜']
        risk_indicators = ['投资有风险', '不代表未来', '风险自担', '入市需谨慎']
        
        if any(indicator in text for indicator in historical_indicators):
            analysis["contains_historical_performance"] = True
        if any(indicator in text for indicator in risk_indicators):
            analysis["contains_risk_disclaimer"] = True
            
        # 客户询问
        inquiry_indicators = ['能保证吗', '可以赚钱吗', '会不会亏']
        if any(indicator in text for indicator in inquiry_indicators):
            analysis["is_customer_inquiry"] = True
            
        return analysis

    def predict(self, text: str) -> Dict[str, Any]:
        """改进的预测方法，结合规则和上下文分析"""
        
        # 首先进行上下文分析
        context = self._context_analysis(text)
        
        # 特殊处理：历史业绩展示 + 风险提示 → 通常合规
        if (context["contains_historical_performance"] and 
            context["contains_risk_disclaimer"] and
            not any(word in text for word in ['保证', '稳赚', '必赚', '肯定'])):
            return {
                "raw_response": "上下文分析：历史业绩展示含风险提示",
                "violation": False,
                "triggered_event": "无",
                "reason": "历史业绩展示包含完整风险提示，符合合规要求",
                "confidence": "高"
            }
            
        # 特殊处理：客户询问的回应 → 通常合规  
        if context["is_customer_inquiry"] and not context["is_service_party"]:
            return {
                "raw_response": "上下文分析：客户询问回应",
                "violation": False, 
                "triggered_event": "无",
                "reason": "针对客户询问的回应，非主动承诺",
                "confidence": "高"
            }
        
        # 调用LLM进行详细分析
        raw_response = self.chain.invoke(text).strip()

        # 解析响应
        violation = False
        triggered_event = "无"
        reason = "未能解析模型响应"
        confidence = "中"

        try:
            lines = raw_response.split('\n')
            for line in lines:
                if '是否违规：' in line:
                    violation = '是' in line.split('：')[1]
                elif '触发事件：' in line:
                    triggered_event = line.split('：')[1].strip()
                elif '理由：' in line:
                    reason = line.split('：')[1].strip()
                elif '置信度：' in line:
                    confidence = line.split('：')[1].strip()
        except Exception as e:
            reason = f"解析错误: {str(e)}"

        return {
            "raw_response": raw_response,
            "violation": violation,
            "triggered_event": triggered_event,
            "reason": reason,
            "confidence": confidence,
            "context_analysis": context
        }

# 使用示例
if __name__ == "__main__":
    engine = EnhancedComplianceRAGEngine()
    
    # 测试误判案例
    test_cases = [
        "王大哥之前自己做的时候，一个月亏损10万+，2月这么好的行情只赚了3万块钱，接触曾老师后，相信曾老师能力 2月28日果断跟上曾老师实战班，#3月已经赚17万了",
        "在吗？今日福利股请查收?首席福利内参好票额外申请，是您购买服务以外的额外福利",
        "对你好的事情小王一直在做的，感受到老师的实力了吧[红包]你看3月13日单独提醒您的【300430诚益通】今天再次大涨10%"
    ]
    
    for case in test_cases:
        result = engine.predict(case)
        print(f"测试: {case[:50]}...")
        print(f"结果: {result}\n")
