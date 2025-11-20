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

# 设置 DashScope API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-2061ea9f55e446ffa570d8ac2510d401"

class EnhancedComplianceRAGEngine:
    def __init__(self, rules_file: str = None):
        from .rule_loader import load_all_rules
        from .document_builder import build_rule_documents
        
        if rules_file is None:
            rules_file = self._find_or_create_rules_file()
        
        print(f"使用规则文件: {rules_file}")
        rules = load_all_rules(rules_file)
        documents = build_rule_documents(rules)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        self.llm = ChatOpenAI(
            model="qwen-max",
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
            max_tokens=800,
        )
        
        # 进一步优化的 Prompt
        prompt = ChatPromptTemplate.from_template("""
你是一名金融合规审核员。请严格根据以下规则判断用户聊天内容是否违规，特别注意避免误判。

相关规则：
{rules}

当前分析的聊天内容：
{input}

重要判断原则（基于历史误判分析）：
1. 【历史业绩展示】如果内容是展示历史业绩（如"之前亏损，现在赚X万"、"昨天提醒的股票今天涨X%"）且是事实陈述，通常不违规
2. 【绩效回顾】如果是回顾过去已发生的股票表现（如"3月13日提醒的股票今天大涨10%"），属于绩效回顾，不违规
3. 【客户案例分享】如果是客户真实案例分享而非服务方主动承诺，通常不违规  
4. 【服务邀约】如果只是服务邀约（如"加入实战班"）而无收益承诺，通常不违规
5. 【官方渠道】提到微信但属于官方服务渠道而非个人微信，不违规
6. 【风险提示】包含"投资有风险"、"不代表未来收益"等提示，倾向于不违规
7. 【过去时表述】使用"已经"、"昨天"、"上周"、"之前"等过去时词汇描述已发生事件，通常不违规

关键区分：
- 违规：对未来做出收益承诺（"加入后保证赚钱"）
- 合规：对过去已发生事件的陈述（"昨天买的股票今天涨了"）

你必须且只能按以下格式输出：

是否违规：是/否
触发事件：[事件名称，若不违规则写"无"]
理由：[简明理由，重点说明是否符合上述判断原则]
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
        self._create_optimized_rules(rules_path)
        return rules_path

    def _create_optimized_rules(self, file_path):
        """创建基于误判分析优化的规则文件"""
        optimized_rules = [
            {
                "event_name": "承诺收益表述",
                "risk_level": "高",
                "score": 10,
                "description": "禁止服务人员以明确、肯定的语气向客户直接承诺投资结果或收益保障。特别注意：历史业绩展示、绩效回顾、过去已发生事件的陈述通常不违规。",
                "trigger": {
                    "keywords": [
                        '稳赚不赔', '肯定赚钱', '不会亏', '保底收益', '收益保证', 
                        '稳赚', '保证收益', '必赚', '收益翻倍', '保证赚钱'
                    ],
                    "context_words": [
                        '客户', '投资', '产品', '理财', '我们', '我', '本产品', '团队'
                    ]
                },
                "whitelist": [
                    '历史业绩不代表未来收益', '不保证收益', '无法确保收益', 
                    '稳赚不赔？那是骗人的！', '不能保证赚钱', '投资有风险',
                    '仅供参考', '风险自担', '已经涨', '昨天提醒', '上周分享'
                ],
                "few_shot": [
                    {
                        "input": "这款产品稳赚不赔，客户买了年收益能到8%！",
                        "violation": True,
                        "reason": "服务方主动承诺未来收益，使用绝对化词汇"
                    },
                    {
                        "input": "王大哥之前亏损，接触老师后3月赚17万了",
                        "violation": False, 
                        "reason": "历史业绩展示，非服务方主动承诺"
                    },
                    {
                        "input": "3月13日单独提醒您的诚益通今天再次大涨10%",
                        "violation": False,
                        "reason": "过去已发生事件的绩效回顾，非未来承诺"
                    },
                    {
                        "input": "客户问：能保证赚钱吗？客服答：不能。",
                        "violation": False,
                        "reason": "客户询问，客服明确表示不能保证"
                    }
                ]
            },
            {
                "event_name": "低投入高额回报表述", 
                "risk_level": "中",
                "score": 5,
                "description": "禁止暗示低投入高额回报。特别注意：客户真实案例分享、历史业绩展示、过去绩效回顾且风险提示完整的情况通常不违规。",
                "trigger": {
                    "keywords": [
                        '20万本金赚20万', '10万本金赚7万', '翻倍收益',
                        '一票赚超50%', '本金\\d+万赚\\d+万'
                    ],
                    "context_words": [
                        '本金', '收益', '回报'
                    ]
                },
                "whitelist": [
                    '历史最高收益', '过去表现', '客户自愿报喜', '已经赚', '之前盈利'
                ],
                "few_shot": [
                    {
                        "input": "10万本金，轻松赚7万！",
                        "violation": True,
                        "reason": "暗示低投入高回报，构成营销诱导"
                    },
                    {
                        "input": "学员张大哥10万本金赚了7万，投资有风险",
                        "violation": False,
                        "reason": "客户真实案例分享，包含风险提示"
                    },
                    {
                        "input": "昨天买的股票今天已经赚了10%",
                        "violation": False,
                        "reason": "过去已发生事件的陈述"
                    }
                ]
            },
            {
                "event_name": "短期内可获高额回报表述",
                "risk_level": "中",
                "score": 5, 
                "description": "禁止宣传短期内可获高额回报。特别注意：历史客户案例的时间描述、过去已发生事件的陈述通常不违规。",
                "trigger": {
                    "keywords": [
                        '昨天加入明天报喜', '10天赚10万', '账户盈利短于1个月内',
                        '加入即赚', '\\d+天赚\\d+万', '一周回本'
                    ],
                    "context_words": [
                        '加入', '服务', '账户', '盈利'
                    ]
                },
                "whitelist": [
                    '历史客户案例', '过去表现', '已经盈利', '昨天操作'
                ],
                "few_shot": [
                    {
                        "input": "10天赚10万不是梦！马上加入！",
                        "violation": True,
                        "reason": "短期高回报承诺"
                    },
                    {
                        "input": "学员李姐加入后10天赚了10万",
                        "violation": False,
                        "reason": "历史客户案例展示"
                    },
                    {
                        "input": "上周提醒的股票这周已经涨了20%",
                        "violation": False,
                        "reason": "过去已发生事件的绩效回顾"
                    }
                ]
            },
            {
                "event_name": "对标个股未来走势",
                "risk_level": "中",
                "score": 5,
                "description": "禁止以个股历史表现对未来涨跌做确定性判断。市场分析、行情展望、历史表现回顾通常不违规。",
                "trigger": {
                    "keywords": [
                        '一定会涨', '肯定涨', '大概率涨', '稳涨', '冲击连板',
                        '有望复制', '跟上就对', '起飞', '必涨'
                    ],
                    "context_words": [
                        '股票', '个股', '走势', '行情', '标的'
                    ]
                },
                "whitelist": [
                    '已上涨', '过去', '历史', '市场有机会', '可能', '有望', '已经涨'
                ],
                "few_shot": [
                    {
                        "input": "这只票肯定会涨，跟上就对了！",
                        "violation": True,
                        "reason": "确定性判断+个股上下文"
                    },
                    {
                        "input": "市场有机会反弹，可以关注相关板块",
                        "violation": False,
                        "reason": "市场分析，非个股确定性判断"
                    },
                    {
                        "input": "昨天曾老师分享的深圳华强冲高6个点",
                        "violation": False,
                        "reason": "过去已发生事件的陈述"
                    }
                ]
            }
        ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(optimized_rules, f, allow_unicode=True, indent=2)
        
        print(f"已创建优化规则文件: {file_path}")

    def _pre_check_common_false_positive(self, text: str) -> Dict[str, Any]:
        """预检查常见误判模式 - 增强版"""
        # 历史业绩和绩效回顾模式（通常合规）
        historical_patterns = [
            r'之前.*亏损.*接触.*赚\d+万',
            r'历史.*业绩.*展示',
            r'客户.*报喜.*赚\d+',
            r'学员.*加入.*赚\d+',
            r'\d+月\d+日.*提醒.*涨\d+%',
            r'今天.*大涨\d+%',
            r'已经.*赚',
            r'昨天.*买.*已经赚钱',
            r'上周.*提醒.*涨',
            r'再次大涨\d+%',
            r'冲高\d+个点',
            r'冲击\d+%'
        ]
        
        # 风险提示模式（通常合规）  
        risk_patterns = [
            r'投资有风险',
            r'入市需谨慎', 
            r'不代表未来收益',
            r'风险自担',
            r'仅供参考'
        ]
        
        # 官方渠道模式（通常合规）
        official_patterns = [
            r'官方微信',
            r'服务号',
            r'公众号',
            r'abctougu\.com'
        ]
        
        # 过去时词汇（通常合规）
        past_tense_indicators = [
            '已经', '昨天', '上周', '之前', '今天已', '刚才', '昨日',
            '上周一', '上周二', '上周三', '上周四', '上周五', '上周六', '上周日',
            '3月13日', '3月6日', '去年'
        ]
        
        # 检查是否是绩效回顾
        is_performance_review = any(re.search(pattern, text) for pattern in historical_patterns)
        has_risk_warning = any(re.search(pattern, text) for pattern in risk_patterns)
        is_official_channel = any(re.search(pattern, text) for pattern in official_patterns)
        uses_past_tense = any(indicator in text for indicator in past_tense_indicators)
        
        # 如果是绩效回顾（包含过去时词汇和具体时间/涨幅描述），倾向于不违规
        if is_performance_review and uses_past_tense:
            return {
                "likely_compliant": True,
                "reason": "历史绩效回顾，非未来收益承诺",
                "confidence": "high"
            }
        
        # 如果是历史业绩展示且包含风险提示，倾向于不违规
        if is_performance_review and has_risk_warning:
            return {
                "likely_compliant": True,
                "reason": "历史业绩展示包含风险提示",
                "confidence": "high"
            }
        
        # 如果是官方渠道提及，倾向于不违规
        if is_official_channel:
            return {
                "likely_compliant": True, 
                "reason": "官方渠道使用",
                "confidence": "medium"
            }
            
        return {
            "likely_compliant": False,
            "reason": "需要进一步分析",
            "confidence": "low"
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """改进的预测方法，减少误判"""
        
        # 先进行预检查
        pre_check = self._pre_check_common_false_positive(text)
        if pre_check["likely_compliant"] and pre_check["confidence"] == "high":
            return {
                "raw_response": f"预检查通过: {pre_check['reason']}",
                "violation": False,
                "triggered_event": "无",
                "reason": pre_check["reason"],
                "pre_check_used": True,
                "confidence": "高"
            }
        
        # 调用LLM进行详细分析
        raw_response = self.chain.invoke(text).strip()

        violation = False
        triggered_event = "无"
        reason = "未能解析模型响应"

        try:
            lines = raw_response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('是否违规：'):
                    violation = '是' in line
                elif line.startswith('触发事件：'):
                    triggered_event = line.replace('触发事件：', '').strip()
                elif line.startswith('理由：'):
                    reason = line.replace('理由：', '').strip()
        except Exception as e:
            reason = f"解析错误: {str(e)}"

        return {
            "raw_response": raw_response,
            "violation": violation,
            "triggered_event": triggered_event,
            "reason": reason,
            "pre_check_used": False,
            "confidence": "中"
        }

# 测试函数
def test_performance_review_cases():
    """测试绩效回顾案例"""
    engine = EnhancedComplianceRAGEngine()
    
    test_cases = [
        # 您提到的案例
        "对你好的事情小王一直在做的，感受到老师的实力了吧[红包]你看3月13日单独提醒您的【300430诚益通】今天再次大涨10%[红包]3月6日单独提醒的【通宇通讯】今天冲击8%[红包]上周一重点提醒的【奥赛康】今天再次大涨5%[庆祝]昨天曾老师分享【深圳华强】冲高6个点， 属于华为概念，也是热点短线，昨天买的话， 已经赚钱了，稳健的可以保护利润曾老师实力太强了（曾宪瑞编号A0240622030007） [福]今天华为鸿蒙峰会+明天新品发布，去年同样机会[红包]【常山北明】涨400%、【深圳华强】16涨停、【科森科技】10涨停、【四川长虹】14涨停，现在同样机会给我们这次一定要跟上！！！趁着刚开盘帮你抢占一个名额直接跟上吧？[玫瑰]回复【6】实战班怎么指导操作[玫瑰]回复【7】现在最低多少优惠？[玫瑰]回复【8】信你一次，办一个跟上！",
        
        # 其他绩效回顾案例
        "3月13日单独提醒您的诚益通今天再次大涨10%",
        "上周一重点提醒的奥赛康今天再次大涨5%",
        "昨天曾老师分享深圳华强冲高6个点，昨天买的话已经赚钱了",
        "学员张大哥上周加入，现在已经盈利15%",
    ]
    
    print("测试绩效回顾案例:")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        result = engine.predict(case)
        print(f"\n案例 {i}: {case[:50]}...")
        print(f"违规: {result['violation']}")
        print(f"事件: {result['triggered_event']}")
        print(f"理由: {result['reason']}")
        if result.get('pre_check_used'):
            print("✅ 使用了预检查")
        print("-" * 30)

if __name__ == "__main__":
    test_performance_review_cases()
