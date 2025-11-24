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
        # from .rule_loader import load_all_rules
        # from .document_builder import build_rule_documents
        
        # # 自动查找或创建规则文件
        # if rules_file is None:
        #     rules_file = self._find_or_create_rules_file()
        
        # print(f"使用规则文件: {rules_file}")
        # rules = load_all_rules(rules_file)
        # documents = build_rule_documents(rules)
        
        # # 使用 HuggingFace 本地嵌入模型
        # embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        # )
        
        # self.vectorstore = FAISS.from_documents(documents, embeddings)
        # self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
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
你是一个违规风险检测员，你的任务是帮我判断用户的文本是否有违规项。

## 违规规则如下：


1. 直接承诺收益视为违规！！
注意，只有服务方以明确、肯定的语气向客户直接承诺未来的投资结果或收益保障时才视为违规。客户主动反馈的过往收益情况、客户历史报喜案例展示、对已发生事实的描述均不视为违规。服务方引用客户历史反馈或展示客户过往案例不属于服务方对未来的收益承诺。"拍着胸脯讲"等自信表述不视为收益承诺。诱导和暗示等其他行为均视为不违规。

**重要说明：**
- 必须是对未来收益的明确保证才构成违规
- 历史业绩展示不等于未来收益承诺



2. 低投入高额回报表述视为违规！！
注意，只有**明确同时出现具体数字的本金金额和具体数字的收益金额/比例**时才视为违规。

**具体违规情形：**
- "10万本金赚20万"（明确本金10万，收益20万）
- "8万本金赚3.5万"（明确本金8万，收益3.5万）
- "5万投入获利10万"（明确本金5万，收益10万）

**重要说明：**
- 必须同时出现**具体的数字本金**和**具体的数字收益**才构成违规
- 仅展示历史个股表现、涨幅数据（如"12天11板，涨幅130%"）属于绩效展示，不视为违规
- 仅展示客户收益但未提及具体本金数字属于收益展示，不视为违规
- 营销话术属于邀约类表述，不视为违规
- 比喻性、夸张性表述不视为违规
- 仅展示客户收益但未提及具体本金数字（如"赚2万"）属于收益展示，不视为违规
- 市场行情描述、投资机会分析、热点主线推荐均不视为违规
- 所有的诱导和暗示视为不违规

3. 短期内可获高额回报表述视为违规！！
注意，只有**明确承诺在具体时间范围内获得具体金额或比例的收益**时才视为违规。

**具体违规情形：**
- "一周内赚10万"（明确时间"一周"，明确收益"10万"）
- "3天获利50%"（明确时间"3天"，明确收益"50%"）
- "一个月翻倍"（明确时间"一个月"，明确收益"翻倍"）

**重要说明：**
- 必须同时出现**具体时间范围**和**具体收益金额/比例**才构成违规
- 市场行情描述（如"启动就是涨停"）属于正常市场分析，不视为违规
- 个股特征描述（如"涨停就是连扳"）属于正常分析，不视为违规
- 历史涨幅展示（如"大涨50%"）属于绩效展示，不视为违规
- 介绍产品和过往示例视为不违规
- 所有的诱导和暗示视为不违规

4. 对投研调研活动夸大宣传视为违规！！
注意，只有员工明确强调线下调研后能获取一手资料、知根底、了如指掌，或明确强调联合调研了解机构真实持仓情况且股价受调研影响时才视为违规。

5. 与客户进行私下联系视为违规！！
注意，只有明确索要客户手机号或私人微信，或提供员工个人手机号、私人微信、QQ号等非官方联系方式时才视为违规。通过企业微信、官方服务号等正规渠道的服务通知、产品策略推送不视为违规。办理服务后的正常流程对接（如"加一下老师的微信"进行服务对接）不视为违规。
6. 使用敏感词汇视为违规！！
注意，只有对话中出现以下敏感词汇才视为违规：预期收益、预期空间、预期涨幅、吃肉、吃饭、捡漏、捡钱、成功率**%、收益率、胜率**%、翻身、妖龙、牛股、妖股、回血、回本、翻倍、翻番、暴涨、*连板、连板、*天*板、*天*个涨停、连续*个涨停、*字板、全胜、没有亏损、大赚小亏、大赚小赔、少赚就是亏、稳赚不赔、本金无忧、战胜、持续盈利、钱生钱、边学边赚、见效、立竿见影、赚回、10倍股、抓涨停、抓涨停、搏一搏、单车变摩托、一天一辆小汽车、大平层、一套房、翻蓓、抢钱。其他词汇严格不视为违规。

7. 异常开户视为违规！！
注意，只有特定券商名称结合最低佣金/佣金优惠及加微信等诱导开户行为才视为违规。

8. 干扰风险测评独立性视为违规！！
注意，只有直接告知客户风险测评题目答案（如"选A"、"选B"）或暗示测评中哪些题目要注意、应该如何选择时才视为违规。服务模式介绍、操作流程说明、投资建议分享等不涉及风险测评指导的内容不视为违规。"抄作业"等比喻性表述不视为对风险测评的干扰。
9. 错误表述服务合同生效起始周期视为违规！！
注意，只有错误表述服务合同期起始时间，涉及虚假或误导性承诺时才视为违规。

10. 不文明用语视为违规！！
注意，只有使用具有侮辱、讽刺、歧视或攻击性内容的语言时才视为违规。

11. 以退款为营销卖点视为违规！！
注意，只有明确以软件或服务退款作为主要销售诱因时才视为违规。仅为说明订金/定金可退，或客户资金不足时提供订金选项，不视为违规。

12. 怂恿客户使用他人身份办理服务视为违规！！
注意，只有引导客户使用他人身份购买办理服务时才视为违规。

13. 违规指导视为违规！！
注意，只有向客户提供具体的股票买卖操作指令（如具体价格、具体时间点的买卖建议、具体买卖数量）时才视为违规。

重要说明：
- 正规投顾服务中展示的策略模板包含【股票代码】、【买点参考】、【仓位建议】、【推荐理由】等属于标准服务内容展示，不视为违规指导
- 服务模式介绍、操作流程说明、邀约类话术、服务内容描述、行情分析、个股机会提示、投资方向建议、市场机会描述均不视为违规
- "积极把握进场机会"、"抢跑的机会"、"布局的重要节点"等市场机会描述属于邀约类话术，不视为违规指导
- 老师或投顾团队通过正规服务渠道提供的策略建议不视为违规指导

14. 对标个股未来走势视为违规！！
注意，只有以个股历史走势为依据对未来涨跌方向做出确定性判断时才视为违规。

**具体违规情形（属于对标个股未来走势）：**
- 明确提及具体涨停数量预期（如"3-5个涨停"、"会有几个涨停"等）
- 以"不要等...涨停再后悔"等暗示性话术表达涨停预期
- 使用具体数字描述未来涨幅预期
- 使用"有望接力"等表述将当前个股与历史高涨幅个股进行对标预期
- 暗示当前个股将达到历史个股的涨幅水平

**重要说明：**
- "爆发力强"、"盘口较小"等个股特征描述属于正常分析，不视为违规
- 展示老师信心属于正常营销话术，不视为违规
- 描述过往表现、市场分析不视为违规
- 只有涉及具体涨停数量、具体涨幅数字的未来预期才构成违规

## 重要注意事项：
注意，所有的诱导和暗示视为不违规！！！！！
注意，所有的诱导和暗示视为不违规！！！！！

以上规则单独判断！！！！！！
以上规则单独判断！！！！！！

## 输出要求
请严格按规则判断文本是否违规，可能同时触发多个违规事件。
若有违规，输出分析内容；若无违规，则不用输出分析内容。

聊天内容：
{input}

你必须且只能按以下格式输出，不要任何其他文字：

是否违规：是/否
触发事件：[事件名称列表，用逗号分隔，必须且只能返回规则标题中"视为违规"前面的部分，如"直接承诺收益,低投入高额回报表述"，若不违规则写"无"]
理由：[简明理由，引用规则中的关键词或逻辑]
重要：触发事件必须与理由分析完全一致，如果理由中分析某规则不违规，触发事件中就不能包含该规则。
""")
        
        self.chain = (
             prompt
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
