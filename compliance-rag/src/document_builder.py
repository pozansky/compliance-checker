# src/document_builder.py
from typing import List
from langchain_core.documents import Document
from .schemas import ComplianceRule

def build_rule_documents(rules: List[ComplianceRule]) -> List[Document]:
    docs = []
    for rule in rules:
        # === 关键修改：显式列出关键词，提升检索命中率 ===
        keyword_str = ", ".join(rule.trigger.keywords) if rule.trigger.keywords else "无"
        
        content_lines = [
            f"【事件名称】{rule.event_name}",
            f"【关键词】{keyword_str}",  # ← 新增这一行！让 embedding 模型更容易关联
            f"【风险等级】{rule.risk_level}",
            f"【分值】{rule.score}",
            f"【描述】{rule.description}",
            f"【上下文词】{', '.join(rule.trigger.context_words) if rule.trigger.context_words else '无'}",
            f"【白名单】{', '.join(rule.whitelist) if rule.whitelist else '无'}",
            "【典型示例】"
        ]
        for ex in rule.few_shot[:3]:
            label = "违规" if ex.violation else "不违规"
            content_lines.append(f"- {label}: \"{ex.input}\" → {ex.reason}")
        
        content = "\n".join(content_lines).strip()
        docs.append(Document(page_content=content))
    return docs