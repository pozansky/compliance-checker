# src/rule_loader.py
import yaml
from pathlib import Path
from typing import List
from .schemas import ComplianceRule

def load_all_rules(rules_file: str = "compliance_rules.yaml") -> List[ComplianceRule]:
    """
    从单个 YAML 文件加载所有合规规则。
    文件内容应为 ComplianceRule 对象的列表。
    """
    file_path = Path(rules_file)
    if not file_path.exists():
        raise FileNotFoundError(f"规则文件未找到: {file_path.absolute()}")

    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("YAML 文件根节点必须是列表（- ...）")

    rules = []
    for i, item in enumerate(data):
        try:
            rule = ComplianceRule(**item)
            rules.append(rule)
        except Exception as e:
            print(f"跳过无效规则 #{i+1}: {e}")

    return rules