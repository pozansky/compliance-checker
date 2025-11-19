from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TriggerConfig(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    regex_patterns: List[str] = Field(default_factory=list)
    context_words: List[str] = Field(default_factory=list)

class FewShotExample(BaseModel):
    input: str
    violation: bool
    reason: str

class ComplianceRule(BaseModel):
    event_name: str
    risk_level: str
    score: int
    description: str
    trigger: TriggerConfig
    whitelist: List[str] = Field(default_factory=list)
    few_shot: List[FewShotExample] = Field(default_factory=list)