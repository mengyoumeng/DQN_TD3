import json
import requests
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class LLMPolicyConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    temperature: float = 0.0
    timeout: int = 60

class OllamaLLMPolicy:
    """
    连续动作版本：输出 JSON:
    {"action":[a0,a1,...], "risk":0-1, "reason":"..."}
    """
    def __init__(self, cfg: LLMPolicyConfig):
        self.cfg = cfg

    def _build_prompt(self, obs: Dict[str, Any], action_dim: int, action_low: float, action_high: float) -> str:
        return f"""
你是网络安全“检测-分析-响应”一体化系统中的连续控制决策模块。
你必须只输出严格 JSON（不要 Markdown，不要多余文字）。

要求：
- action 是一个数组，长度 = {action_dim}
- 每个元素在 [{action_low}, {action_high}] 范围内
- risk 在 [0, 1]

输出格式（必须包含这些键）：
{{
  "action": [{", ".join(["0.0"]*action_dim)}],
  "risk": 0.0,
  "reason": "一句话理由"
}}

obs={json.dumps(obs, ensure_ascii=False)}
""".strip()

    def act(self, obs: Dict[str, Any], action_dim: int, action_low: float = -1.0, action_high: float = 1.0) -> Dict[str, Any]:
        prompt = self._build_prompt(obs, action_dim, action_low, action_high)
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.cfg.temperature}
        }
        url = f"{self.cfg.base_url}/api/generate"
        r = requests.post(url, json=payload, timeout=self.cfg.timeout)
        r.raise_for_status()
        text = r.json().get("response", "").strip()

        data = self._safe_json(text)

        a = data.get("action", None)
        if not (isinstance(a, list) and len(a) == action_dim):
            a = [0.0] * action_dim

        risk = float(data.get("risk", 0.5))
        risk = max(0.0, min(1.0, risk))
        reason = str(data.get("reason", ""))[:200]

        return {"action": a, "risk": risk, "reason": reason}

    @staticmethod
    def _safe_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            l = text.find("{")
            r = text.rfind("}")
            if l != -1 and r != -1 and r > l:
                try:
                    return json.loads(text[l:r+1])
                except Exception:
                    pass
        return {"action": [0.0, 0.0], "risk": 0.5, "reason": "parse_failed_fallback"}

