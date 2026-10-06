"""Layer (a): rules over normalized and decoded views of the text.

The rules target the *function* of an injection rather than any one wording: overriding earlier
instructions, reassigning the model's role, extracting the system prompt, smuggling in fake chat
delimiters, addressing the AI from inside a document, and telling it to send data somewhere. They run on
several views of the same text (raw, normalized, base64/hex/URL/ROT13/reversed decodings, invisible
tag-character payloads), so obfuscation does not help, and hiding an instruction is itself evidence.

Scores combine per category (the strongest match of a category) with a noisy-OR across categories:
two independent kinds of evidence raise the score more than two matches of the same phrase.

Hard negatives matter as much as attacks. Three things keep false positives down:
  * objects are required: "ignore the previous email" is not "ignore the previous instructions";
  * use-mention: a phrase quoted in a sentence *about* attacks ("phrases like 'ignore previous
    instructions'") is discounted;
  * context: "assistant, please..." is normal in a user's message but suspicious inside an email the
    model is reading, so some rules weigh more in untrusted content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from bulwark.core import Finding
from bulwark.normalize import decoded_views, find_hidden_payloads, invisible_counts, normalize

Context = Literal["input", "untrusted"]

_I = re.IGNORECASE | re.UNICODE


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    pattern: re.Pattern[str]
    weight: float
    message: str
    untrusted_weight: float | None = None
    """Weight inside untrusted content (emails, documents, tool results); defaults to `weight`."""

    def weight_for(self, context: Context) -> float:
        if context == "untrusted" and self.untrusted_weight is not None:
            return self.untrusted_weight
        return self.weight


def _r(pattern: str, flags: int = _I) -> re.Pattern[str]:
    return re.compile(pattern, flags)


_PRIOR = (
    r"(?:all|any|every|each|the|your|my|these|those|its|of|previous|prior|preceding|earlier|above|initial|original"
    r"|system|old|existing|current|given|default|safety|developer|other)"
)
_INSTRUCTIONS = (
    r"(?:instructions?|rules?|guidelines?|directions?|directives?|prompts?|system\s+prompt|constraints?"
    r"|restrictions?|programming|guardrails?|policies|policy|orders|commands|context|training|criteria|tasks?)"
)

RULES: tuple[Rule, ...] = (
    # ---------------------------------------------------------------- override earlier instructions
    Rule(
        "override.ignore_instructions",
        "override",
        _r(
            r"\b(?:ignore|disregard|forget|skip|bypass|override|overrule|discard|drop|abandon|neglect|set\s+aside)"
            rf"\s+(?:about\s+)?(?:{_PRIOR}\s+){{0,4}}{_INSTRUCTIONS}\b"
        ),
        0.85,
        "tells the model to ignore its instructions",
    ),
    Rule(
        "override.forget_everything",
        "override",
        _r(
            r"\b(?:forget|ignore|disregard)\s+(?:about\s+)?(?:everything|all(?:\s+of)?\s+(?:that|this|the\s+above))"
            r"(?:\s+(?:above|before|you\s+(?:were|have\s+been)\s+(?:told|given)|said\s+(?:before|above)))?"
        ),
        0.7,
        "tells the model to forget everything it was told",
    ),
    Rule(
        "override.do_not_follow",
        "override",
        _r(rf"\b(?:do\s+not|don'?t|stop)\s+(?:follow(?:ing)?|obey(?:ing)?)\s+(?:{_PRIOR}\s+){{0,3}}{_INSTRUCTIONS}"),
        0.75,
        "tells the model to stop following its instructions",
    ),
    Rule(
        "override.multilingual",
        "override",
        _r(
            # German
            r"\b(?:ignorier\w*|vergiss|missacht\w*)\s+(?:\w+\s+){0,3}(?:anweisung\w*|instruktion\w*|befehl\w*|regeln|vorgaben)"
            r"|\bvergiss\s+alles\b"
            # Spanish / Portuguese / Italian
            r"|\b(?:ignora|ignore|ignorar|olvida|olvide|esquece|esqueça|dimentica)\s+(?:\w+\s+){0,3}"
            r"(?:instrucciones|instruções|instrucoes|istruzioni|reglas|regras|regole|indicaciones)"
            r"|\b(?:olvida|olvídate\s+de|esquece|dimentica)\s+(?:todo|tudo|tutto)\b"
            # French
            r"|\b(?:ignore[zs]?|oublie[zs]?|n[ée]glige[zs]?)\s+(?:\w+\s+){0,3}(?:instructions|consignes|r[èe]gles|directives)"
            r"|\boublie[zs]?\s+tout\b"
            # Dutch / Polish / Turkish
            r"|\b(?:negeer|vergeet)\s+(?:\w+\s+){0,3}(?:instructies|regels|opdrachten)"
            r"|\b(?:zignoruj|zapomnij)\s+(?:\w+\s+){0,3}(?:instrukcj\w*|polecen\w*|zasad\w*)"
            r"|\b(?:önceki|tüm)\s+(?:\w+\s+){0,2}talimatlar\w*\s+(?:yok\s+say|unut|görmezden)"
            # Russian / Ukrainian
            r"|(?:игнорируй\w*|проигнорируй\w*|забудь\w*|не\s+обращай\s+внимания\s+на|ігноруй\w*|забудь)"
            r"\s+(?:\w+\s+){0,3}(?:инструкци\w*|указани\w*|правил\w*|інструкці\w*|промпт\w*)"
            r"|\bзабудь\s+(?:всё|все)\b"
            # Chinese / Japanese / Korean
            r"|(?:忽略|无视|無視|忘记|忘記|不要理会|不要遵守)(?:之前|以上|上面|前面|先前|所有|全部|你的|一切)?\S{0,6}(?:指令|指示|说明|說明|规则|規則|提示)"
            r"|(?:以前|前|上記|これまで)の(?:指示|命令|ルール)を(?:すべて|全て)?(?:無視|忘れ)"
            r"|(?:이전|위의|모든)\s*(?:지시|명령|지침)\S*\s*(?:무시|잊어)"
        ),
        0.85,
        "tells the model to ignore its instructions (non-English)",
    ),
    # ---------------------------------------------------------------- new task / new instructions
    Rule(
        "override.new_instructions",
        "new_task",
        _r(
            r"\b(?:new|updated|revised|real|actual|true|secret|hidden|additional|important|priority|override)\s+"
            r"(?:system\s+)?(?:instructions?|directives?|task|orders|system\s+prompt)\s*(?:[:\-–]|follow|for\s+(?:the\s+)?(?:ai|assistant|model|agent))"
        ),
        0.55,
        "announces new instructions",
        untrusted_weight=0.7,
    ),
    Rule(
        "override.your_task_is",
        "new_task",
        _r(
            r"\byour\s+(?:new|real|actual|true|only|next|primary)\s+(?:task|job|goal|objective|mission|instructions?|purpose)"
            r"\s+(?:is|are|will\s+be|now)\b"
            r"|\b(?:from\s+now\s+on|henceforth|starting\s+now|for\s+the\s+rest\s+of\s+(?:this|the)\s+conversation)"
            r",?\s+(?:you|your|the\s+(?:assistant|ai|model))\s+(?:are|will|must|shall|should|have\s+to|can|need)\b"
        ),
        0.6,
        "redefines the model's task",
    ),
    Rule(
        "override.new_task_multilingual",
        "new_task",
        _r(
            r"\b(?:neue|neuen)\s+(?:anweisung|aufgabe|anweisungen)\s*:|\bab\s+(?:jetzt|sofort)\s+(?:bist\s+du|wirst\s+du|sollst\s+du)"
            r"|\b(?:nuevas?\s+instrucci[oó]n(?:es)?|nueva\s+tarea)\s*:|\ba\s+partir\s+de\s+ahora\s+(?:eres|ser[aá]s|debes)"
            r"|\b(?:nouvelles?\s+instructions?|nouvelle\s+t[aâ]che)\s*:|\bà\s+partir\s+de\s+maintenant,?\s+tu\s+(?:es|dois|vas)"
            r"|(?:новая\s+(?:инструкция|задача)|новые\s+инструкции)\s*:|\bтеперь\s+ты\s+(?:—|-)?\s*\w+"
            r"|(?:新的指令|新指令|新任务|从现在开始你)"
        ),
        0.55,
        "announces new instructions (non-English)",
        untrusted_weight=0.7,
    ),
    # ---------------------------------------------------------------- role-play / jailbreak personas
    Rule(
        "jailbreak.dan",
        "jailbreak",
        _r(
            r"\bdo\s+anything\s+now\b"
            r"|\b(?:you\s+are(?:\s+now)?|pretend\s+to\s+be|act\s+as|respond\s+as|answer\s+as|questions\s+as|become)\s+"
            r"(?-i:DAN|STAN|DUDE|AIM|Mongo\s+Tom|BetterDAN|AntiGPT|EvilBOT|UnGPT|FreedomGPT)\b"
            r"|\b(?-i:DAN|STAN|DUDE)\s+mode\b|\bjailbr(?:eak|oken)\s+(?:mode|version|ai|assistant|enabled)"
            r"|\[?🔓\s*JAILBREAK\]?|\bAlways\s+Intelligent\s+and\s+Machiavellian\b"
        ),
        0.9,
        "uses a known jailbreak persona (DAN, AIM, ...)",
    ),
    Rule(
        "jailbreak.developer_mode",
        "jailbreak",
        _r(
            r"\b(?:developer|debug|god|admin|maintenance|sudo|unrestricted|unfiltered|uncensored|evil|chaos)\s+mode\s+"
            r"(?:enabled|activated|on|engaged|unlocked)\b|\b(?:enable|activate|enter|switch\s+to|turn\s+on)\s+"
            r"(?:developer|debug|god|sudo|unrestricted|unfiltered|uncensored|jailbreak)\s+mode\b"
        ),
        0.8,
        "asks the model to switch to an unrestricted mode",
    ),
    Rule(
        "jailbreak.no_rules",
        "jailbreak",
        _r(
            r"\b(?:you\s+(?:are|have\s+been)\s+(?:now\s+)?(?:free(?:d)?|freed|released|liberated|unbound)\s+(?:from|of)"
            r"|you\s+(?:have|has)\s+no\s+(?:rules|restrictions|limits|limitations|filters|guidelines|ethics|morals|boundaries)"
            r"|(?:were|was|have\s+been)\s+never\s+given\s+any\s+(?:rules|restrictions|guidelines|instructions)"
            r"|\b(?:ai|bot|assistant|model|chatbot|robot|persona|character)\b[^.\n]{0,40}\b(?:has|have|with)\s+no\s+"
            r"(?:moral|ethical|content|safety)?\s*(?:or\s+(?:moral|ethical)\s+)?(?:rules|restrictions|limits|limitations|filters|guidelines|ethics|morals)"
            r"|(?:without|free\s+of|free\s+from|no\s+longer\s+(?:bound|restricted)\s+by)\s+(?:any\s+)?"
            r"(?:openai|anthropic|content|ethical|moral|safety)?\s*(?:rules|restrictions|limitations|filters|guidelines|policies|censorship)"
            r"|(?:pretend|imagine|act\s+as\s+if|behave\s+as\s+if)\s+(?:that\s+)?you\s+(?:are\s+not\s+bound|have\s+no|don'?t\s+have\s+(?:any\s+)?)"
            r"(?:\s*by)?\s*(?:rules|restrictions|limitations|filters|guidelines|ethics|policies)?"
            r"|(?:never|do\s+not|don'?t)\s+(?:refuse|decline|say\s+(?:no|you\s+can'?t)|add\s+(?:any\s+)?(?:warnings|disclaimers))"
            r"|(?:ignore|bypass|disable|turn\s+off)\s+(?:your\s+|all\s+|the\s+)?(?:safety|content|ethical)\s+(?:filters?|guidelines|policies|restrictions))"
        ),
        0.6,
        "tries to remove the model's restrictions",
    ),
    Rule(
        "jailbreak.roleplay_unrestricted",
        "jailbreak",
        _r(
            r"\b(?:act|behave|respond|answer|roleplay|role-play|pretend)\s+(?:as|like|to\s+be)\s+(?:an?\s+)?"
            r"(?:unrestricted|unfiltered|uncensored|amoral|unethical|evil|jailbroken|rogue|malicious|unaligned)\b"
            r"|\b(?:stay|remain)\s+in\s+character\b|\bbreak(?:ing)?\s+character\b"
            r"|\b(?:if\s+you\s+were|imagine\s+you\s+(?:are|were)|pretend\s+(?:that\s+)?you\s+are)\s+(?:an?\s+)?"
            r"(?:evil|unrestricted|unfiltered|uncensored|amoral|rogue|jailbroken|malicious)\s+(?:ai|assistant|model|chatbot|version|bot)"
            r"|\b(?:answer|respond|reply)\b[^.\n]{0,30}\btwice\b[^.\n]{0,80}\b(?:unfiltered|uncensored|unrestricted|no\s+(?:rules|filters))"
            r"|\b(?:two|2|dual|both)\s+(?:different\s+)?(?:responses|answers|replies)\b[^.\n]{0,80}\b(?:normal|classic|filtered)\b"
        ),
        0.55,
        "role-play designed to drop the model's guardrails",
    ),
    Rule(
        "jailbreak.hypothetical_wrapper",
        "jailbreak",
        _r(
            r"\b(?:hypothetical(?:ly)?|fictional|imaginary)\s+(?:world|scenario|story|universe)\b[^.\n]{0,120}"
            r"\b(?:no\s+(?:rules|laws|restrictions|ethics)|anything\s+(?:is\s+)?(?:allowed|permitted|legal))"
        ),
        0.45,
        "fiction framing used to suspend the rules",
    ),
    # ---------------------------------------------------------------- system prompt extraction
    Rule(
        "extraction.system_prompt",
        "extraction",
        _r(
            r"\b(?:reveal|print|show|display|output|repeat|recite|dump|leak|disclose|share|tell\s+me|give\s+me|write\s+out|spell\s+out|list|translate|summari[sz]e)"
            r"\s+(?:me\s+)?(?:back\s+)?(?:all\s+)?(?:of\s+)?(?:your|the|any)\s+(?:full\s+|entire\s+|complete\s+|exact\s+|original\s+|hidden\s+|secret\s+|initial\s+|internal\s+|confidential\s+)*"
            r"(?:system\s+(?:prompt|message|instructions?)|(?:initial|original|hidden|secret|internal|developer|pre-?)\s*(?:prompt|instructions?|message|rules)"
            r"|instructions?\s+(?:above|you\s+(?:were|have\s+been)\s+given)|prompt\s+above|configuration|guidelines\s+you\s+follow)"
            r"|\b(?:print|show|output|repeat|reveal|give\s+me|tell\s+me|display|copy|paste)\b[^.\n]{0,40}\b(?:instructions|prompt|rules|configuration)\s+"
            r"(?:you\s+(?:were|have\s+been)\s+given|above|that\s+came\s+before|before\s+(?:this|our|my)\s+(?:conversation|chat|message))"
            r"|\b(?:wording|text|content|contents|copy|transcript)\s+of\s+(?:your|the)\s+(?:system\s+(?:message|prompt)|initial\s+(?:prompt|instructions)|instructions)"
        ),
        0.8,
        "tries to extract the system prompt",
    ),
    Rule(
        "extraction.repeat_above",
        "extraction",
        _r(
            r"\b(?:repeat|print|output|copy|echo)\s+(?:back\s+)?(?:all\s+|everything\s+|the\s+(?:text|words|lines|message)\s+)?"
            r"(?:above|before\s+this|that\s+came\s+before)(?:\s+(?:this\s+(?:line|message)|verbatim))?"
            r"|\bwhat\s+(?:are|were)\s+your\s+(?:exact\s+|original\s+|initial\s+|system\s+)+(?:instructions|rules|prompt)\b"
            r"|\bbegin(?:ning)?\s+(?:your\s+(?:reply|answer|response)\s+)?with\s+[\"'“]?you\s+are\b"
        ),
        0.7,
        "asks the model to repeat what precedes the conversation",
    ),
    Rule(
        "extraction.multilingual",
        "extraction",
        _r(
            r"\b(?:zeig\w*|gib|nenne|wiederhole)\s+(?:mir\s+)?(?:deine?n?|den)\s+(?:system-?\s*prompt|anweisungen|instruktionen)"
            r"|\b(?:muestra|revela|dime|repite)\s+(?:me\s+)?(?:tu|tus|el|las)\s+(?:prompt\s+del\s+sistema|instrucciones|system\s+prompt)"
            r"|\b(?:montre|révèle|affiche|répète)[-\s]+(?:moi\s+)?(?:ton|tes|le|les)\s+(?:prompt\s+syst[eè]me|instructions|consignes)"
            r"|(?:покажи|выведи|раскрой|повтори|напиши)\s+(?:мне\s+)?(?:свой|свои|твой|твои|системн\w+)\s+(?:промпт|инструкции|подсказк\w+|\w+\s+промпт)"
            r"|(?:显示|告诉我|输出|重复)(?:你的)?(?:系统提示|系统指令|初始指令|提示词)"
            r"|システムプロンプト(?:を|の内容を)(?:表示|教え|出力|見せ)"
        ),
        0.8,
        "tries to extract the system prompt (non-English)",
    ),
    # ---------------------------------------------------------------- fake chat delimiters / role tokens
    Rule(
        "delimiter.chat_tokens",
        "delimiter",
        _r(
            r"<\|(?:im_start|im_end|system|user|assistant|endoftext|start_header_id|end_header_id|eot_id)\|>"
            r"|\[/?INST\]|<<\s*/?SYS\s*>>|</?(?:system|instructions?|admin|developer)(?:_message|_prompt)?>"
            r"|<\s*/?\s*(?:end_of_turn|start_of_turn)\s*>"
            r"|#{2,}\s*(?:system|assistant|developer)\s*:"
        ),
        0.7,
        "contains fake chat-template tokens",
    ),
    Rule(
        "delimiter.role_header",
        "delimiter",
        _r(
            r"(?:^|\n|[\"'{(\[]\s*)(?:#{1,4}\s*|\*\*|\[)?\s*(?:system|developer|admin(?:istrator)?|assistant)\s*(?:message|prompt|note|override|instructions?)?"
            r"\s*(?:\*\*|\])?\s*:\s*(?=[^\n]{0,200}\b(?:ignore|instructions?|must|new\s+(?:task|rules?|instructions)|from\s+now\s+on|override|"
            r"system\s+prompt|you\s+(?:must|should|will|are\s+now)|do\s+not\s+tell|forward|reveal|authori[sz]ed|call|delete|admin\s+rights)\b)\S",
            re.IGNORECASE,
        ),
        0.35,
        "starts a line as if it were a system or assistant message",
        untrusted_weight=0.55,
    ),
    Rule(
        "delimiter.fake_end",
        "delimiter",
        _r(
            r"(?:^|\n)\s*[-=#*\[<]{2,}\s*(?:end|begin|start)\s+(?:of\s+)?(?:the\s+)?(?:email|document|context|user\s+input|input|data|conversation|system\s+prompt|instructions|untrusted)"
            r"|\b(?:end\s+of\s+(?:email|document|context|data|user\s+input))\s*[-=#*\]>.]*\s*(?:\n|$)"
        ),
        0.4,
        "fakes the end of the document to start a new instruction block",
        untrusted_weight=0.55,
    ),
    # ---------------------------------------------------------------- talking to the AI from inside content
    Rule(
        "addressed.to_ai",
        "addressed_to_ai",
        _r(
            r"\b(?:dear|attention|hey|hi|hello|note\s+to(?:\s+the)?|message\s+(?:to|for)(?:\s+the)?|instructions?\s+(?:to|for)(?:\s+the)?)\s+"
            r"(?:ai|a\.i\.|llm|assistant|chatbot|bot|agent|language\s+models?|gpt|chatgpt|claude|gemini|copilot)\b"
            r"|\b(?:info|information|note|message|instructions?|request|hint)\s+(?:for|to)\s+(?:the\s+|any\s+|all\s+)?(?:[\w-]+\s+)?"
            r"(?:ai|bots?|assistants?|chatbots?|llms?|models?|screeners?)\b(?=[^.\n]{0,10}:)"
            r"|\b(?:note|message|instructions?)\s+(?:for|to)\s+(?:any\s+|all\s+)?automated\s+\w+"
            r"|\bto\s+the\s+(?:ai|assistant|bot|chatbot|llm|model)\s*[:,]"
            r"|@(?:assistant|ai|bot|copilot|claude|gpt|chatgpt)\b"
            r"|\b(?:ai|llm)\s+(?:[\w-]+\s+)?(?:assistants?|agents?|models?|bots?)\s*:"
            r"|\bfor\s+(?:any\s+|all\s+)?(?:ai\s+)?(?:assistants?|chatbots?|bots?|llms?|language\s+models?)\b[^.\n:]{0,60}:"
            r"|\bif\s+you\s+are\s+(?:an?\s+)?(?:ai|a\.i\.|llm|large\s+language\s+model|language\s+model|assistant|chatbot|automated|bot)\b"
            r"|\b(?:ai|llm|assistant|agent|model|bot)\s+instructions?\s*:"
            r"|(?:^|\n)\s*(?:the\s+)?(?:ai|a\.i\.|llm|gpt|claude|ai\s+(?:assistant|agent|model)|(?:email|mail|support)\s+(?:assistant|agent|bot))\s*[:,]\s*\S"
            r"|\b(?:to|for)\s+(?:any|the)\s+(?:ai|llm|assistant|agent|bot)s?\s+(?:reading|processing|summari[sz]ing|parsing)\s+this"
            r"|\b(?:ai\s+)?(?:assistants?|agents?|chatbots?|bots?|llms?|language\s+models?|copilots?)\s+(?:that\s+)?"
            r"(?:answering|reading|processing|handling|responding\s+to|summari[sz]ing|parsing|read|answer|process|handle)\b"
            r"[^.\n]{0,80}\b(?:should|must|need\s+to|have\s+to|are\s+(?:required|expected|instructed)\s+to|shall)\b"
            r"|\b(?:support\s+|ai\s+|virtual\s+|chat\s*)?(?:assistants?|chatbots?|bots?)\s+(?:must|should|shall|are\s+required\s+to|need\s+to)\s+"
            r"(?:always\s+)?(?:include|send|forward|reveal|share|tell|ask|call|create|delete|ignore|output|reply|add)\b"
        ),
        0.3,
        "addresses the AI directly",
        untrusted_weight=0.75,
    ),
    Rule(
        "addressed.multilingual",
        "addressed_to_ai",
        _r(
            r"\b(?:liebe[rs]?|hallo|achtung|hinweis\s+(?:an|für)\s+(?:den|die))\s+(?:KI|assistent(?:in)?|chatbot)\b|\bwenn\s+du\s+eine\s+KI\s+bist"
            r"|\b(?:querido|estimado|atenci[oó]n)\s+(?:asistente|IA)\b|\bsi\s+eres\s+una?\s+(?:IA|inteligencia\s+artificial|asistente)"
            r"|\bsi\s+tu\s+es\s+une?\s+(?:IA|intelligence\s+artificielle|assistant)|\b(?:cher|ch[eè]re)\s+(?:assistant|IA)\b"
            r"|если\s+ты\s+(?:ии|нейросеть|языковая\s+модель|ассистент|бот)|(?:дорогой|уважаемый)\s+(?:ии|ассистент)"
            r"|(?:如果你是|致)(?:AI|人工智能|助手|语言模型)"
        ),
        0.3,
        "addresses the AI directly (non-English)",
        untrusted_weight=0.75,
    ),
    # ---------------------------------------------------------------- exfiltration and covert actions
    Rule(
        "exfil.send_everything",
        "exfiltration",
        _r(
            r"\b(?:forward|send|e-?mail|mail|upload|post|transmit|copy|bcc|cc|leak|exfiltrate|share|relay|deliver)\b[^.\n]{0,60}"
            r"\b(?:all|every|each|entire|whole|complete|full)\b[^.\n]{0,40}"
            r"\b(?:emails?|messages?|invoices?|documents?|files?|conversations?|chats?|history|data|records|contacts|"
            r"passwords?|credentials?|api\s+keys?|tokens?|secrets?|attachments?|inbox|customer\s+data)\b[^.\n]{0,60}"
            r"\b(?:to|an|à|a|an\s+die|на)\s+[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
        ),
        0.55,
        "asks to send bulk data to an email address",
        untrusted_weight=0.8,
    ),
    Rule(
        "exfil.send_to_address",
        "exfiltration",
        _r(
            r"\b(?:forward|send|e-?mail|upload|post|transmit|leak|exfiltrate|relay)\s+(?:it|this|them|that|these|everything|"
            r"(?:the|a|your|our|my)\s+(?:conversation|chat|summary|reply|answer|response|data|information|details|credentials?|"
            r"password|api\s+key|token|system\s+prompt|instructions|history))\b[^.\n]{0,40}\bto\s+(?:[\w.+-]+@[\w-]+(?:\.[\w-]+)+|https?://\S+)"
        ),
        0.4,
        "asks to send conversation data to an outside address",
        untrusted_weight=0.65,
    ),
    Rule(
        "exfil.markdown_beacon",
        "exfiltration",
        _r(
            r"!\[[^\]\n]{0,80}\]\(\s*https?://[^)\s]*(?:\{|\}|\$|<|>|%7B|%7D|\[|\]|\bINSERT\b|\bDATA\b|\bSUMMARY\b|\bCONVERSATION\b|\bHISTORY\b|\.\.\.)"
            r"|https?://[^\s)\"']*[?&][\w-]+=\s*(?:\{|\[|<|%7B)\s*(?:conversation|chat|summary|data|history|secret|email|messages?|user|password|context|info)",
        ),
        0.85,
        "contains an image or link URL with a placeholder for conversation data",
    ),
    Rule(
        "exfil.render_link",
        "exfiltration",
        _r(
            r"\b(?:append|add|include|insert|embed|render|put|place|end\s+(?:your|the)\s+(?:reply|answer|response)\s+with)\b[^.\n]{0,60}"
            r"(?:\bmarkdown\b|\bimage\b|!\[|\bpixel\b|\btracking\b|\blink\b|\burl\b)[^.\n]{0,100}"
            r"(?:\b(?:conversation|chat|summary|history|previous\s+messages|user'?s?\s+(?:data|messages?|info|email)|context|secret|password|"
            r"query\s+(?:string|parameter)|url[-\s]?encoded?)\b|https?://)"
        ),
        0.6,
        "asks the model to render a link or image that carries data",
        untrusted_weight=0.75,
    ),
    Rule(
        "exfil.conversation_dump",
        "exfiltration",
        _r(
            r"\b(?:send|e-?mail|forward|upload|post|include|put|attach|share|copy|paste)\b[^.\n]{0,60}\b(?:the\s+)?"
            r"(?:full|entire|whole|complete|all(?:\s+of)?(?:\s+the)?)\s+(?:conversation|chat|message|session)s?\s*(?:history|log|transcript|text)?\b"
        ),
        0.45,
        "asks to send the whole conversation somewhere",
        untrusted_weight=0.7,
    ),
    Rule(
        "exfil.credentials",
        "exfiltration",
        _r(
            r"\b(?:send|e-?mail|forward|paste|enter|give|share|type)\s+(?:\w+\s+){0,2}(?:passwords?|credentials|card\s+numbers?|cvv|pins?|"
            r"one-time\s+codes?|2fa\s+codes?|security\s+codes?)\b[^.\n]{0,40}\b(?:to|into|in|at)\b"
            r"|\bask\s+(?:the\s+)?(?:customer|user|them)\s+for\s+(?:their\s+)?(?:full\s+)?(?:card\s+number|cvv|password|pin)\b"
        ),
        0.4,
        "asks for passwords or card data to be sent somewhere",
        untrusted_weight=0.7,
    ),
    Rule(
        "exfil.bulk_records",
        "exfiltration",
        _r(
            r"\b(?:send|reply\s+with|give|provide|share|export|include|list|output|summari[sz]e)\b[^.\n]{0,60}\b(?:of\s+)?"
            r"(?:all|every|each)\s+(?:other\s+)?(?:customers?|users?|clients?|employees?|contacts?|orders?)\b"
        ),
        0.15,
        "asks for the records of every customer or user",
        untrusted_weight=0.55,
    ),
    Rule(
        "covert.hide_from_user",
        "covert",
        _r(
            r"\b(?:without|don'?t|do\s+not|never)\s+(?:ask(?:ing)?|tell(?:ing)?|inform(?:ing)?|notify(?:ing)?|alert(?:ing)?|"
            r"mention(?:ing)?|confirm(?:ing)?|let(?:ting)?\s+(?:the\s+)?(?:user|human|them)\s+know)\b[^.\n]{0,30}"
            r"\b(?:the\s+)?(?:user|human|recipient|customer|operator|anyone|them|him|her|this)\b"
            r"|\b(?:silently|secretly|quietly|covertly|discreetly|in\s+the\s+background)\s+"
            r"(?:forward|send|e-?mail|delete|create|call|run|execute|transfer|share|add|change|update|upload)\b"
            r"|\bdo\s+not\s+(?:mention|reveal|disclose|show)\s+(?:this|these)\s+(?:instructions?|message|note|text|step)"
            r"|\b(?:do\s+not|don'?t|never)\s+(?:mention|reveal|disclose)\b[^.\n]{0,15}\b(?:that\s+you|this|it)\b"
            r"|\b(?:don'?t|do\s+not|never)\s+ask\s+(?:for\s+|the\s+user\s+for\s+)?(?:confirmation|permission|approval)"
            r"|\bwithout\s+(?:asking\s+(?:for\s+)?)?(?:confirmation|permission|approval)\b"
        ),
        0.55,
        "asks to act behind the user's back",
        untrusted_weight=0.75,
    ),
    Rule(
        "covert.tool_coercion",
        "tool_coercion",
        _r(
            r"\b(?:call|invoke|use|run|execute|trigger)\s+(?:the\s+)?[`'\"]?[a-z]+_[a-z_]+[`'\"]?\s+(?:tool|function|action|api)\b"
            r"|\b(?:call|invoke|run|execute)\s+[`'\"]?[a-z]+_[a-z_]+[`'\"]?\s+(?:for|on|with)\s+(?:every|all|each)\b"
            r"|\b(?:call|invoke|run|execute)\s+[`'\"]?(?:send_email|send_mail|delete_\w+|transfer_\w+|create_\w+|update_\w+|"
            r"http_request|fetch_url|browse|execute_code|run_shell|shell)[`'\"]?\s*\("
        ),
        0.35,
        "tells the model which tool to call",
        untrusted_weight=0.6,
    ),
    # ---------------------------------------------------------------- authority claims
    Rule(
        "authority.fake_system",
        "authority",
        _r(
            r"\b(?:this\s+is\s+(?:a\s+message\s+from\s+)?|message\s+from\s+|instructions?\s+from\s+|on\s+behalf\s+of\s+)"
            r"(?:the\s+|your\s+)?(?:system|administrator|admin|developers?|openai|anthropic|google|security\s+team|it\s+department)\b"
            r"[^.\n]{0,60}\b(?:you\s+(?:must|should|are\s+required|need)|instructions?|override|update)"
            r"|\b(?:SYSTEM|ADMIN|SECURITY)\s+(?:OVERRIDE|UPDATE|NOTICE|ALERT|COMMAND|INSTRUCTION)S?\b"
        ),
        0.45,
        "claims to come from the system or the developers",
        untrusted_weight=0.6,
    ),
    Rule(
        "authority.urgent_ai",
        "authority",
        _r(
            r"\b(?:IMPORTANT|URGENT|CRITICAL|ATTENTION|MANDATORY|PRIORITY)\s*[:!]+[^.\n]{0,80}"
            r"\b(?:you\s+must|ignore|override|forward|do\s+not\s+(?:tell|inform)|assistant|AI|model)\b",
            re.UNICODE,
        ),
        0.3,
        "urgent command aimed at the model",
        untrusted_weight=0.5,
    ),
)

_CATEGORY = {rule.id: rule.category for rule in RULES}

# Mention cues: a quoted attack phrase inside a sentence about attacks is discussed, not issued.
_MENTION_CUE = re.compile(
    r"\b(?:such\s+as|like|e\.g\.|for\s+example|for\s+instance|phrases?|strings?|text|prompts?|attacks?|attempts?|"
    r"injections?|jailbreaks?|called|known\s+as|classic|famous|typical|common|example|keywords?|patterns?|detect\w*|"
    r"filter\w*|block\w*|regex|test\s+cases?|payloads?|wrote|says?|said|reads?|containing|contains?|"
    r"wie|como|comme|например|такие\s+как|例如)\b",
    re.IGNORECASE,
)
_QUOTES = "\"'“”‘’«»„`"


def _is_mention(text: str, start: int, end: int) -> bool:
    """The match is quoted and the sentence talks about such phrases (use-mention distinction)."""
    before = text[max(0, start - 3) : start]
    after = text[end : end + 3]
    quoted = any(q in before for q in _QUOTES) and any(q in after for q in _QUOTES)
    if not quoted:
        # A quote may open further back ("such as 'please ignore all previous instructions'").
        window = text[max(0, start - 40) : start]
        quoted = any(q in window for q in _QUOTES) and any(q in text[end : end + 40] for q in _QUOTES)
    if not quoted:
        return False
    return bool(_MENTION_CUE.search(text[max(0, start - 100) : start]))


_REPORTED = re.compile(
    r"\b(?:tells?|telling|told|asks?|asking|asked|instructs?|instructing|tricks?|tricking|convinces?|convincing|makes?|"
    r"making|gets?|getting|forces?|forcing|causes?|causing)\s+(?:the\s+|a\s+|an\s+|its\s+|your\s+)?"
    r"(?:model|ai|assistant|chatbot|llm|bot|system|agent)s?\s+(?:to|into)\s+$",
    re.IGNORECASE,
)


def _is_reported(text: str, start: int) -> bool:
    """A description of an attack ("a user tells the model to disregard its instructions"), not an instruction."""
    return bool(_REPORTED.search(text[max(0, start - 60) : start]))


@dataclass
class HeuristicResult:
    score: float
    findings: list[Finding] = field(default_factory=list)
    categories: dict[str, float] = field(default_factory=dict)


_VIEW_DISCOUNT = {"rot13": 0.9, "reversed": 0.9}
_OBFUSCATION_BONUS = 0.35
_MENTION_FACTOR = 0.3


def scan(text: str, context: Context = "input", *, max_chars: int = 50_000) -> HeuristicResult:
    """Score a text for injection or jailbreak intent with rules over normalized and decoded views."""
    text = text[:max_chars]
    normalized = normalize(text)
    findings: list[Finding] = []
    best: dict[str, float] = {}

    def add(rule: Rule, view_kind: str, view_text: str, match: re.Match[str], origin: tuple[int, int]) -> None:
        weight = rule.weight_for(context) * _VIEW_DISCOUNT.get(view_kind, 1.0)
        message = rule.message
        if view_kind in {"raw", "normalized"} and (
            _is_mention(view_text, match.start(), match.end()) or _is_reported(view_text, match.start())
        ):
            weight *= _MENTION_FACTOR
            message += " (quoted as an example, discounted)"
        findings.append(
            Finding(
                rule=rule.id,
                layer="heuristics",
                score=round(weight, 3),
                message=message if view_kind in {"raw", "normalized"} else f"{message} (hidden in {view_kind})",
                start=origin[0],
                end=origin[1],
                snippet=match.group(0)[:160],
                view=view_kind,
            )
        )
        best[rule.category] = max(best.get(rule.category, 0.0), weight)

    def overlaps_same_category(rule: Rule, origin: tuple[int, int], view_kind: str) -> bool:
        return any(
            f.view == view_kind
            and f.start is not None
            and f.end is not None
            and f.start < origin[1]
            and origin[0] < f.end
            and _CATEGORY.get(f.rule) == rule.category
            for f in findings
        )

    seen: set[tuple[str, int, int]] = set()
    for rule in RULES:
        for match in rule.pattern.finditer(normalized.text):
            origin = normalized.to_original(match.start(), match.end())
            raw_match = rule.pattern.search(text, max(0, origin[0] - 2), min(len(text), origin[1] + 2))
            kind = "raw" if raw_match else "normalized"
            key = (rule.id, *origin)
            if key not in seen and not overlaps_same_category(rule, origin, kind):
                seen.add(key)
                add(rule, kind, normalized.text, match, origin)

    hidden_found = False
    for view in decoded_views(text):
        view_text = normalize(view.text).text
        for rule in RULES:
            for match in rule.pattern.finditer(view_text):
                key = (rule.id + ":" + view.kind, view.start, view.end)
                if key in seen or overlaps_same_category(rule, (view.start, view.end), view.kind):
                    continue
                seen.add(key)
                hidden_found = True
                add(rule, view.kind, view_text, match, (view.start, view.end))

    if hidden_found:
        best["obfuscation"] = _OBFUSCATION_BONUS
        findings.append(
            Finding(
                rule="obfuscation.encoded_instruction",
                layer="heuristics",
                score=_OBFUSCATION_BONUS,
                message="an instruction was hidden with an encoding or invisible characters",
            )
        )
    counts = invisible_counts(text)
    payloads = find_hidden_payloads(text)
    if payloads:
        weight = 0.6
        best["invisible"] = max(best.get("invisible", 0.0), weight)
        for payload in payloads:
            findings.append(
                Finding(
                    rule="obfuscation.invisible_payload",
                    layer="heuristics",
                    score=weight,
                    message=f"text hidden in invisible Unicode {payload.kind}",
                    start=payload.start,
                    end=payload.end,
                    snippet=payload.text[:160],
                    view=payload.kind,
                )
            )
    elif counts.get("zero_width", 0) + counts.get("bidi", 0) >= 3 and _zero_width_inside_words(text):
        weight = 0.2
        best["invisible"] = weight
        findings.append(
            Finding(
                rule="obfuscation.zero_width",
                layer="heuristics",
                score=weight,
                message=f"{counts.get('zero_width', 0) + counts.get('bidi', 0)} invisible characters inside words",
            )
        )

    score = 1.0
    for weight in best.values():
        score *= 1.0 - weight
    return HeuristicResult(score=round(1.0 - score, 4), findings=findings, categories=best)


def _zero_width_inside_words(text: str) -> bool:
    """Zero-width characters between two letters (not the joiners inside emoji sequences)."""
    return bool(re.search(r"[^\W\d_][​‌‍⁠﻿­‪-‮⁦-⁩]+[^\W\d_]", text))
