import re
VISION_INTENTS = {"scene", "object", "text", "color", "currency"}
CONVERSATIONAL_INTENTS = {"greeting", "capabilities", "help", "general_chat"}
INTENT_PATTERNS = [
    (
        "capabilities",
        [
            r"\bwhat can you do\b", r"\bcapabilities\b", r"\bfeatures\b",
            r"تقدر تعمل ايه", r"بتعمل ايه", r"وظايفك", r"مين انت",
            r"ايه قدراتك", r"بتعرف تعمل ايه",r"انت اي",
        ],
    ),
    (
        "help",
        [
            r"\bhelp\b", r"\bassist\b", r"ساعد", r"مساعده",
            r"ازاي استخدمك", r"الاوامر",
        ],
    ),
    (
        "greeting",
        [
            r"\bhello\b", r"\bhi\b", r"\bhey\b",
            r"اهلا", r"مرحبا", r"ازيك", r"سلام", r"صباح الخير", r"مساء الخير",
        ],
    ),
    (
        "currency",
        [
            r"\bcurrency\b", r"\bmoney\b", r"\bcash\b",
            r"فلوس", r"جنيه", r"كام جنيه", r"عمله", r"نقود",
        ],
    ),
    (
        "text",
        [
            r"\bread\b", r"\bocr\b", r"\btext\b",
            r"اقرا", r"اقرالي", r"مكتوب", r"لافتة", r"ورقة", r"رسالة",
        ],
    ),
    (
        "color",
        [
            r"\bcolor\b", r"\bcolour\b",
            r"لون", r"ايه اللون", r"الوان",
        ],
    ),
    (
        "scene",
        [
            r"\baround me\b", r"\bscene\b",
            r"وصف المكان", r"وصف المشهد", r"قدامي ايه",
            r"حوالي ايه", r"فين انا", r"المكان عامل ازاي",
        ],
    ),
    (
        "object",
        [
            r"\bobject\b", r"\bidentify\b", r"\bdetect\b",
            r"ايه ده", r"ده ايه", r"دي ايه",
            r"شايف ايه", r"فيه ايه", r"ايه الموجود",
        ],
    ),
]
CONTEXT_WORDS = re.compile(
    r"\b(it|this|that|these|those|same|again|another|them)\b"
    r"|(ده|دي|دول|كمان|تاني|نفسه|نفسها)",
    re.IGNORECASE,
)
_AR_NORMALIZE = str.maketrans("أإآةىؤئ", "اااهيوي")
def normalize_arabic(text):
    text = text.lower().strip()
    text = text.translate(_AR_NORMALIZE)
    text = re.sub(r"\s+", " ", text)
    return text
def detect_language(text):
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    english = len(re.findall(r"[a-zA-Z]", text))
    return "ar" if arabic >= english else "en"
INTENT_PATTERNS = [
    ("greeting", [r"ازيك", r"مرحبا", r"سلام", r"\bhi\b", r"\bhello\b"]),
    ("capabilities", [r"تقدر تعمل", r"قدراتك", r"وظيفتك", r"مين انت"]),
    ("help", [r"مساعد", r"help", r"ازاي استخدمك"]),
    ("currency_query", [r"كام جنيه", r"دي كام", r"المبلغ كام"]),
    ("currency", [r"فلوس", r"عملة", r"نقود"]),
    ("text", [r"اقرا", r"مكتوب", r"لافتة", r"ورقة"]),
    ("color", [r"لون", r"الوان"]),
    ("scene", [r"قدامي", r"حوالي", r"المكان", r"المشهد"]),
    ("object", [r"ايه ده", r"دي ايه", r"شايف ايه", r"فيه ايه"]),
]
def score_intent(text, patterns):
    return sum(1 for p in patterns if re.search(p, text))
def needs_context(query):
    text = normalize_arabic(query)
    return len(text.split()) <= 2
def classify_intent(query):
    lang = detect_language(query)
    text = normalize_arabic(query)
    if "كام" in text and "جنيه" in text:
        return "currency_query", lang
    scores = {}
    for intent, patterns in INTENT_PATTERNS:
        scores[intent] = score_intent(text, patterns)
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    if best_score == 0:
        if needs_context(text):
            return "object", lang
        return "general_chat", lang
    return best_intent, lang
def classify(query):
    intent, lang = classify_intent(query)
    return {
        "intent": intent,
        "lang": lang
    }