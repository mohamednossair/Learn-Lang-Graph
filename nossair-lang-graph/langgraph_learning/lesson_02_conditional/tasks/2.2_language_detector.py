# =============================================================
# TASK 2.2 — Language Detector & Router
# Flow: START → detect_language → route → translator → END
# =============================================================
# Goal:
#   Detect the language of an input text (English/Arabic/French)
#   using keyword/character heuristics (no LLM needed).
#   Route to the appropriate translator node.
#
# State: {text: str, language: str, translation_note: str}
# =============================================================

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# ── STEP 1: State ─────────────────────────────────────────────

class LanguageState(TypedDict):
    text: str               # input text to detect
    language: str           # detected: english / arabic / french
    translation_note: str   # output note from translator node


# ── STEP 2: Detection Node ────────────────────────────────────
# TODO: Detect language by heuristics.
# Hints:
#   Arabic  — any character in range \u0600-\u06FF
#   French  — words like "bonjour", "merci", "je", "le", "la", "est"
#   English — default fallback

def detect_language(state: LanguageState) -> dict:
    # TODO: implement detection, return {"language": "..."}
    text = state["text"]
    
    # Check for Arabic characters (Unicode range \u0600-\u06FF)
    if any(0x0600 <= ord(char) <= 0x06FF for char in text):
        return {"language": "arabic"}
    
    # Check for French keywords
    text_lower = text.lower()
    french_keywords = ["bonjour", "merci", "je", "le", "la", "est"]
    if any(keyword in text_lower for keyword in french_keywords):
        return {"language": "french"}
    
    # Default to English
    return {"language": "english"}


# ── STEP 3: Translator Nodes ──────────────────────────────────

def english_handler(state: LanguageState) -> dict:
    # TODO: return translation_note confirming English detected
    return {"translation_note": "English detected. No translation needed."}


def arabic_handler(state: LanguageState) -> dict:
    # TODO: return translation_note — e.g. "Arabic detected. Would translate to English."
    return {"translation_note": "Arabic detected. Would translate to English."}


def french_handler(state: LanguageState) -> dict:
    # TODO: return translation_note — e.g. "French detected. Would translate to English."
    return {"translation_note": "French detected. Would translate to English."}


# ── STEP 4: Routing Function ──────────────────────────────────

def route_by_language(state: LanguageState) -> Literal["english_handler", "arabic_handler", "french_handler"]:
    # TODO: map state["language"] → node name
    language = state["language"]
    if language == "arabic":
        return "arabic_handler"
    elif language == "french":
        return "french_handler"
    else:
        return "english_handler"


# ── STEP 5: Build Graph ───────────────────────────────────────

graph_builder = StateGraph(LanguageState)

# TODO: add nodes, edges, conditional edges
graph_builder.add_node(detect_language)
graph_builder.add_node(english_handler)
graph_builder.add_node(arabic_handler)
graph_builder.add_node(french_handler)

graph_builder.add_edge(START, "detect_language")

graph_builder.add_conditional_edges("detect_language", route_by_language, {
    "english_handler": "english_handler",
    "arabic_handler": "arabic_handler",
    "french_handler": "french_handler",
})

graph_builder.add_edge("english_handler", END)
graph_builder.add_edge("arabic_handler", END)
graph_builder.add_edge("french_handler", END)

graph = graph_builder.compile()


# ── STEP 6: Test ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?",
        "مرحبا، كيف حالك؟",
    ]

    for text in tests:
        print("\n" + "=" * 55)
        print(f"Input: {text}")
        result = graph.invoke({"text": text, "language": "", "translation_note": ""})
        print(f"Language : {result['language']}")
        print(f"Note     : {result['translation_note']}")
