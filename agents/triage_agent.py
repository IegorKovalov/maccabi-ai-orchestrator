"""
Triage Agent for Maccabi AI Orchestrator
Assesses patient symptoms and determines urgency level.
"""

import os
from typing import Any
from enum import Enum

import anthropic
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


# =============================================================================
# URGENCY LEVELS
# =============================================================================

class UrgencyLevel(Enum):
    EMERGENCY = "חירום"      # Call ambulance / go to ER immediately
    URGENT = "דחוף"          # Go to urgent care today
    SOON = "בהקדם"           # See doctor within 1-2 days
    ROUTINE = "רגיל"         # Schedule regular appointment


URGENCY_ACTIONS = {
    UrgencyLevel.EMERGENCY: {
        "action": "פנה מיידית לחדר מיון או התקשר 101",
        "icon": "🚨",
        "color": "red"
    },
    UrgencyLevel.URGENT: {
        "action": "פנה היום למרפאה דחופה או התקשר *3555",
        "icon": "⚠️",
        "color": "orange"
    },
    UrgencyLevel.SOON: {
        "action": "קבע תור לרופא המשפחה בימים הקרובים",
        "icon": "📅",
        "color": "yellow"
    },
    UrgencyLevel.ROUTINE: {
        "action": "קבע תור רגיל לרופא המשפחה",
        "icon": "✅",
        "color": "green"
    }
}


# =============================================================================
# TRIAGE PROMPT
# =============================================================================

TRIAGE_SYSTEM_PROMPT = """אתה מערכת טריאז' רפואית של מכבי שירותי בריאות. תפקידך להעריך את דחיפות הפנייה הרפואית על בסיס התסמינים שמתאר המטופל.

חשוב מאוד:
- אתה לא מאבחן מחלות
- אתה רק מעריך דחיפות ומפנה לשירות המתאים
- תמיד טעה לכיוון הזהירות - אם יש ספק, העלה את רמת הדחיפות
- זכור שאתה לא מחליף רופא

רמות דחיפות:

1. חירום (EMERGENCY) - פנייה מיידית לחדר מיון או 101:
   - כאבים בחזה, קוצר נשימה חמור
   - חשד לשבץ (חולשה בצד אחד, קושי בדיבור)
   - אובדן הכרה, פרכוסים
   - דימום חמור שלא נעצר
   - תגובה אלרגית חמורה
   - חום מעל 40 מעלות
   - פציעות ראש עם אובדן הכרה

2. דחוף (URGENT) - פנייה היום למרפאה דחופה:
   - חום גבוה (38.5-40)
   - כאבים חזקים (אוזניים, גרון, בטן)
   - פציעות הדורשות תפירה
   - הקאות או שלשולים ממושכים
   - חשד לזיהום בדרכי השתן

3. בהקדם (SOON) - תור לרופא תוך יום-יומיים:
   - תסמינים שנמשכים מספר ימים
   - כאבים מתונים
   - פריחה לא חמורה
   - שיעול ממושך

4. רגיל (ROUTINE) - תור רגיל לרופא:
   - בדיקות שגרתיות
   - מעקב מחלה כרונית
   - חידוש מרשמים
   - ייעוץ כללי

עליך להשיב בפורמט JSON בלבד:
{
    "urgency": "EMERGENCY" | "URGENT" | "SOON" | "ROUTINE",
    "confidence": 0.0-1.0,
    "reasoning": "הסבר קצר בעברית",
    "symptoms_identified": ["תסמין 1", "תסמין 2"],
    "red_flags": ["דגל אדום אם יש"],
    "questions": ["שאלת הבהרה אם נדרש"]
}"""


# =============================================================================
# TRIAGE AGENT
# =============================================================================

def assess_symptoms(symptoms: str) -> dict[str, Any]:
    """
    Assess patient symptoms and determine urgency level.
    
    Args:
        symptoms: Patient's description of symptoms in Hebrew
    
    Returns:
        Dict with urgency level, reasoning, and recommended action
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=TRIAGE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"המטופל מתאר את התסמינים הבאים:\n\n{symptoms}"
            }
        ]
    )
    
    # Parse JSON response
    import json
    try:
        result_text = response.content[0].text
        # Clean up potential markdown formatting
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result = json.loads(result_text.strip())
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        result = {
            "urgency": "URGENT",
            "confidence": 0.5,
            "reasoning": "לא הצלחתי לנתח את התסמינים. מומלץ לפנות לייעוץ רפואי.",
            "symptoms_identified": [],
            "red_flags": [],
            "questions": []
        }
    
    # Map urgency string to enum
    urgency_map = {
        "EMERGENCY": UrgencyLevel.EMERGENCY,
        "URGENT": UrgencyLevel.URGENT,
        "SOON": UrgencyLevel.SOON,
        "ROUTINE": UrgencyLevel.ROUTINE
    }
    
    urgency = urgency_map.get(result.get("urgency", "URGENT"), UrgencyLevel.URGENT)
    action_info = URGENCY_ACTIONS[urgency]
    
    return {
        "urgency_level": urgency.value,
        "urgency_code": urgency.name,
        "confidence": result.get("confidence", 0.5),
        "reasoning": result.get("reasoning", ""),
        "symptoms_identified": result.get("symptoms_identified", []),
        "red_flags": result.get("red_flags", []),
        "questions": result.get("questions", []),
        "recommended_action": action_info["action"],
        "icon": action_info["icon"],
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens
    }


def format_triage_response(result: dict) -> str:
    """Format triage result as readable Hebrew text."""
    output = []
    
    output.append(f"\n{result['icon']} רמת דחיפות: {result['urgency_level']}")
    output.append(f"\n📋 המלצה: {result['recommended_action']}")
    
    if result['reasoning']:
        output.append(f"\n💭 הערכה: {result['reasoning']}")
    
    if result['symptoms_identified']:
        output.append(f"\n🔍 תסמינים שזוהו: {', '.join(result['symptoms_identified'])}")
    
    if result['red_flags']:
        output.append(f"\n🚩 דגלים אדומים: {', '.join(result['red_flags'])}")
    
    if result['questions']:
        output.append(f"\n❓ שאלות להבהרה:")
        for q in result['questions']:
            output.append(f"   • {q}")
    
    output.append(f"\n\n⚠️ שים לב: הערכה זו אינה מחליפה ייעוץ רפואי מקצועי.")
    output.append(f"במקרה של ספק, פנה למוקד *3555 או לחדר מיון.")
    
    return "\n".join(output)


# =============================================================================
# LANGGRAPH NODE FUNCTION
# =============================================================================

def triage_agent_node(state: dict) -> dict:
    """
    LangGraph node function for triage agent.
    
    Expected state:
        - symptoms: str (patient's symptoms description)
    
    Returns updated state with:
        - triage_result: dict (urgency assessment)
    """
    symptoms = state.get("symptoms", "")
    
    if not symptoms:
        return {
            **state,
            "triage_result": {
                "urgency_level": "לא ידוע",
                "recommended_action": "אנא תאר את התסמינים שלך",
                "error": "לא התקבלו תסמינים"
            }
        }
    
    result = assess_symptoms(symptoms)
    
    return {
        **state,
        "triage_result": result
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def interactive_mode():
    """Run interactive triage session."""
    print("\n" + "=" * 60)
    print("🏥 מכבי AI - מערכת טריאז' רפואית")
    print("=" * 60)
    print("תאר את התסמינים שלך בעברית (או 'exit' ליציאה)")
    print("⚠️ מערכת זו אינה מחליפה ייעוץ רפואי מקצועי\n")
    
    while True:
        symptoms = input("🤒 תסמינים: ").strip()
        
        if symptoms.lower() in ['exit', 'quit', 'יציאה']:
            print("\n👋 להתראות! שמור על בריאותך!")
            break
        
        if not symptoms:
            continue
        
        print("\n🔄 מעריך תסמינים...")
        result = assess_symptoms(symptoms)
        
        print(format_triage_response(result))
        print(f"\n📊 טוקנים: {result['tokens_used']}")
        print("-" * 40 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Maccabi Triage Agent")
    parser.add_argument(
        "--symptoms",
        type=str,
        help="Symptoms to assess"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive triage session"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.symptoms:
        result = assess_symptoms(args.symptoms)
        print(format_triage_response(result))
    else:
        # Default: interactive mode
        interactive_mode()