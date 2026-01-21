"""
Email Agent for Maccabi AI Orchestrator
Sends email summaries using Gmail API.
"""

import os
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Gmail API credentials path
GMAIL_CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH", "config/gmail_credentials.json")
GMAIL_TOKEN_PATH = "config/gmail_token.json"

# Default sender info
DEFAULT_SENDER_NAME = "מכבי AI Assistant"


# =============================================================================
# GMAIL SERVICE SETUP
# =============================================================================

def get_gmail_service():
    """
    Create Gmail API service with OAuth2 credentials.
    
    First time: Opens browser for authentication.
    After: Uses saved token.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    
    creds = None
    
    # Load existing token
    if os.path.exists(GMAIL_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_PATH, SCOPES)
    
    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(GMAIL_CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Gmail credentials not found at {GMAIL_CREDENTIALS_PATH}. "
                    "Please download from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save token for future use
        with open(GMAIL_TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)


# =============================================================================
# EMAIL FUNCTIONS
# =============================================================================

def create_message(
    to: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None
) -> dict:
    """
    Create an email message.
    
    Args:
        to: Recipient email address
        subject: Email subject
        body_text: Plain text body
        body_html: Optional HTML body
    
    Returns:
        Message dict ready for Gmail API
    """
    if body_html:
        message = MIMEMultipart('alternative')
        message.attach(MIMEText(body_text, 'plain', 'utf-8'))
        message.attach(MIMEText(body_html, 'html', 'utf-8'))
    else:
        message = MIMEText(body_text, 'plain', 'utf-8')
    
    message['to'] = to
    message['subject'] = subject
    
    # Encode for Gmail API
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    return {'raw': raw}


def send_email(
    to: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None
) -> dict[str, Any]:
    """
    Send an email via Gmail API.
    
    Args:
        to: Recipient email address
        subject: Email subject
        body_text: Plain text body
        body_html: Optional HTML body
    
    Returns:
        Dict with success status and message ID
    """
    try:
        service = get_gmail_service()
        message = create_message(to, subject, body_text, body_html)
        
        result = service.users().messages().send(
            userId='me',
            body=message
        ).execute()
        
        return {
            "success": True,
            "message_id": result.get('id'),
            "to": to,
            "subject": subject
        }
    
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": str(e),
            "to": to,
            "subject": subject
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to send email: {str(e)}",
            "to": to,
            "subject": subject
        }


# =============================================================================
# EMAIL TEMPLATES
# =============================================================================

def create_triage_summary_email(
    triage_result: dict,
    patient_symptoms: str
) -> tuple[str, str, str]:
    """
    Create email content for triage summary.
    
    Returns:
        Tuple of (subject, body_text, body_html)
    """
    urgency = triage_result.get('urgency_level', 'לא ידוע')
    action = triage_result.get('recommended_action', '')
    reasoning = triage_result.get('reasoning', '')
    symptoms = triage_result.get('symptoms_identified', [])
    red_flags = triage_result.get('red_flags', [])
    
    subject = f"סיכום טריאז' מכבי - רמת דחיפות: {urgency}"
    
    body_text = f"""
סיכום הערכת טריאז' - מכבי שירותי בריאות
=========================================

תסמינים שדווחו:
{patient_symptoms}

רמת דחיפות: {urgency}

המלצה: {action}

הערכה: {reasoning}

תסמינים שזוהו: {', '.join(symptoms) if symptoms else 'לא זוהו'}

דגלים אדומים: {', '.join(red_flags) if red_flags else 'אין'}

-----------------------------------------
⚠️ שים לב: הערכה זו אינה מחליפה ייעוץ רפואי מקצועי.
במקרה של ספק, פנה למוקד *3555 או לחדר מיון.

מכבי שירותי בריאות
"""
    
    body_html = f"""
<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; direction: rtl; }}
        .header {{ background: #0066cc; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .urgency {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
        .urgency.emergency {{ color: #cc0000; }}
        .urgency.urgent {{ color: #ff6600; }}
        .urgency.soon {{ color: #ffcc00; }}
        .urgency.routine {{ color: #00cc00; }}
        .section {{ margin: 15px 0; padding: 10px; background: #f5f5f5; }}
        .warning {{ background: #fff3cd; padding: 10px; border-right: 4px solid #ffc107; }}
        .footer {{ margin-top: 20px; padding-top: 10px; border-top: 1px solid #ccc; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 סיכום הערכת טריאז' - מכבי שירותי בריאות</h1>
    </div>
    <div class="content">
        <div class="section">
            <h3>📝 תסמינים שדווחו:</h3>
            <p>{patient_symptoms}</p>
        </div>
        
        <div class="urgency {triage_result.get('urgency_code', '').lower()}">
            {triage_result.get('icon', '')} רמת דחיפות: {urgency}
        </div>
        
        <div class="section">
            <h3>📋 המלצה:</h3>
            <p><strong>{action}</strong></p>
        </div>
        
        <div class="section">
            <h3>💭 הערכה:</h3>
            <p>{reasoning}</p>
        </div>
        
        <div class="section">
            <h3>🔍 תסמינים שזוהו:</h3>
            <p>{', '.join(symptoms) if symptoms else 'לא זוהו'}</p>
        </div>
        
        {"<div class='section'><h3>🚩 דגלים אדומים:</h3><p>" + ', '.join(red_flags) + "</p></div>" if red_flags else ""}
        
        <div class="warning">
            ⚠️ <strong>שים לב:</strong> הערכה זו אינה מחליפה ייעוץ רפואי מקצועי.
            במקרה של ספק, פנה למוקד *3555 או לחדר מיון.
        </div>
        
        <div class="footer">
            <p>מכבי שירותי בריאות | www.maccabi4u.co.il | *3555</p>
        </div>
    </div>
</body>
</html>
"""
    
    return subject, body_text, body_html


def create_rag_summary_email(
    query: str,
    answer: str,
    sources: list[dict]
) -> tuple[str, str, str]:
    """
    Create email content for RAG query summary.
    
    Returns:
        Tuple of (subject, body_text, body_html)
    """
    subject = f"תשובה לשאלתך - מכבי AI"
    
    sources_text = "\n".join([f"  • {s['file']}" for s in sources]) if sources else "לא זמין"
    
    body_text = f"""
תשובה לשאלתך - מכבי שירותי בריאות
==================================

השאלה שלך:
{query}

התשובה:
{answer}

מקורות:
{sources_text}

-----------------------------------------
לפרטים נוספים, פנה למוקד *3555

מכבי שירותי בריאות
"""
    
    body_html = f"""
<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; direction: rtl; }}
        .header {{ background: #0066cc; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .section {{ margin: 15px 0; padding: 10px; background: #f5f5f5; }}
        .footer {{ margin-top: 20px; padding-top: 10px; border-top: 1px solid #ccc; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 תשובה לשאלתך - מכבי שירותי בריאות</h1>
    </div>
    <div class="content">
        <div class="section">
            <h3>❓ השאלה שלך:</h3>
            <p>{query}</p>
        </div>
        
        <div class="section">
            <h3>💬 התשובה:</h3>
            <p>{answer}</p>
        </div>
        
        <div class="section">
            <h3>📚 מקורות:</h3>
            <ul>
                {"".join([f"<li>{s['file']}</li>" for s in sources]) if sources else "<li>לא זמין</li>"}
            </ul>
        </div>
        
        <div class="footer">
            <p>לפרטים נוספים, פנה למוקד *3555</p>
            <p>מכבי שירותי בריאות | www.maccabi4u.co.il</p>
        </div>
    </div>
</body>
</html>
"""
    
    return subject, body_text, body_html


# =============================================================================
# HIGH-LEVEL FUNCTIONS
# =============================================================================

def send_triage_summary(
    to: str,
    triage_result: dict,
    patient_symptoms: str
) -> dict[str, Any]:
    """Send triage summary email."""
    subject, body_text, body_html = create_triage_summary_email(
        triage_result, patient_symptoms
    )
    return send_email(to, subject, body_text, body_html)


def send_rag_summary(
    to: str,
    query: str,
    answer: str,
    sources: list[dict]
) -> dict[str, Any]:
    """Send RAG query summary email."""
    subject, body_text, body_html = create_rag_summary_email(
        query, answer, sources
    )
    return send_email(to, subject, body_text, body_html)


# =============================================================================
# LANGGRAPH NODE FUNCTION
# =============================================================================

def email_agent_node(state: dict) -> dict:
    """
    LangGraph node function for email agent.
    
    Expected state:
        - email_to: str (recipient address)
        - email_type: str ("triage" or "rag")
        - triage_result: dict (if type is "triage")
        - symptoms: str (if type is "triage")
        - rag_response: dict (if type is "rag")
        - query: str (if type is "rag")
    
    Returns updated state with:
        - email_result: dict (success status)
    """
    email_to = state.get("email_to", "")
    email_type = state.get("email_type", "")
    
    if not email_to:
        return {
            **state,
            "email_result": {
                "success": False,
                "error": "לא צוינה כתובת מייל"
            }
        }
    
    if email_type == "triage":
        result = send_triage_summary(
            to=email_to,
            triage_result=state.get("triage_result", {}),
            patient_symptoms=state.get("symptoms", "")
        )
    elif email_type == "rag":
        rag_response = state.get("rag_response", {})
        result = send_rag_summary(
            to=email_to,
            query=state.get("query", ""),
            answer=rag_response.get("answer", ""),
            sources=rag_response.get("sources", [])
        )
    else:
        result = {
            "success": False,
            "error": f"סוג מייל לא מוכר: {email_type}"
        }
    
    return {
        **state,
        "email_result": result
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def test_email_setup():
    """Test Gmail API setup."""
    print("\n🔧 בודק הגדרות Gmail API...")
    
    if not os.path.exists(GMAIL_CREDENTIALS_PATH):
        print(f"\n❌ קובץ credentials לא נמצא: {GMAIL_CREDENTIALS_PATH}")
        print("\nלהגדרת Gmail API:")
        print("1. צור פרויקט ב-Google Cloud Console")
        print("2. הפעל Gmail API")
        print("3. צור OAuth 2.0 credentials")
        print(f"4. הורד את הקובץ ושמור אותו כ: {GMAIL_CREDENTIALS_PATH}")
        return False
    
    try:
        service = get_gmail_service()
        print("✅ Gmail API מוגדר בהצלחה!")
        return True
    except Exception as e:
        print(f"❌ שגיאה: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Maccabi Email Agent")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test Gmail API setup"
    )
    parser.add_argument(
        "--send-test",
        type=str,
        metavar="EMAIL",
        help="Send test email to specified address"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_email_setup()
    elif args.send_test:
        print(f"\n📧 שולח מייל בדיקה ל-{args.send_test}...")
        result = send_email(
            to=args.send_test,
            subject="בדיקת מערכת - מכבי AI",
            body_text="זהו מייל בדיקה ממערכת מכבי AI.\n\nאם קיבלת מייל זה, המערכת עובדת כראוי!",
            body_html="<h1>בדיקת מערכת</h1><p>זהו מייל בדיקה ממערכת מכבי AI.</p><p>אם קיבלת מייל זה, המערכת עובדת כראוי! ✅</p>"
        )
        if result["success"]:
            print(f"✅ נשלח בהצלחה! Message ID: {result['message_id']}")
        else:
            print(f"❌ שגיאה: {result['error']}")
    else:
        test_email_setup()