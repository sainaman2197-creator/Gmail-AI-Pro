"""
AI Summarizer, Spam Filter & Executive Digest Engine
Generates individual email summaries, line-by-line important highlights,
intelligent Spam & Promotion isolation, and comprehensive Master AI Digests 
across all inbox messages.
Supports fast built-in NLP heuristics as well as Google Gemini AI API.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Support both new Google GenAI SDK and legacy SDK
HAS_GEMINI = False
try:
    from google import genai
    HAS_GEMINI = True
    SDK_TYPE = "new"
except ImportError:
    try:
        import google.generativeai as genai_legacy
        HAS_GEMINI = True
        SDK_TYPE = "legacy"
    except ImportError:
        HAS_GEMINI = False
        SDK_TYPE = None


def detect_spam_and_promotions(subject: str, body: str, sender: str) -> Dict[str, Any]:
    """
    Intelligently classify whether an email is Spam, Promotional, or Clean/Important.
    Returns:
        is_spam (bool),
        is_promo (bool),
        category ('Spam', 'Promotion', 'Clean'),
        spam_score (0-100),
        reason (str)
    """
    sub_lower = subject.lower()
    body_lower = body.lower()
    sender_lower = sender.lower()
    text = f"{sub_lower} {body_lower}"

    spam_score = 0
    promo_score = 0
    spam_reasons = []
    promo_reasons = []

    # High-confidence Spam / Phishing triggers (Scams, Phishing, Fake Lottery, Fraud)
    high_spam_triggers = [
        ("lottery", "Lottery or fake prize claim"),
        ("winner", "Claims recipient is a lottery winner"),
        ("won $", "Cash prize claim"),
        ("won 1000", "Cash prize claim"),
        ("claim your prize", "Unsolicited prize claim"),
        ("crypto investment return", "Crypto scam indicators"),
        ("bitcoin transfer", "Suspicious cryptocurrency transfer"),
        ("guaranteed profit", "Financial scam keyword"),
        ("wire transfer", "Urgent wire request"),
        ("nigerian prince", "Classic scam pattern"),
        ("viagra", "Pharmaceutical spam"),
        ("casino", "Gambling / Casino spam"),
        ("deposit 100", "Gambling / Scam pattern"),
        ("verify your bank account immediately", "Phishing phrase"),
        ("account suspended click link below", "Phishing phrase"),
        ("unauthorized login click here", "Phishing phrase"),
        ("you have inherited", "Inheritance scam"),
        ("make money fast", "Get rich quick scam"),
        ("work from home earn 5000", "Employment scam")
    ]

    for trigger, reason_text in high_spam_triggers:
        if trigger in text:
            spam_score += 40
            spam_reasons.append(reason_text)

    # Suspicious sender indicators for spam/scam
    if any(k in sender_lower for k in ["winner", "prize", "jackpot", "rewards-claim", "cashnow", "crypto"]):
        spam_score += 40
        spam_reasons.append("Suspicious scam sender address")

    # ALL CAPS subject detection with scam words
    words = [w for w in subject.split() if len(w) > 3]
    if words and sum(1 for w in words if w.isupper()) / len(words) > 0.6 and spam_score > 0:
        spam_score += 20
        spam_reasons.append("Subject is heavily uppercase shouting")

    # Commercial / Promotional triggers (Newsletters, Deals, Discounts)
    promo_triggers = [
        ("unsubscribe", "Contains marketing unsubscribe link"),
        ("click here to opt-out", "Contains opt-out link"),
        ("view in browser", "Standard marketing email header"),
        ("50% off", "Discount / promotional offer"),
        ("limited time offer", "Marketing urgency"),
        ("flash sale", "Promotional flash sale"),
        ("special discount", "Promotional discount"),
        ("buy now", "Sales pitch"),
        ("free gift", "Promotional gift"),
        ("exclusive deal", "Promotional deal"),
        ("coupon code", "Coupon marketing"),
        ("shop our", "Retail marketing"),
        ("newsletter digest", "Newsletter publication"),
        ("weekly digest", "Weekly digest"),
        ("promotions@", "Marketing sender address"),
        ("marketing@", "Marketing sender address"),
        ("newsletter@", "Newsletter sender address")
    ]

    for trigger, reason_text in promo_triggers:
        if trigger in text or trigger in sender_lower:
            promo_score += 20
            if reason_text not in promo_reasons:
                promo_reasons.append(reason_text)

    # Final Classification
    if spam_score >= 40:
        return {
            "is_spam": True,
            "is_promo": False,
            "category": "🚫 Spam / Phishing",
            "spam_score": min(100, spam_score),
            "reasons": spam_reasons if spam_reasons else ["Suspicious scam / phishing patterns detected"],
            "badge_color": "#d32f2f"
        }
    elif promo_score >= 20 or "unsubscribe" in text:
        return {
            "is_spam": False,
            "is_promo": True,
            "category": "📢 Promotion / Newsletter",
            "spam_score": min(40, promo_score),
            "reasons": promo_reasons if promo_reasons else ["Promotional email / Subscription newsletter"],
            "badge_color": "#f57c00"
        }
    else:
        return {
            "is_spam": False,
            "is_promo": False,
            "category": "🟢 Clean & Important",
            "spam_score": 0,
            "reasons": ["Legitimate direct communication"],
            "badge_color": "#2e7d32"
        }


def classify_topic(subject: str, body: str) -> str:
    """Classify the email into a distinct business topic category."""
    text = f"{subject} {body}".lower()
    
    if any(w in text for w in ["interview", "job", "career", "hiring", "resume", "cv", "recruiter", "selected", "offer letter", "assessment"]):
        return "💼 Career & Hiring"
    elif any(w in text for w in ["invoice", "payment", "bill", "billing", "receipt", "due date", "subscription", "price", "usd", "inr", "$"]):
        return "💳 Finance & Billing"
    elif any(w in text for w in ["meeting", "schedule", "google meet", "zoom", "teams", "calendar", "appointment", "sync up", "discussion"]):
        return "📅 Meeting & Scheduling"
    elif any(w in text for w in ["security", "password", "otp", "login", "ssh key", "verify", "authentication", "alert", "suspicious", "revoke"]):
        return "🔐 Security & Account"
    elif any(w in text for w in ["pull request", "github", "gitlab", "commit", "deployment", "bug", "jira", "sprint", "code review", "backend", "api", "database"]):
        return "🛠️ Tech & Development"
    elif any(w in text for w in ["newsletter", "digest", "weekly", "promo", "discount", "offer", "marketing", "webinar", "blog"]):
        return "📢 Newsletter & Marketing"
    else:
        return "📌 General Notification"


def extract_dates_and_deadlines(text: str) -> List[str]:
    """Extract dates, times, and day mentions from email text."""
    found_dates = []
    
    # Matches patterns like: 28th September, Sept 28, 2026-09-28, 28/09/2026, Friday, tomorrow, 2:00 PM
    date_patterns = [
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{2,4})?\b",
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,\s*\d{2,4})?\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}:\d{2}\s*(?:am|pm|ist|utc|gmt|est|pst)\b",
        r"\b(?:tomorrow|by eod|by end of day|eod|asap)\b"
    ]
    
    for pat in date_patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            clean_m = m.strip()
            if clean_m.lower() not in [d.lower() for d in found_dates] and len(clean_m) > 1:
                found_dates.append(clean_m)
                
    return found_dates[:4]


def extract_action_items_from_text(subject: str, body: str) -> List[str]:
    """Identify action items, requests, and to-dos in an email."""
    actions = []
    text = f"{subject}\n{body}"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    action_keywords = [
        "please", "kindly", "request", "action required", "need you to", "make sure to",
        "let us know", "availability", "review", "submit", "complete", "pay before", 
        "due date", "push your", "confirm", "respond", "send your", "join us"
    ]
    
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in action_keywords):
            clean_line = re.sub(r"^[\d\.\-\*\•\>\s]+", "", line).strip()
            if 10 < len(clean_line) < 220:
                if clean_line not in actions:
                    actions.append(clean_line)
                    
    return actions[:3]


def analyze_single_email(
    email_data: Dict[str, Any], 
    gemini_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a rich AI analysis card for a single email with spam detection and 1-line highlights."""
    subject = email_data.get("subject", "")
    body = email_data.get("body_plain", "") or email_data.get("body_html", "")
    sender = email_data.get("sender", "")
    
    # Check Spam & Promotion Classification
    spam_info = detect_spam_and_promotions(subject, body, sender)

    # Built-in High-Speed NLP Engine
    topic = classify_topic(subject, body)
    dates = extract_dates_and_deadlines(body)
    action_items = extract_action_items_from_text(subject, body)
    
    sub_lower = subject.lower()
    body_lower = body.lower()
    is_urgent = any(w in sub_lower or w in body_lower for w in ["urgent", "action required", "asap", "critical", "immediately", "deadline", "verification"])
    
    if spam_info["is_spam"]:
        urgency = "🚫 Low (Spam)"
    elif is_urgent:
        urgency = "🔥 High"
    elif action_items:
        urgency = "⚡ Medium"
    else:
        urgency = "🟢 Normal"
    
    # Generate concise 1-2 sentence executive summary
    clean_body = " ".join(body.split())
    if not clean_body:
        summary = "No message body provided in this email."
    else:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_body) if len(s.strip()) > 15]
        if sentences:
            summary = sentences[0]
            if len(summary) > 220:
                summary = summary[:217] + "..."
            if len(sentences) > 1 and len(summary) < 140:
                summary += " " + sentences[1][:120]
        else:
            summary = clean_body[:200]

    # Generate 1-Line Crisp Crucial Takeaway (Important Message Per Line)
    if spam_info["is_spam"]:
        one_line_important = f"⚠️ [SPAM SUSPECT] {spam_info['reasons'][0]} — Do not click links."
    elif action_items:
        one_line_important = f"🎯 Action: {action_items[0]}"
    elif is_urgent:
        one_line_important = f"🔥 Urgent Attention: {summary[:120]}"
    elif dates:
        one_line_important = f"📅 Key Timeline: Mentioned {dates[0]} — {summary[:90]}"
    else:
        one_line_important = f"ℹ️ Update: {summary[:120]}"

    # Key bullet points
    key_points = []
    if spam_info["is_spam"]:
        key_points.append(f"🚫 Flagged as Spam/Scam: {', '.join(spam_info['reasons'])}")
    elif spam_info["is_promo"]:
        key_points.append(f"📢 Promotional message / Marketing newsletter.")
    
    if is_urgent and not spam_info["is_spam"]:
        key_points.append("⚠️ Requires urgent attention or immediate review.")
    if action_items and not spam_info["is_spam"]:
        key_points.append(f"📌 Action Request: {action_items[0]}")
    if dates:
        key_points.append(f"📅 Mentioned Dates / Timeline: {', '.join(dates)}")
    if email_data.get("has_attachments"):
        key_points.append(f"📎 Contains {len(email_data.get('attachments', []))} attachment(s).")
    if not key_points:
        key_points.append("ℹ️ Standard informational notification.")

    return {
        "topic": topic,
        "summary": summary,
        "one_line_important": one_line_important,
        "urgency": urgency,
        "action_required": (len(action_items) > 0 or is_urgent) and not spam_info["is_spam"],
        "action_items": action_items if not spam_info["is_spam"] else [],
        "key_points": key_points,
        "detected_dates": dates,
        "spam_info": spam_info
    }


def generate_inbox_master_digest(
    emails: List[Dict[str, Any]], 
    gemini_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    GENERATE MASTER AI DIGEST:
    Analyzes ALL incoming Gmail messages collectively and produces:
    1. Executive Overview
    2. Line-by-Line Important Takeaways
    3. Spam & Promotions Segregation Matrix
    4. Master Action Items Matrix
    5. Category Distribution
    6. Urgent & Critical Attention Table
    7. Upcoming Deadlines & Dates
    8. Exportable Markdown Report
    """
    if not emails:
        return {
            "total_emails": 0,
            "executive_summary": "No emails found in the current view to summarize.",
            "clean_emails": [],
            "spam_emails": [],
            "promo_emails": [],
            "line_by_line_highlights": [],
            "action_items": [],
            "category_counts": {},
            "urgent_emails": [],
            "upcoming_dates": [],
            "top_senders": {},
            "markdown_report": "# Inbox AI Digest\n\nNo emails available."
        }

    analyzed_emails = []
    clean_emails = []
    spam_emails = []
    promo_emails = []
    line_by_line_highlights = []
    
    category_counts: Dict[str, int] = {}
    urgent_emails = []
    all_action_items = []
    all_dates = []
    sender_counts: Dict[str, int] = {}

    for mail in emails:
        analysis = analyze_single_email(mail, gemini_api_key=None)
        sender_clean = mail.get("sender", "Unknown").split("<")[0].strip() or mail.get("sender", "Unknown")
        
        email_record = {
            "id": mail.get("id"),
            "sender": sender_clean,
            "sender_raw": mail.get("sender"),
            "subject": mail.get("subject"),
            "date": mail.get("date"),
            "analysis": analysis,
            "unread": mail.get("unread", False),
            "has_attachments": mail.get("has_attachments", False)
        }
        analyzed_emails.append(email_record)

        # Line-by-line important highlight record
        line_by_line_highlights.append({
            "id": mail.get("id"),
            "subject": mail.get("subject"),
            "sender": sender_clean,
            "date": mail.get("date"),
            "category": analysis["topic"],
            "urgency": analysis["urgency"],
            "one_line_important": analysis["one_line_important"],
            "is_spam": analysis["spam_info"]["is_spam"],
            "is_promo": analysis["spam_info"]["is_promo"]
        })

        # Spam vs Promo vs Clean segregation
        if analysis["spam_info"]["is_spam"]:
            spam_emails.append(email_record)
        elif analysis["spam_info"]["is_promo"]:
            promo_emails.append(email_record)
        else:
            clean_emails.append(email_record)
        
        # Category counting
        cat = analysis["topic"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Senders counting
        sender_counts[sender_clean] = sender_counts.get(sender_clean, 0) + 1
        
        # Urgent emails (clean only)
        if ("High" in analysis["urgency"] or analysis["action_required"]) and not analysis["spam_info"]["is_spam"]:
            urgent_emails.append({
                "subject": mail.get("subject"),
                "sender": sender_clean,
                "date": mail.get("date"),
                "urgency": analysis["urgency"],
                "action": analysis["action_items"][0] if analysis["action_items"] else "Review email details",
                "topic": cat
            })
            
        # Action items aggregation (clean only)
        for act in analysis["action_items"]:
            all_action_items.append({
                "task": act,
                "from": sender_clean,
                "subject": mail.get("subject"),
                "topic": cat,
                "date": mail.get("date")
            })
            
        # Dates aggregation
        for d in analysis["detected_dates"]:
            if not analysis["spam_info"]["is_spam"]:
                all_dates.append({
                    "date_mention": d,
                    "subject": mail.get("subject"),
                    "from": sender_clean
                })

    # Generate Executive Summary narrative
    total = len(emails)
    clean_count = len(clean_emails)
    spam_count = len(spam_emails)
    promo_count = len(promo_emails)
    urgent_count = len(urgent_emails)
    top_cat = max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else "General"
    
    exec_summary = (
        f"📊 **Inbox Executive Briefing ({datetime.now().strftime('%d %b %Y, %I:%M %p')})**\n\n"
        f"Analyzed **{total} total messages**: **{clean_count} Important / Work emails**, "
        f"**{spam_count} Flagged Spam/Scam**, and **{promo_count} Newsletters/Promotions**.\n\n"
        f"🚨 **{urgent_count} emails** require your immediate attention. "
        f"The primary focus area in your inbox is **{top_cat}** ({category_counts.get(top_cat, 0)} emails). "
    )
    if all_action_items:
        exec_summary += f"We identified **{len(all_action_items)} key action items/tasks**. "
    if all_dates:
        exec_summary += f"There are **{len(all_dates)} scheduled deadlines/calendar dates** to track."

    # Generate Exportable Markdown Report
    report_lines = [
        "# 📬 Executive AI Inbox Digest & Line-by-Line Report",
        f"**Generated on:** {datetime.now().strftime('%A, %d %B %Y at %I:%M %p')}",
        f"**Total Analyzed:** {total} | **Work/Clean:** {clean_count} | **Spam Quarantined:** {spam_count} | **Urgent:** {urgent_count}",
        "",
        "---",
        "## 📝 Executive Overview",
        exec_summary,
        "",
        "## 🎯 Line-by-Line Important Takeaways (Har Mail Ka Key Message)",
    ]
    
    for idx, item in enumerate(line_by_line_highlights, 1):
        status_tag = "🚫 [SPAM]" if item["is_spam"] else ("📢 [PROMO]" if item["is_promo"] else "🟢 [CLEAN]")
        report_lines.append(f"{idx}. {status_tag} **{item['subject']}** (`{item['sender']}`)")
        report_lines.append(f"   - 💡 **Crucial Point:** {item['one_line_important']}")
        report_lines.append(f"   - ⚡ **Urgency:** {item['urgency']} | **Category:** {item['category']}")

    if urgent_emails:
        report_lines.extend([
            "",
            "## 🚨 Priority & Action Required Emails",
        ])
        for idx, u in enumerate(urgent_emails, 1):
            report_lines.append(f"{idx}. **{u['subject']}** (From: `{u['sender']}`) - *{u['urgency']}*")
            report_lines.append(f"   - 🎯 **Action:** {u['action']}")

    if spam_emails:
        report_lines.extend([
            "",
            "## 🚫 Quarantined Spam / Phishing Messages",
        ])
        for idx, sp in enumerate(spam_emails, 1):
            reasons = ", ".join(sp["analysis"]["spam_info"]["reasons"])
            report_lines.append(f"{idx}. **{sp['subject']}** (From: `{sp['sender']}`)")
            report_lines.append(f"   - ⚠️ **Spam Trigger:** {reasons} (Score: {sp['analysis']['spam_info']['spam_score']}/100)")

    markdown_report = "\n".join(report_lines)

    return {
        "total_emails": total,
        "clean_emails": clean_emails,
        "spam_emails": spam_emails,
        "promo_emails": promo_emails,
        "line_by_line_highlights": line_by_line_highlights,
        "executive_summary": exec_summary,
        "action_items": all_action_items,
        "category_counts": category_counts,
        "urgent_emails": urgent_emails,
        "upcoming_dates": all_dates,
        "top_senders": sender_counts,
        "markdown_report": markdown_report
    }


def generate_ai_reply(
    email_data: Dict[str, Any], 
    tone: str = "Professional", 
    custom_instructions: str = "",
    gemini_api_key: Optional[str] = None
) -> str:
    """Draft an AI-assisted reply based on the incoming email."""
    sender = email_data.get("sender", "there").split("<")[0].strip() or "there"
    subject = email_data.get("subject", "")
    body = email_data.get("body_plain", "") or email_data.get("body_html", "")

    greeting = f"Hi {sender}," if not sender.lower().startswith("no-reply") else "Hi,"
    notes_clause = f"\n\nRegarding your notes: {custom_instructions}" if custom_instructions else ""

    if tone == "Professional":
        return (
            f"{greeting}\n\n"
            f"Thank you for reaching out regarding '{subject}'.\n\n"
            f"I have reviewed the details and appreciate you sharing this update. "
            f"I will go through the points thoroughly and get back to you shortly with any necessary deliverables or answers.{notes_clause}\n\n"
            f"Best regards,\n[Your Name]"
        )
    elif tone == "Friendly & Casual":
        return (
            f"{greeting}\n\n"
            f"Thanks for the email! Hope you are having a great week.\n\n"
            f"Got your message regarding '{subject}'. Everything looks good on my end. "
            f"Let's sync up if anything else is needed.{notes_clause}\n\n"
            f"Cheers,\n[Your Name]"
        )
    elif tone == "Short & Direct":
        return (
            f"{greeting}\n\n"
            f"Acknowledging receipt of '{subject}'. Will review and update you soon.{notes_clause}\n\n"
            f"Thanks,\n[Your Name]"
        )
    elif tone == "Polite Decline":
        return (
            f"{greeting}\n\n"
            f"Thank you for reaching out regarding '{subject}'.\n\n"
            f"Unfortunately, I will not be able to accommodate this request at the present moment due to prior commitments.{notes_clause}\n\n"
            f"Thank you for your understanding.\n\nBest regards,\n[Your Name]"
        )
    else:
        return (
            f"{greeting}\n\n"
            f"Thank you for your email regarding '{subject}'. Could you please provide a few additional details so I can assist you better?{notes_clause}\n\n"
            f"Best regards,\n[Your Name]"
        )
