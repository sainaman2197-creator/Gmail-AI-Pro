"""
Gmail Service Module
Handles direct Gmail connection using IMAP (SSL) for fetching emails 
and SMTP (STARTTLS) for sending replies, with robust MIME multipart 
parsing and attachment extraction. Zero database dependency.
"""

import imaplib
import smtplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import io
import re
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup

# Optional PDF parsing
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


def decode_mime_header(header_value: Optional[str]) -> str:
    """Safely decode RFC 2047 MIME encoded headers."""
    if not header_value:
        return ""
    
    decoded_parts = []
    try:
        parts = decode_header(header_value)
        for part, encoding in parts:
            if isinstance(part, bytes):
                if encoding:
                    try:
                        decoded_parts.append(part.decode(encoding, errors="replace"))
                    except Exception:
                        decoded_parts.append(part.decode("utf-8", errors="replace"))
                else:
                    decoded_parts.append(part.decode("utf-8", errors="replace"))
            else:
                decoded_parts.append(str(part))
        return "".join(decoded_parts)
    except Exception:
        return str(header_value)


def clean_html_to_text(html_content: str) -> str:
    """Convert HTML email body to readable clean plain text."""
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        # Remove scripts and styles
        for tag in soup(["script", "style", "meta", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join(line for line in lines if line)
        return clean_text
    except Exception:
        return html_content


def extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract selectable text from PDF bytes."""
    text = ""
    if HAS_PYPDF:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    pages.append(t.strip())
            text = "\n\n".join(pages).strip()
        except Exception:
            pass

    if not text and HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages = []
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t.strip():
                        pages.append(t.strip())
                text = "\n\n".join(pages).strip()
        except Exception:
            pass

    return text


def parse_email_message(msg_id: str, msg_bytes: bytes) -> Dict[str, Any]:
    """Parse a raw RFC822 email message into a structured dictionary."""
    msg = email.message_from_bytes(msg_bytes)

    subject = decode_mime_header(msg.get("Subject", "No Subject"))
    sender = decode_mime_header(msg.get("From", "Unknown Sender"))
    to = decode_mime_header(msg.get("To", ""))
    date_str = msg.get("Date", "")
    message_id = msg.get("Message-ID", "")
    
    # Try parsing date into friendly format
    try:
        parsed_date = email.utils.parsedate_to_datetime(date_str)
        date_formatted = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        date_formatted = date_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body_plain = ""
    body_html = ""
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            
            # Check if this part is an attachment
            filename = part.get_filename()
            if filename:
                filename = decode_mime_header(filename)
                file_bytes = part.get_payload(decode=True)
                if file_bytes:
                    att_info = {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "bytes": file_bytes,
                        "extracted_text": ""
                    }
                    if filename.lower().endswith(".pdf"):
                        att_info["extracted_text"] = extract_pdf_text_from_bytes(file_bytes)
                    elif filename.lower().endswith((".txt", ".csv", ".json", ".log")):
                        try:
                            att_info["extracted_text"] = file_bytes.decode("utf-8", errors="replace")[:10000]
                        except Exception:
                            pass
                    attachments.append(att_info)
            else:
                # Text or HTML content
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_plain += payload.decode(charset, errors="replace")
                elif content_type == "text/html" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_html += payload.decode(charset, errors="replace")
    else:
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            decoded_text = payload.decode(charset, errors="replace")
            if content_type == "text/html":
                body_html = decoded_text
            else:
                body_plain = decoded_text

    # If only HTML exists, create plain text from it
    if not body_plain and body_html:
        body_plain = clean_html_to_text(body_html)

    # Detect if message is forwarded
    sub_lower = subject.lower()
    body_lower = body_plain.lower()
    is_forwarded = (
        sub_lower.startswith(("fwd:", "fw:")) 
        or "---------- forwarded message ---------" in body_lower
        or "begin forwarded message" in body_lower
        or "forwarded message" in body_lower
    )

    # Detect importance
    is_important = (
        any(k in sub_lower or k in body_lower for k in [
            "urgent", "action required", "important", "interview", "selected",
            "deadline", "asap", "offer letter", "critical", "verification code", "otp"
        ])
    )

    return {
        "id": msg_id,
        "message_id": message_id,
        "sender": sender,
        "to": to,
        "subject": subject,
        "date": date_formatted,
        "body_plain": body_plain.strip(),
        "body_html": body_html.strip(),
        "is_forwarded": is_forwarded,
        "is_important": is_important,
        "attachments": attachments,
        "has_attachments": len(attachments) > 0,
        "unread": False
    }


class GmailClient:
    """Live Gmail Client using standard IMAP/SMTP over SSL."""

    def __init__(self, email_address: str, app_password: str):
        self.email_address = email_address.strip()
        # Clean spaces if user copied 16-char app password with spaces (e.g. abcd efgh ijkl mnop)
        self.app_password = app_password.strip().replace(" ", "")
        self.imap_host = "imap.gmail.com"
        self.imap_port = 993
        self.smtp_host = "smtp.gmail.com"
        self.smtp_port = 587

    def test_connection(self) -> Tuple[bool, str]:
        """Verify Gmail credentials via IMAP."""
        try:
            mail = imaplib.IMAP4_SSL(self.imap_host, self.imap_port)
            mail.login(self.email_address, self.app_password)
            mail.logout()
            return True, "Successfully connected to Gmail! 🟢"
        except imaplib.IMAP4.error as e:
            return False, f"Gmail Authentication failed: {str(e)}. Please check your Gmail address and 16-character Google App Password."
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    def get_available_folders(self) -> List[str]:
        """List accessible Gmail folders."""
        try:
            mail = imaplib.IMAP4_SSL(self.imap_host, self.imap_port)
            mail.login(self.email_address, self.app_password)
            status, folder_list = mail.list()
            mail.logout()
            folders = ["INBOX", "[Gmail]/Spam", "[Gmail]/Starred", "[Gmail]/Sent Mail", "[Gmail]/Trash"]
            return folders
        except Exception:
            return ["INBOX", "[Gmail]/Spam"]

    def fetch_emails(
        self, 
        folder: str = "INBOX", 
        limit: int = 25, 
        search_criteria: str = "ALL"
    ) -> List[Dict[str, Any]]:
        """Fetch emails directly from Gmail folder."""
        emails_list = []
        try:
            mail = imaplib.IMAP4_SSL(self.imap_host, self.imap_port)
            mail.login(self.email_address, self.app_password)
            
            # Select mailbox folder
            status, _ = mail.select(f'"{folder}"', readonly=True)
            if status != "OK":
                status, _ = mail.select("INBOX", readonly=True)
                
            # Search messages
            status, search_data = mail.search(None, search_criteria)
            if status != "OK" or not search_data[0]:
                mail.logout()
                return []

            msg_ids = search_data[0].split()
            # Fetch latest emails first
            latest_ids = msg_ids[-limit:][::-1]

            for mid in latest_ids:
                status, msg_data = mail.fetch(mid, "(RFC822 FLAGS)")
                if status == "OK" and msg_data:
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            raw_email = response_part[1]
                            parsed = parse_email_message(mid.decode("utf-8", errors="ignore"), raw_email)
                            
                            # Check flags for UNSEEN (Unread)
                            if len(response_part) > 0 and isinstance(response_part[0], bytes):
                                flags_str = response_part[0].decode("utf-8", errors="ignore")
                                if "\\Seen" not in flags_str:
                                    parsed["unread"] = True
                                    
                            emails_list.append(parsed)

            mail.logout()
        except Exception as e:
            raise RuntimeError(f"Error fetching emails from Gmail: {str(e)}")

        return emails_list

    def send_reply(
        self, 
        to_email: str, 
        subject: str, 
        body_text: str, 
        in_reply_to: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Send an email or reply via Gmail SMTP."""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_address
            msg["To"] = to_email
            
            if not subject.lower().startswith("re:") and in_reply_to:
                subject = f"Re: {subject}"
            msg["Subject"] = subject
            
            if in_reply_to:
                msg["In-Reply-To"] = in_reply_to
                msg["References"] = in_reply_to

            msg.attach(MIMEText(body_text, "plain"))

            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.ehlo()
            server.starttls()
            server.login(self.email_address, self.app_password)
            server.send_message(msg)
            server.quit()

            return True, "Email reply sent successfully via Gmail SMTP! 🚀"
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"


def get_mock_emails() -> List[Dict[str, Any]]:
    """Generate realistic sample emails (including clean, urgent, spam & promo) for preview."""
    return [
        {
            "id": "mock_1",
            "message_id": "<mock1@google.com>",
            "sender": "Google Careers <careers-noreply@google.com>",
            "to": "you@domain.com",
            "subject": "Interview Invitation: AI & Software Engineering Role",
            "date": datetime.now().strftime("%Y-%m-%d 10:30:00"),
            "body_plain": (
                "Hi there,\n\nWe were very impressed with your background and would like to invite you for a technical interview "
                "for the AI Software Engineer position. Please let us know your availability for this coming Friday, 28th September between 2:00 PM and 5:00 PM IST.\n\n"
                "The round will cover Python backend architecture, AI agents, and system design.\n\n"
                "Looking forward to hearing from you soon.\n\nBest regards,\nGoogle Talent Acquisition Team"
            ),
            "body_html": "<p>Hi there,</p><p>We were very impressed with your background and would like to invite you for a <b>technical interview</b> for the <b>AI Software Engineer</b> position.</p><p>Please let us know your availability for this coming Friday between 2:00 PM and 5:00 PM IST.</p>",
            "is_forwarded": False,
            "is_important": True,
            "has_attachments": False,
            "attachments": [],
            "unread": True
        },
        {
            "id": "mock_2",
            "message_id": "<mock2@aws.amazon.com>",
            "sender": "AWS Billing <no-reply-aws@amazon.com>",
            "to": "you@domain.com",
            "subject": "Urgent: Monthly AWS Cloud Services Invoice #INV-98231 Available",
            "date": datetime.now().strftime("%Y-%m-%d 08:15:00"),
            "body_plain": (
                "Dear Customer,\n\nYour monthly statement for AWS Cloud services covering EC2 instances, S3 storage, and Bedrock AI endpoints is now ready.\n"
                "Total Amount Due: $42.50 USD.\nPayment Due Date: October 05, 2026.\n\n"
                "Automatic deduction will be processed from your default credit card. Please review the invoice in your AWS Management Console.\n\n"
                "Thank you for choosing Amazon Web Services."
            ),
            "body_html": "<p>Dear Customer,</p><p>Your monthly statement for <b>AWS Cloud services</b> is now ready.</p><p><b>Total Amount Due:</b> $42.50 USD<br><b>Payment Due Date:</b> October 05, 2026</p>",
            "is_forwarded": False,
            "is_important": True,
            "has_attachments": True,
            "attachments": [{
                "filename": "AWS_Invoice_INV98231.pdf",
                "content_type": "application/pdf",
                "size": 14200,
                "bytes": b"%PDF-mock-bytes",
                "extracted_text": "AWS Cloud Services Invoice. Total Due: $42.50. Payment due by Oct 05, 2026."
            }],
            "unread": True
        },
        {
            "id": "mock_3",
            "message_id": "<mock3@project-team.internal>",
            "sender": "Rahul Sharma (Team Lead) <rahul.sharma@techcorp.io>",
            "to": "you@domain.com",
            "subject": "Fwd: Project Sprint 4 Goals & Architecture Sync",
            "date": datetime.now().strftime("%Y-%m-%d 07:45:00"),
            "body_plain": (
                "---------- Forwarded message ---------\n"
                "From: VP Engineering <vp@techcorp.io>\n"
                "Subject: Sprint 4 Deliverables\n\n"
                "Hi Team,\n\nHere are the core deliverables for Sprint 4:\n"
                "1. Connect Gmail IMAP/SMTP client directly without SQL database.\n"
                "2. Implement Inbox Master AI Digest to summarize all incoming emails at once.\n"
                "3. Enable 1-click AI Draft replies for quick productivity.\n\n"
                "Please push your pull requests before Thursday 6 PM for code review.\n\nThanks,\nRahul"
            ),
            "body_html": "<p><b>---------- Forwarded message ---------</b></p><p>Hi Team,</p><p>Here are the core deliverables for Sprint 4:</p><ul><li>Connect Gmail IMAP/SMTP directly</li><li>Implement Inbox Master AI Digest</li><li>Enable 1-click AI Draft replies</li></ul>",
            "is_forwarded": True,
            "is_important": False,
            "has_attachments": False,
            "attachments": [],
            "unread": False
        },
        {
            "id": "mock_4",
            "message_id": "<mock4@spam-winner.xyz>",
            "sender": "International Lottery Claims <winner@cashnow-lottery.xyz>",
            "to": "you@domain.com",
            "subject": "CONGRATULATIONS: You Won $5,000,000 in International Crypto Lottery!!!",
            "date": datetime.now().strftime("%Y-%m-%d 06:30:00"),
            "body_plain": (
                "DEAR BENEFICIARY,\n\nYOU HAVE BEEN SELECTED AS THE LUCKY WINNER OF $5,000,000 USD IN OUR BITCOIN TRANSFER SWEEPSTAKES.\n"
                "TO CLAIM YOUR PRIZE IMMEDIATELY, SEND YOUR BANK ACCOUNT DETAILS AND WIRE TRANSFER FEE OF $100 TO PROCEED.\n\n"
                "DO NOT DELAY, CLAIM NOW BEFORE DEADLINE EXPIRES!"
            ),
            "body_html": "<p><b>CONGRATULATIONS!</b> You won $5,000,000 lottery crypto prize. Send wire transfer fee to claim immediately!</p>",
            "is_forwarded": False,
            "is_important": False,
            "has_attachments": False,
            "attachments": [],
            "unread": True
        },
        {
            "id": "mock_5",
            "message_id": "<mock5@marketing-cloud.com>",
            "sender": "Cloud Promo Hub <promotions@clouddeals-hub.com>",
            "to": "you@domain.com",
            "subject": "🔥 Flash Sale: 50% OFF All Cloud Subscriptions - Limited Time Offer!",
            "date": datetime.now().strftime("%Y-%m-%d 05:40:00"),
            "body_plain": (
                "Exclusive Deal Just For You!\n\nUpgrade your server infrastructure today and get 50% off your annual billing.\n"
                "Use coupon code CLOUD50 at checkout. Click here to view in browser or buy now.\n\n"
                "To stop receiving these emails, click here to unsubscribe."
            ),
            "body_html": "<p><b>50% OFF Flash Sale</b> on cloud subscriptions. Use coupon CLOUD50. Click unsubscribe to opt-out.</p>",
            "is_forwarded": False,
            "is_important": False,
            "has_attachments": False,
            "attachments": [],
            "unread": False
        },
        {
            "id": "mock_6",
            "message_id": "<mock6@github.com>",
            "sender": "GitHub Security <security-alerts@github.com>",
            "to": "you@domain.com",
            "subject": "Security Notice: New SSH key added to your account",
            "date": datetime.now().strftime("%Y-%m-%d 04:15:00"),
            "body_plain": (
                "Hi user,\n\nA new SSH public key (id_ed25519_antigravity) was added to your GitHub account from IP 192.168.1.1.\n"
                "If you performed this action, no further steps are needed.\n"
                "If you did not add this key, please visit your account security settings immediately to revoke it.\n\nGitHub Security Team"
            ),
            "body_html": "<p>Hi user,</p><p>A new SSH public key was added to your GitHub account.</p><p>If this was not you, please revoke it immediately.</p>",
            "is_forwarded": False,
            "is_important": True,
            "has_attachments": False,
            "attachments": [],
            "unread": False
        }
    ]
