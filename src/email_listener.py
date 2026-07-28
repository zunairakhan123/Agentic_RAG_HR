"""
Robust Background IMAP(Internet Message Access Protocol) Listener for incoming department email replies.
Includes MIME header decoding and dynamic Thread ID extraction.
"""
import os
import time
import imaplib
import email
from email.header import decode_header
import requests
import re
from dotenv import load_dotenv

load_dotenv()

EMAIL_ACCOUNT = os.getenv("SMTP_EMAIL")
EMAIL_PASSWORD = os.getenv("SMTP_PASSWORD")
WEBHOOK_URL = "http://localhost:8000/webhook/department-reply"

def decode_email_subject(raw_subject):
    """Decodes MIME encoded Base64 email subjects (e.g., =?UTF-8?B?...)."""
    if not raw_subject:
        return "No Subject"
    
    decoded_parts = []
    for content, encoding in decode_header(raw_subject):
        if isinstance(content, bytes):
            # Decode using the specified encoding, fallback to utf-8
            decoded_parts.append(content.decode(encoding or 'utf-8', errors='replace'))
        else:
            decoded_parts.append(content)
    return "".join(decoded_parts)

def extract_thread_id(subject: str):
    """Extracts whatever ID is inside the [Ref: ...] brackets, ensuring it is not empty."""
    if not subject: return None
    match = re.search(r"\[Ref:\s*([^\]]+)\]", subject)
    if match:
        tid = match.group(1).strip()
        # Ensure it actually captured a valid ID, not just blank spaces
        return tid if tid else None
    return None

def get_email_body(msg):
    """Extracts plain text body from the email payload with safe UTF-8 decoding."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                # [FIX]: Move the errors argument inside the string .decode() method
                payload_bytes = part.get_payload(decode=True)
                if payload_bytes:
                    return payload_bytes.decode('utf-8', errors='replace')
    else:
        payload_bytes = msg.get_payload(decode=True)
        if payload_bytes:
            return payload_bytes.decode('utf-8', errors='replace')
    
    return "No text body found."

def check_inbox():
    """Connects to Gmail IMAP and looks for new unread replies."""
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
        mail.select("inbox")

        status, response = mail.search(None, "UNSEEN")
        unread_msg_nums = response[0].split()

        if unread_msg_nums:
            print(f"[*] Found {len(unread_msg_nums)} unread email(s). Checking for Agent tags...")

            for num in unread_msg_nums:
                status, data = mail.fetch(num, "(RFC822)")
                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)
                
                # Retrieve and explicitly decode the subject
                raw_subject = msg.get("Subject", "")
                clean_subject = decode_email_subject(raw_subject)
                sender = msg.get("From", "Unknown Sender")
                
                print(f"    - Checking Unread Email Subject: {clean_subject}")
                
                thread_id = extract_thread_id(clean_subject)
                if thread_id:
                    print(f"    -> [MATCH!] Agent Reply Detected! Thread: {thread_id}")
                    body = get_email_body(msg)
                    
                    payload = {
                        "thread_id": thread_id,
                        "department": sender,
                        "reply_body": body.strip()
                    }
                    
                    try:
                        res = requests.post(WEBHOOK_URL, json=payload)
                        print(f"    [✓] Webhook triggered successfully. FastAPI Status: {res.status_code}")
                    except requests.exceptions.ConnectionError:
                        print("    [!] Webhook failed: Ensure FastAPI server is running.")
                else:
                    print("    -> [Ignored] Not an agent thread.")

        mail.logout()
    except Exception as e:
        print(f"[IMAP ERROR] {e}")

if __name__ == "__main__":
    print("[*] Starting NextBridge IMAP Listener...")
    print(f"[*] Monitoring {EMAIL_ACCOUNT} for department replies.")
    while True:
        check_inbox()
        time.sleep(10)