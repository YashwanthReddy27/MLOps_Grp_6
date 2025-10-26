import smtplib
from email.message import EmailMessage
import os

def send_email_with_attachment(
    sender_email, 
    sender_password, 
    recipient_email, 
    subject, 
    body, 
    attachments=None, 
    smtp_server="smtp.gmail.com", 
    smtp_port=587
):
    # Create the email
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach files if any
    if attachments:
        for file_path in attachments:
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue
            with open(file_path, "rb") as f:
                file_data = f.read()
                file_name = os.path.basename(file_path)
            msg.add_attachment(
                file_data,
                maintype="application",
                subtype="octet-stream",
                filename=file_name,
            )

    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.starttls()  # Secure connection
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Error sending email: {e}")

# Example usage
if __name__ == "__main__":
    send_email_with_attachment(
        sender_email="projectmlops@gmail.com",
        sender_password="axhh ojnp axum udnj",  # Use an app password, not your normal one
        recipient_email="anirudhshrikanth65@gmail.com",
        subject="Test Email with Attachment",
        body="Hi there,\n\nThis is a test email with an attachment.\n\nBest,\nAnirudh",
        attachments=["expanded_fitness_data_1.csv"],
    )