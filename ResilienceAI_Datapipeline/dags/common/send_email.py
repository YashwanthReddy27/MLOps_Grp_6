import smtplib
from email.message import EmailMessage
import yaml
import logging

class AlertEmail:

    def __init__(self):
        with open("email_config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.logger = logging.getLogger(__name__)

    def send_email_with_attachment(
        self,
        recipient_email: str, 
        subject: str, 
        body: str, 
        attachments=None, 
        sender_email: str=self.config['sender_email'], 
        sender_password: str=self.config['sender_password'], 
        smtp_server: str=self.config['smtp_server'], 
        smtp_port: int=self.config['smtp_port']
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
            self.logger.info(" Email sent successfully!")
        except Exception as e:
            self.logger.error(f" Error sending email: {e}")