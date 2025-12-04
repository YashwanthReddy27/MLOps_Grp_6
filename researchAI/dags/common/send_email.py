import smtplib
from email.message import EmailMessage
import yaml
import logging
import os

class AlertEmail:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, "config", "email_config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            self.logger.info("Email configuration loaded successfully.")

    def send_email_with_attachment(
        self,
        recipient_email: str, 
        subject: str, 
        body: str, 
        attachments=None,
        smtp_server="smtp.gmail.com"
    ):
        # Create the email
        msg = EmailMessage()
        msg["From"] = self.config['sender_email']
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
            with smtplib.SMTP(smtp_server, self.config['smtp_port']) as smtp:
                smtp.starttls()  # Secure connection
                smtp.login(self.config['sender_email'], self.config['sender_password'])
                smtp.send_message(msg)
            self.logger.info(" Email sent successfully!")
        except Exception as e:
            self.logger.error(f" Error sending email: {e}")

    def get_logger(self):
        return self.logger
