import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import traceback

load_dotenv()


class NotificationManager:
    def __init__(self):
        self.from_addr = os.getenv('EMAIL')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.email_list = email_addresses = os.getenv('EMAIL_LIST').split(',')

        self.email_template = """
            <!DOCTYPE html>
            <html>

            <head>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                        color: #333;
                        text-align: center;
                    }

                    .container {
                        width: 80%;
                        margin: auto;
                        background-color: #f9f9f9;
                        padding: 20px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }

                    .header {
                        background-color: #FF9843;
                        color: white;
                        padding: 10px;
                        text-align: center;
                    }

                    .section {
                        margin-top: 20px;
                        padding: 15px;
                        background-color: #ffffff;
                        border: 1px solid #ddd;
                    }

                    .section h1 {
                        color: #FF9843;
                    }

                    .details {
                        background-color: #FF9843;
                        color: white;
                        border-radius: 10px;
                        padding: 1em
                    }

                    .footer {
                        text-align: center;
                        padding: 10px;
                        background-color: #f1f1f1;
                        margin-top: 20px;
                    }
                </style>
            </head>"""

    def send_trade_alert(self, trade_update):
        msg = MIMEMultipart()
        msg['Subject'] = 'Trade Update'

        details = '\n'.join(f'{k}: {v}' for k,
                            v in trade_update['details'].items())

        email_body = f"""
            <body>
                <div class="container">
                    <!-- Trade Updates -->
                    <div class="section">
                        <h1>{trade_update['update_title']}</h1>
                        <div class="details">
                            <h3>{details}</h3>
                        </div>
                    </div>
                </div>
            </body>"""

        msg.attach(MIMEText(self.email_template + email_body, 'html'))

        self._send_email(msg)

    def send_error_alert(self, error_details):
        msg = MIMEMultipart()
        msg['Subject'] = 'Error Alert'

        email_body = f"""
            <body>
                <div class="container">
                    <div class="section">
                        <h1>{error_details['title']}</h1>
                        <div class="details">
                            <h3>{error_details['traceback']}</h3>
                        </div>
                    </div>
                </div>
            </body>"""

        msg.attach(MIMEText(self.email_template + email_body, 'html'))

        self._send_email(msg)

    # Daily portfolio update

    def send_daily_update(self, daily_update):
        msg = MIMEMultipart()
        msg['Subject'] = 'Daily Update'

        email_body = f"""
            <body>
                <div class="container">
                    <div class="section">
                        <h1>Daily Update</h1>
                        <div class="details">
                            <h3>Portfolio Value: {daily_update['portfolio_value']}</h3>
                            <h3>Positions: {daily_update['positions']}</h3>
                        </div>
                    </div>
                </div>
            </body>"""

        msg.attach(MIMEText(self.email_template + email_body, 'html'))

        self._send_email(msg)

    def _send_email(self, msg):
        msg['From'] = self.from_addr
        msg['To'] = ', '.join(self.email_list)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(self.from_addr, self.password)
        text = msg.as_string()
        server.sendmail(self.from_addr, self.email_list, text)
        server.quit()


def main():
    notification_manager = NotificationManager()
    notification_manager.send_trade_alert({'update_title': 'Order filled', 'details': {
                                          'symbol': 'AAPL', 'side': 'buy', 'quantity': 10}})
    try:
        raise Exception('Test')
    except Exception as e:
        notification_manager.send_error_alert(
            {'title': 'Error', 'traceback': traceback.format_exc()})
    notification_manager.send_daily_update(
        {'portfolio_value': 1000, 'positions': ['AAPL', 'TSLA']})


if __name__ == '__main__':
    main()
