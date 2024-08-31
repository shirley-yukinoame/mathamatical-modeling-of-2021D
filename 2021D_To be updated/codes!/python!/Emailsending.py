import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime

# 配置 QQ 邮箱的邮件服务器信息
SMTP_SERVER = 'smtp.qq.com'
SMTP_PORT = 465  # 如果使用 TLS，则为 587
EMAIL_USER = 'asfw67@qq.com'  # 替换为你的 QQ 邮箱地址
EMAIL_PASSWORD = 'wyx512467'  # 替换为你在 QQ 邮箱中获得的授权码

# 设置发件人、收件人及邮件主题
def send_email(subject, body, to_email, attachment=None):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = to_email
    msg['Subject'] = subject

    # 邮件内容
    msg.attach(MIMEText(body, 'plain'))

    # 附件（如果有的话）
    if attachment:
        with open(attachment, 'rb') as file:
            attach = MIMEApplication(file.read(), _subtype='pdf')
            attach.add_header('Content-Disposition', 'attachment', filename=attachment)
            msg.attach(attach)

    # 发送邮件
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, to_email, msg.as_string())
            print(f'邮件成功发送到 {to_email}')
    except Exception as e:
        print(f'发送邮件失败: {e}')

# 生成每日报告（这里简单地返回一个字符串）
def generate_daily_report():
    today = datetime.now().strftime('%Y-%m-%d')
    return f'这是 {today} 的每日报告'

# 主程序
if __name__ == "__main__":
    subject = '每日电子邮件报告'
    body = generate_daily_report()
    to_email = 'recipient_email@example.com'  # 替换为接收者的邮件地址
    attachment = None  # 如果需要添加附件，可以在这里指定附件路径，例如 'report.pdf'
    send_email(subject, body, to_email, attachment)
