import smtplib
from email.mime.text import MIMEText
from email.header import Header


def send_message(email, code):
    """
    Sends email message to confirm email during password recovery
    @param email: user's email to confirm
    @param code: generated four-number code
    """
    from_address = "speaker-trainer@yandex.ru"
    pwd = "qvnasubjnjtqgnnc"
    msg = MIMEText(f"Код для восстановления пароля: {code}.")
    msg["From"] = from_address
    msg["To"] = email
    msg["Subject"] = Header("Восстановление пароля")
    post = smtplib.SMTP_SSL("smtp.yandex.ru")
    post.login(from_address, pwd)
    post.send_message(msg)
    post.quit()
