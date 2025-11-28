---
title: "smtplib"
---
简单邮件传输协议SMTP (**S**imple **Mail** **T**ransfer **P**rotocol) 是一个相对简单的基于文本的协议。在其之上指定了一条消息的一个或多个接收者，然后消息文本会被传输。

| 服务商 | SMTP服务器 | 端口 | 加密方式 |
| :-----| :----: | :----: | :-:
| QQ邮箱 | smtp.qq.com | 587 | TLS |
| 腾讯企业邮箱 | smtp.exmail.qq.com | 465 | SSL |
| 163邮箱 | smtp.163.com | 25/465 | TLS/SSL | 
| Gmail | smtp.gmail.com | 587 | TLS |
| Outlook | smtp-mail.outlook.com |587 | TLS |
> - SSL: Secure Sockets Layer 套接字安全协议
> - TLS: Transport Layer Security 传输层安全协议，是目前广泛使用的 SSL 的升级版和替代品
			
	

```python
import smtplib
from email.mime.text import MIMEText

msg = MIMEText(
    _text,              # 邮件正文文本内容
    _subtype='plain',   # 文本子类型，默认为 'plain'（纯文本）
    _charset=None       # 字符编码，默认为 'us-ascii'
)

msg["Subject"] = ""     # 邮件标题
msg['From'] = ""        # 邮件发送任
msg['To'] = ""          # 邮件接收人
msg['Cc'] = ""          # 邮件抄送人
```

- tls连接，✅推荐
```python
try:
    # 创建SMTP连接, 输入smtp服务器及端口号
    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()                       # 启用TLS加密
        server.login(sender_email, password)    # <发件人邮箱, 授权码>
        server.send_message(
            msg,
            from_addr=None,                     # 默认使用msg["From"]
            to_addr=None                        # 默认使用msg["To"], msg["Cc"], msg["Bcc"] (收件人，抄送，密送)
        )
    print("邮件发送成功！")

except Exception as e:
    print(f"邮件发送失败: {e}")
```
- ssl连接，❌不推荐
```python
smtplib.SMTP_SSL
```