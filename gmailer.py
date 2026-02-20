import smtplib
import mimetypes

from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

# https://datamakessense.com/easy-scheduled-emailing-with-python-for-typical-bi-needs/


class GMailer:
    def __init__(self, subject, recipients, sender, senderpass):
        self.subject = subject
        self.recipients = recipients
        self.htmlbody = ""
        self.sender = sender
        self.senderpass = senderpass
        self.attachments = []

    def send(self):
        msg = MIMEMultipart("alternative")
        msg["From"] = self.sender
        msg["Subject"] = self.subject
        msg["To"] = ", ".join(
            self.recipients
        )  # to must be array of the form ['mailsender135@gmail.com']
        msg.preamble = "preamble goes here"
        # check if there are attachments if yes, add them
        if self.attachments:
            self.attach(msg)
        # add html body after attachments
        msg.attach(MIMEText(self.htmlbody, "html"))

        # send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(self.sender, self.senderpass)
            server.sendmail(self.sender, self.recipients, msg.as_string())

    def htmladd(self, html):
        self.htmlbody = self.htmlbody + "<p></p>" + html

    def attach(self, msg):
        for f in self.attachments:

            ctype, encoding = mimetypes.guess_type(f)
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"

            maintype, subtype = ctype.split("/", 1)

            if maintype == "text":
                fp = open(f)
                # Note: we should handle calculating the charset
                attachment = MIMEText(fp.read(), _subtype=subtype)
                fp.close()
            elif maintype == "image":
                fp = open(f, "rb")
                attachment = MIMEImage(fp.read(), _subtype=subtype)
                fp.close()
            elif maintype == "audio":
                fp = open(f, "rb")
                attachment = MIMEAudio(fp.read(), _subtype=subtype)
                fp.close()
            else:
                fp = open(f, "rb")
                attachment = MIMEBase(maintype, subtype)
                attachment.set_payload(fp.read())
                fp.close()
                encoders.encode_base64(attachment)
            attachment.add_header("Content-Disposition", "attachment", filename=f)
            attachment.add_header("Content-ID", "<{}>".format(f))
            msg.attach(attachment)

    def addattach(self, files):
        self.attachments = self.attachments + files
