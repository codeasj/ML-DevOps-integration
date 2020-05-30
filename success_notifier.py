#!/usr/bin/env python
# coding: utf-8

# In[12]:


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
sender="hostadd123@gmail.com"
passd="host_address"
guest="47guestadd12345@gmail.com"
subject="Notification of successful completion of model   "
content = '''Hi, 
				Developer this is to notify you about your last commit. The last commit that you had taken into consideration and based on that the trained model has given best accuracy .
				Congratulations on your success.
			THANK YOU ...'''
message = MIMEMultipart()
message['From']=sender
message['To']=guest
message['Subject']=subject
message.attach(MIMEText(content, 'plain'))
smtpserver=smtplib.SMTP('smtp.gmail.com', 587)
smtpserver.ehlo()
smtpserver.starttls()
smtpserver.ehlo()
smtpserver.login(sender,passd)
text = message.as_string()
smtpserver.sendmail(sender,guest,text)
smtpserver.close()
print('Successfully sent your mail')


# In[ ]:




