import smtplib
import os
import shutil
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText

def send_email():
    user = os.getenv('EMAIL_USER')
    password = os.getenv('EMAIL_PASS')
    recipients = ['jordifekou@gmail', 'zogobrice20@gmail.com']

    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = 'Modèle Entraîné et Documentation'

    body = 'Veuillez trouver ci-joint le modèle entraîné et la documentation générée.'
    msg.attach(MIMEText(body, 'plain'))

    # Zip the model directory
    model_dir = 'model'
    model_zip_filename = 'model.zip'
    shutil.make_archive('model', 'zip', model_dir)

    # Attach the model zip file
    with open(model_zip_filename, 'rb') as attachment:
        part = MIMEApplication(attachment.read(), Name=model_zip_filename)
        part['Content-Disposition'] = f'attachment; filename="{model_zip_filename}"'
        msg.attach(part)

    # Zip the doc directory
    doc_dir = 'doc'
    doc_zip_filename = 'doc.zip'
    shutil.make_archive('doc', 'zip', doc_dir)

    # Attach the doc zip file
    with open(doc_zip_filename, 'rb') as attachment:
        part = MIMEApplication(attachment.read(), Name=doc_zip_filename)
        part['Content-Disposition'] = f'attachment; filename="{doc_zip_filename}"'
        msg.attach(part)

    # Send email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(user, password)
            server.sendmail(user, recipients, msg.as_string())
    except smtplib.SMTPAuthenticationError as e:
        print(f"Authentication error: {e}")
    except Exception as e:
        print(f"Error: {e}")

    # Clean up the zip files
    os.remove(model_zip_filename)
    os.remove(doc_zip_filename)

if __name__ == '__main__':
    send_email()
