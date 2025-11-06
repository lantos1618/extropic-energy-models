#!/usr/bin/env python3
"""Send analysis email via SendGrid"""

import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_analysis():
    # Read analysis
    with open('/home/ubuntu/extropic_mandlebrot/analysis.md', 'r') as f:
        analysis = f.read()

    message = Mail(
        from_email='agent@lambda.run',
        to_emails='l.leong1618@gmail.com',  # TODO: Get real email
        subject='Energy-Based Mandelbrot: Mathematical Feasibility Analysis',
        html_content=f'<pre>{analysis}</pre>')

    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(f"Email sent! Status code: {response.status_code}")
        return response.status_code
    except Exception as e:
        print(f"Error sending email: {e}")
        return None

if __name__ == '__main__':
    send_analysis()
