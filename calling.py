# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = 'AC802d335077a61423009b8111a61a918e'
auth_token = 'c0ad5ca9dfeeea5b5562081d9411d32b'
client = Client(account_sid, auth_token)

call = client.calls.create(
                        url='http://demo.twilio.com/docs/voice.xml',
                        to='+919990339112',
                        from_='+14174532983'
                    )

print(call.sid)