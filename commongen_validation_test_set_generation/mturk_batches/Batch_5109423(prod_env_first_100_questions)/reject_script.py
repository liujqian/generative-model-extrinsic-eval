import boto3

def reject(assignment_id:str):
    mturk = boto3.client('mturk',
                         endpoint_url='https://mturk-requester.us-east-1.amazonaws.com/',
                         # endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com/',
                         region_name='us-east-1')