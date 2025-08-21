import pandas as pd
import torch
from glob import glob
import numpy as np
import os
import boto3
import io
import logging
from botocore.exceptions import ClientError

def read_file(s3, file_name, bucket):
    """Upload a file to an S3 bucket
    :param s3: Initialized S3 client object
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :return: object if file was uploaded, else False
    """
    try:
        #with open(file_name, "rb") as f:
        obj = s3.get_object(Bucket=bucket, Key=file_name)
        #print(obj)
        myObject = obj['Body'].read()
    except ClientError as e:
        logging.error(e)
        return None
    return myObject



def Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY):
    # Initialize the S3 client
    s3 = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY
    )
    return s3


def get_list_objects(s3, S3_BUCKET_NAME):
    # List the objects in our bucket
    response = s3.list_objects(Bucket=S3_BUCKET_NAME)
    for item in response['Contents']:
        print(item['Key'])
    return response