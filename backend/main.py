from fastapi import FastAPI, HTTPException
import boto3

client = boto3.client(
    "s3",
    # aws_access_key_id="3rk6M3VBuOIEVvyZ5R4A",
    # aws_secret_access_key="oJ6Sg1RXD0av593TRpSjyRm2ie6kxAbtpjUfvuMn",
    aws_access_key_id="eLsPxc1ntuTqavYVhnvc",
    aws_secret_access_key="0IgMaMG3U4AqhRUHZJXbUrSQ0UC26JVtobMpuM79",
    endpoint_url="http://192.168.0.134:9000"
)
 

app = FastAPI()

@app.get("/s3/{file_name}")
def get_file(file_name: str):
    response = client.generate_presigned_url(
        'get_object', Params={'Bucket': "machine-learning", 'Key': file_name})
    # if object exists, return url
    # else return 404
    try:
        client.head_object(Bucket="machine-learning", Key=file_name)
        return response
    except:
        return HTTPException(status_code=404, detail="File not found")


