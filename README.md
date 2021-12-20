## Alfabet API
### Docker-compose Instructions

* Build: `docker-compose build alfabet-api`
* Create .env file.  Must contain following variables:
```
AWS_ACCESS_KEY_ID - Access key id for aws account that has read access to s3 bucket
AWS_SECRET_ACCESS_KEY - Access key secret
AWS_REGION=us-west-2
S3_ENDPOINT=s3.us-west-2.amazonaws.com
MODEL_BASE_PATH='s3://alfabet-models/v0.1.1'
MODEL_NAME=output_model
```

* Run: `docker-compose up`
* API: http://localhost:8000/docs


### Docker Instructions

* Build: `docker build -t bde-api .`
* Run: `docker run --rm -p 8000:8000 -v root:/root bde-api:latest`
The root volume is created so the models don't have to be downloaded every startup.
* Test (assumes only one running docker): `docker exec -it $(docker ps -q) pytest`
* API: http://localhost:8000/docs
