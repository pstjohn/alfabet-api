## BDE API

### Docker Instructions

* Build: `docker build -t bde-api .`
* Run: `docker run --rm -p 8000:8000 -v root:/root bde-api:latest`
The root volume is created so the models don't have to be downloaded every startup.
* Test (assumes only one running docker): `docker exec -it $(docker ps -q) pytest`
* API: http://localhost:8000/docs
