from api import api, preprocess_smiles
from fastapi.exceptions import HTTPException
from fastapi.testclient import TestClient
import numpy
import os
import pandas

fastapi_client = TestClient(api)


def tests_preprocess():
    data = preprocess_smiles("CCO")
    assert data == "CCO"
    try:
        data = preprocess_smiles("B")
    except HTTPException as e:
        assert e.status_code == 400
        assert e.detail["status"] == 'outlier'
    try:
        data = preprocess_smiles("X")
    except HTTPException as e:
        assert e.status_code == 400
        assert e.detail == 'Invalid smiles'


def tests_api():
    data = fastapi_client.get('/preprocess/CCO').json()
    assert data == 'CCO'

    resp = fastapi_client.get('/preprocess/B')
    assert resp.status_code == 400
    data = resp.json()
    assert data['detail']['status'] == 'outlier'

    resp = fastapi_client.get('/preprocess/X')
    assert resp.status_code == 400
    data = resp.json()
    assert data['detail'] == 'Invalid smiles'


def tests_predict():
    desired = pandas.read_json(os.path.join(os.path.dirname(__file__), "CCO.json"))
    data = fastapi_client.get('/predict_bdes/CCO').text
    ret = pandas.read_json(data)
    pandas.testing.assert_frame_equal(desired, ret, check_exact=False)


def tests_neighbors():
    desired = pandas.read_json(os.path.join(os.path.dirname(__file__), "neighbors_CCO_5.json"))
    data = fastapi_client.get('/neighbors/CCO/5').text
    ret = pandas.read_json(data)
    pandas.testing.assert_frame_equal(desired, ret, check_exact=False)
