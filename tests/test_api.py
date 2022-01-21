import os

import numpy as np
import pandas
import pytest
import rdkit.Chem
from api import api
from fastapi.testclient import TestClient

fastapi_client = TestClient(api)


def test_canonicalize():
    response = fastapi_client.get("/canonicalize/C1=CC=CC=C1")
    assert response.status_code == 200
    assert response.json() == "c1ccccc1"

    response = fastapi_client.get("/canonicalize/C1")
    assert response.status_code == 400

    response = fastapi_client.get("/canonicalize/")
    assert response.status_code == 404


def test_fragment():
    response = fastapi_client.get("/fragment/C1=CC=CC=C1")
    assert response.status_code == 200
    ret = pandas.read_json(response.text)
    desired = pandas.read_json(
        os.path.join(os.path.dirname(__file__), "benzene_frags.json")
    )
    pandas.testing.assert_frame_equal(desired, ret, check_exact=False)


def test_featurize():
    response = fastapi_client.get("/featurize/C1=CC=CC=C1")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "atom",
        "bond",
        "bond_indices",
        "connectivity",
        "is_valid",
    }


def test_validate():
    response = fastapi_client.get("/validate/CB")
    assert response.status_code == 400
    data = response.json()
    assert not data["detail"]["is_valid"]

    response = fastapi_client.get("/validate/C1=CC=CC=C1")
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"]


@pytest.mark.skip()
def test_predict():
    response = fastapi_client.get("/predict/C1=CC=CC=C1?drop_duplicates=True")
    assert response.status_code == 200
    ret = pandas.read_json(response.text)
    assert np.isclose(ret.bde, 110.9, rtol=1e-3)
    assert np.isclose(ret.bde_pred, 110.9, rtol=1e-2)
    assert len(ret) == 1

    response = fastapi_client.get("/predict/C1=CC=CC=C1?drop_duplicates=False")
    assert response.status_code == 200
    ret = pandas.read_json(response.text)
    assert len(ret) == 6

    # we don't have DFT values for all bonds in this molecule
    response = fastapi_client.get("/predict/CC(C)CC1CCCCC1?drop_duplicates=True")
    assert response.status_code == 200
    ret = pandas.read_json(response.text)
    assert len(ret) == 10
    assert len(ret.dropna(subset=["bde"])) == 7


def test_draw():
    response = fastapi_client.get("/draw/C1=CC=CC=C1/6")
    assert response.status_code == 200

    response = fastapi_client.get("/draw/CC/10")
    assert response.status_code == 400


def test_neighbors():
    response = fastapi_client.get("/neighbors/CC/0")
    assert response.status_code == 200
    neighbor_df = pandas.read_json(response.text)
    for _, row in neighbor_df.iterrows():
        mol = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(row.molecule))
        bond = mol.GetBondWithIdx(row.bond_index)
        assert bond.GetEndAtom().GetSymbol() == "C"
        assert bond.GetBeginAtom().GetSymbol() == "C"
        assert bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE

    response = fastapi_client.get("/neighbors/CC/10")
    assert response.status_code == 400
