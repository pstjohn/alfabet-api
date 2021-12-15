import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from alfabet.drawing import draw_bde, draw_mol, draw_mol_outlier
from alfabet.fragment import canonicalize_smiles, get_fragments
from alfabet.neighbors import get_neighbors
from alfabet.prediction import (
    format_predictions_into_dataframe,
    tf_model_forward,
    validate_inputs,
)
from alfabet.preprocessor import get_features
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

api = FastAPI()
api.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
logger = logging.getLogger(__name__)


class Bond(BaseModel):
    molecule: str
    bond_index: int
    bond_type: str
    fragment1: str
    fragment2: str
    is_valid_stereo: bool


class Features(BaseModel):
    atom: List[int]
    bond_indices: List[int]
    bond: List[int]
    connectivity: List[List[int]]
    is_valid: Optional[bool]


class BondPrediction(Bond):
    bde_pred: float
    bdfe_pred: float
    bde: Optional[float]
    bdfe: Optional[float]
    set: Optional[str]
    has_dft_bde: bool


class Neighbor(BaseModel):
    molecule: str
    bond_index: int
    fragment1: str
    fragment2: str
    bde: float
    bdfe: float
    set: str
    distance: float


class Outlier(BaseModel):
    SMILES: str
    missing_atom: List[int]
    missing_bond: List[int]
    is_valid: bool


class Message(BaseModel):
    message: str


def pandas_to_records(df: pd.DataFrame) -> List[Dict]:
    """Convert a pandas dataframe to a list of dicts in a records format,
    dropping entries that are na (i.e., optional entries)"""
    return [row.dropna().to_dict() for _, row in df.iterrows()]


@api.get("/canonicalize/{smiles}", responses={400: {"model": Message}})
async def canonicalize(smiles: str):
    try:
        return canonicalize_smiles(smiles)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid smiles")


@api.get("/fragment/{smiles}", response_model=List[Bond])
async def fragment(smiles: str = Depends(canonicalize)):
    return get_fragments(smiles).to_dict(orient="records")


@api.get("/featurize/{smiles}", response_model=Features)
async def featurize(smiles: str = Depends(canonicalize), pad: bool = True):
    return {key: val.tolist() for key, val in get_features(smiles, pad).items()}


@api.get(
    "/validate/{smiles}", response_model=Features, responses={400: {"model": Outlier}}
)
async def validate(smiles: str, features: Features = Depends(featurize)):
    is_outlier, missing_atom, missing_bond = validate_inputs(dict(features))
    if is_outlier:
        raise HTTPException(
            status_code=400,
            detail={
                "is_valid": False,
                "SMILES": smiles,
                "missing_atom": missing_atom.tolist(),
                "missing_bond": missing_bond.tolist(),
            },
        )
    else:
        features_out = dict(features)
        features_out["is_valid"] = True
        return features_out


def predict_bdes(
    features: Features = Depends(validate),
) -> Tuple[List[float], List[float]]:
    """This function should get replaced with the tf-serving CURL. Ideally this would
    also get called as a coroutine, since we'd get a lot of speedup by not blocking for
    each TFServing call

    Args:
        features (Features, optional): the input dictionary to the TF model

    Returns:
        Tuple[List[float], List[float]]: Predicted BDEs and BDFEs as list of floats
    """
    return tf_model_forward(features)


@api.get("/predict/{smiles}/{bond_index}", response_model=BondPrediction)
@api.get("/predict/{smiles}", response_model=List[BondPrediction])
async def predict(
    fragments: List[Bond] = Depends(fragment),
    features: Features = Depends(validate),
    drop_duplicates: bool = False,
    bond_index: Optional[int] = None,
):

    features = dict(features)
    features.pop("is_valid")
    bde, bdfe = predict_bdes(features)

    fragments = pd.DataFrame.from_records(fragments)
    bde_pred = format_predictions_into_dataframe(
        bde, bdfe, fragments, drop_duplicates=drop_duplicates
    )

    if bond_index is not None:
        solo_bde = bde_pred[bde_pred["bond_index"] == bond_index]
        return pandas_to_records(solo_bde)[0]
    return pandas_to_records(bde_pred)


@api.get(
    "/draw/{smiles}/{bond_index}",
    response_class=Response,
    responses={200: {"content": {"image/svg+xml": {}}}, 400: {"model": Message}},
)
@api.get(
    "/draw/{smiles}",
    response_class=Response,
    responses={200: {"content": {"image/svg+xml": {}}}},
)
async def draw(
    smiles: str = Depends(canonicalize),
    features: Features = Depends(featurize),
    bond_index: Optional[int] = None,
):
    if bond_index is not None:
        try:
            svg = draw_bde(smiles, bond_index)
        except RuntimeError as ex:
            raise HTTPException(status_code=400, detail=str(ex))
    else:
        is_outlier, missing_atom, missing_bond = validate_inputs(dict(features))
        if not is_outlier:
            svg = draw_mol(smiles)
        else:
            svg = draw_mol_outlier(smiles, missing_atom, missing_bond)
    return Response(content=svg, media_type="image/svg+xml")


@api.get(
    "/neighbors/{smiles}/{bond_index}",
    response_model=List[Neighbor],
    responses={400: {"model": Message}},
)
async def neighbors(bond_index: int, features: Features = Depends(validate)):
    features = dict(features)
    features.pop("is_valid")
    if bond_index in features["bond_indices"]:
        return pandas_to_records(get_neighbors(features, bond_index))
    else:
        raise HTTPException(status_code=400, detail="Invalid bond index")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000)
