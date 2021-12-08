from typing import List, Optional
import logging
import pandas as pd
from alfabet.drawing import draw_bde, draw_mol_outlier, draw_mol
from alfabet.fragment import get_fragments, canonicalize_smiles
from alfabet.neighbors import get_neighbors
from alfabet.prediction import predict_bdes, validate_inputs
from alfabet.preprocessor import get_features
from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

api = FastAPI()
api.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])
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
    set: str
    svg: Optional[str]
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


@api.get("/canonicalize/{smiles}", responses={400: {"model": Message}})
async def canonicalize(smiles: str):
    try:
        return canonicalize_smiles(smiles)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid smiles")


@api.get("/fragment/{smiles}", response_model=List[Bond])
async def fragment(smiles: str = Depends(canonicalize)):
    return get_fragments(smiles).to_dict(orient='records')


@api.get("/featurize/{smiles}", response_model=Features)
async def featurize(smiles: str = Depends(canonicalize)):
    return {key: val.tolist() for key, val in get_features(smiles).items()}


@api.get("/validate/{smiles}", response_model=Features,
         responses={400: {"model": Outlier}})
async def validate(
        smiles: str,
        features: Features = Depends(featurize)):
    is_outlier, missing_atom, missing_bond = validate_inputs(dict(features))
    if is_outlier:
        raise HTTPException(status_code=400,
                            detail={
                                "is_valid": False,
                                "SMILES": smiles,
                                "missing_atom": missing_atom.tolist(),
                                "missing_bond": missing_bond.tolist(),
                            })
    else:
        features_out = dict(features)
        features_out['is_valid'] = True
        return features_out


@api.get("/predict/{smiles}/{bond_index}", response_model=BondPrediction)
@api.get("/predict/{smiles}", response_model=List[BondPrediction])
async def predict(fragments: List[Bond] = Depends(fragment),
                  features: Features = Depends(validate),
                  drop_duplicates: bool = False,
                  bond_index: Optional[int] = None):
    features = dict(features)
    features.pop('is_valid')
    fragments = pd.DataFrame.from_records(fragments)
    bde_pred = predict_bdes(fragments, features, drop_duplicates)
    if bond_index is not None:
        solo_bde = bde_pred[bde_pred['bond_index'] == bond_index]
        return solo_bde.to_dict(orient='records')[0]
    return bde_pred.to_dict(orient='records')


@api.get("/draw/{smiles}/{bond_index}", response_class=Response, responses={200: {"content": {"image/svg+xml": {}}}})
@api.get("/draw/{smiles}", response_class=Response, responses={200: {"content": {"image/svg+xml": {}}}})
async def draw(smiles: str = Depends(canonicalize),
               features: Features = Depends(featurize),
               bond_index: Optional[int] = None):
    if bond_index is not None:
        svg = draw_bde(smiles, bond_index)
    else:
        is_outlier, missing_atom, missing_bond = validate_inputs(dict(features))
        if not is_outlier:
            svg = draw_mol(smiles)
        else:
            svg = draw_mol_outlier(smiles, missing_atom, missing_bond)
    return Response(content=svg, media_type="image/svg+xml")


@api.get("/neighbors/{smiles}/{bond_index}", response_model=List[Neighbor])
async def neighbors(bond_index: int, features: Features = Depends(validate)):
    features = dict(features)
    features.pop('is_valid')
    return get_neighbors(features, bond_index).to_dict(orient='records')


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000)
