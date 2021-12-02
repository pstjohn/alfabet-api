from fastapi import FastAPI, HTTPException
from alfabet.fragment import canonicalize_smiles
from alfabet.neighbors import find_neighbor_bonds
from alfabet.prediction import predict_bdes, check_input


api = FastAPI()


def preprocess_smiles(smiles: str):
    try:
        can_smiles = canonicalize_smiles(smiles)
        if not can_smiles:
            raise HTTPException(status_code=400, detail="Missing smiles in url")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid smiles")
    # TODO: Catch exception here?
    is_outlier, missing_atom, missing_bond = check_input(can_smiles)
    if is_outlier:
        outlier_detail = {'status': 'outlier',
                          'missing atoms': missing_atom.tolist(),
                          'missing bond': missing_bond.tolist()}
        raise HTTPException(status_code=400, detail=outlier_detail)
    return can_smiles


# Preprocess smiles into vector format
@api.get("/preprocess/{smiles}")
def get_preprocess(smiles: str):
    return preprocess_smiles(smiles)
    # Not sure if this should be returning a string
    # return preprocess_smiles(smiles).to_dict(orient='records')


# Predict BDEs from smiles string
@api.get("/predict_bdes/{smiles}")
def get_predict_bdes(smiles: str):
    can_smiles = preprocess_smiles(smiles)
    bde_df = predict_bdes(can_smiles, draw=False)
    return bde_df.to_dict(orient='records')


# Get bond_index neighbor from smiles string
@api.get("/neighbors/{smiles}/{bond_index}")
def get_neighbors(smiles: str, bond_index: int):
    can_smiles = preprocess_smiles(smiles)
    neighbor_df = find_neighbor_bonds(
        can_smiles, bond_index, draw=False).drop(['rid', 'bdfe'], 1)
    return neighbor_df.to_dict(orient='records')
