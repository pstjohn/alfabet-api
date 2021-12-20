'''Create the json string to send to a tensorflow prediction model for bde
'''
# from alfabet.fragment import get_fragments, canonicalize_smiles
from alfabet.preprocessor import get_features
# import numpy as np
import json
import sys

# Example is CCO
smiles = sys.argv[1]
features = {key: val.tolist() for key, val in get_features(smiles).items()}
output = {"instances": [features]}
print(json.dumps(output))
cmd = "curl -d '%s' -X POST http://localhost:8501/v1/models/output_model:predict" % json.dumps(output)
print(cmd)


# c_smiles = canonicalize_smiles(smiles)
# fragments = get_fragments(smiles).to_dict(orient='records')
#{key: tf.constant(np.expand_dims(val, 0), name=val) for key, val in inputs.items()}
