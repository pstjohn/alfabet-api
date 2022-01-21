from pathlib import Path

import tensorflow as tf
from alfabet.prediction import model
from alfabet.preprocessor import get_features, preprocessor

batched_input_spec = {
    key: tf.TensorSpec(shape=[None] + list(val.shape), dtype=val.dtype, name=key)
    for key, val in preprocessor.output_signature.items()
}


class Servable(tf.Module):
    def __init__(self, model, name=None):
        super().__init__(name)
        self.model = model
        self.embedding_model = tf.keras.Model(model.inputs, [model.layers[31].input])

    @tf.function
    def predict(self, inputs):
        return self.model(inputs)

    @tf.function
    def embedding(self, inputs):
        return self.embedding_model(inputs)


servable = Servable(model)
# predictions = servable.predict(inputs)
# embeddings = servable.embedding(inputs)

Path("models/output_model/1").mkdir(parents=True, exist_ok=True)

tf.saved_model.save(
    servable,
    "models/output_model/1",
    signatures={
        "serving_default": servable.predict.get_concrete_function(batched_input_spec),
        "predict": servable.predict.get_concrete_function(batched_input_spec),
        "embedding": servable.embedding.get_concrete_function(
            batched_input_spec, name="regress"
        ),
    },
)


del servable


dataset = tf.data.Dataset.from_generator(
    lambda: (get_features(smiles) for smiles in ["CC", "CCO"]),
    output_signature=preprocessor.output_signature,
).padded_batch(batch_size=2)

for inputs in dataset:
    break


servable_loaded = tf.saved_model.load("models/output_model/1")
servable_loaded.predict(inputs)
servable_loaded.embedding(inputs)
