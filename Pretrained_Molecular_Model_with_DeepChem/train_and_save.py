import deepchem as dc
from deepchem.molnet import load_delaney
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def save_pretrained(model, model_dir='saved_model', config=None):
    os.makedirs(model_dir, exist_ok=True)
    model.model.save(os.path.join(model_dir, 'keras_model.h5'))
    if config:
        with open(os.path.join(model_dir, 'config.txt'), 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

def build_model(n_features, n_tasks):
    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(1000, activation='relu')(inputs)
    outputs = layers.Dense(n_tasks, activation='linear')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    print("📦 Loading dataset...")
    tasks, datasets, transformers = load_delaney(featurizer='ECFP', splitter='random')
    train_dataset, valid_dataset, test_dataset = datasets
    n_tasks = len(tasks)
    n_features = train_dataset.get_data_shape()[0]

    print(f"🧮 Tasks: {n_tasks}, Features: {n_features}")
    print("🔧 Building model...")
    keras_model = build_model(n_features, n_tasks)

    print("🔁 Wrapping with DeepChem KerasModel...")
    model = dc.models.KerasModel(
        model=keras_model,
        loss=dc.models.losses.L2Loss(),
        output_types=["prediction"],
        learning_rate=0.001,
        batch_size=32
    )

    print("🚀 Training...")
    model.fit(train_dataset, nb_epoch=10)

    print("💾 Saving...")
    save_pretrained(model, model_dir='saved_model', config={
        "n_features": n_features,
        "n_tasks": n_tasks,
        "layer_sizes": [1000]
    })

    print("✅ Done! Model output shape:", keras_model.output_shape)

if __name__ == "__main__":
    main()