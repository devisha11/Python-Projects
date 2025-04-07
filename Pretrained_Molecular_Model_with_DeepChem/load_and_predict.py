import deepchem as dc
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1. Reload the same dataset & featurizer used in training
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP', splitter='random')
train_dataset, valid_dataset, test_dataset = datasets

# 2. Rebuild the same model architecture
n_features = train_dataset.X.shape[1]
model_arch = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(1000, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(1)
])

# 3. Wrap in KerasModel and point to saved_model directory
model = dc.models.KerasModel(
    model=model_arch,
    loss=dc.models.losses.L2Loss(),
    model_dir='saved_model'
)

# 4. Restore weights from disk
model.restore()

# 5. Predict on test set
predictions = model.predict(test_dataset)
print("✅ Predictions (first 5):", predictions[:5])

# 6. Evaluate model performance
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
scores = model.evaluate(test_dataset, [metric], transformers)
print("📈 Test set performance:", scores)