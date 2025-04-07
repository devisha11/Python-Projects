# Pretrained Molecular Model using DeepChem

This project demonstrates how to train a molecular property prediction model using DeepChem, save it, and load it later for making predictions.

It includes:
- Loading a molecular dataset
- Building and training a Keras-based regression model
- Saving the trained model and configuration
- Loading the model and making predictions on test data

---

## Project Files

- `train_and_save.py` – Trains the model and saves it in a predefined format.
- `load_and_predict.py` – Loads the saved model and performs predictions.
- `model_utils.py` – Contains helper functions for saving and loading models.
- `saved_model/` – Directory created after training, containing the saved model and config.

---

## How to Use

### 1. Install Required Packages

```bash
pip install deepchem tensorflow scikit-learn
