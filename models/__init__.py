"""Load given module for training and prediction."""
import importlib

def build_model(model_name, weights_file=None, **kwargs):
  """Build the desired model."""
  mod = importlib.import_module("."+model_name, __name__)
  model = mod.build_model(**kwargs)
  if weights_file:
    try:
      model.load_weights(weights_file, by_name=True)
      print("Loaded existing model:", weights_file)
    except Exception as e: # pylint: disable=broad-except
      print("Error loading model:", e)
  return model
