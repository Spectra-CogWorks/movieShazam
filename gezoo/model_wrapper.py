import numpy as np
from facenet_models import FacenetModel

_model = FacenetModel()

def feed_mtcnn(image, threshold=0.8):
  """Feeds the model an image get boxes for faces with certain proababilities

  Parameters
  ----------
  image: np.ndarray
    Image to feed to MTCNN

  threshold: float
    Probability at which include boxes

  Returns
  -------
  boxes: np.ndarray
    A shape (N, 4) array for each box over the threshold
  """
  # ! Please test to find a new threshold value
  global _model

  boxes, probabilities, _ = _model.detect(image)
  if boxes is None:
    return np.zeros(shape=(0,4),dtype=np.float32)

  boxes = boxes[[
    i for i in range(len(probabilities)) if probabilities[i] is not None and probabilities[i] > threshold
  ]]
  
  for box in boxes:
    box.reshape(4)

  return boxes

def compute_fingerprints(image, boxes):
  """Fingerprint faces in `boxes` in `image`

  Parameters
  ----------
  image: np.ndarray
    Image to feed to InverseNet

  boxes: np.ndarray
    Boxes in the image

  Returns
  -------
  fingeprints: np.ndarray
    Fingerprints in np.ndarray of shape (N, 512)
  """
  return _model.compute_descriptors(image, boxes)