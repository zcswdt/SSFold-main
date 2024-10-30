import cv2
import numpy as np
import torch.nn.functional as F

#------------------------------------------------------------------------------

# Get the binary mask of the fabric
# 1 -> fabric
def get_cloth_mask(img):
  grey = np.mean(img, -1)
  # grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  mask = np.zeros(grey.shape)

  mask[grey > 0] = 1.0

  # kernel = np.ones((5,5), dtype='uint8')
  # mask = cv2.erode(mask, kernel)
  # debug=True
  # if debug:
  #   cv2.imshow("mask", mask)
  #   cv2.waitKey(0)

  return mask

#------------------------------------------------------------------------------

# Get all indices of mask
def get_indices_from_mask(mask):
  return np.transpose(np.nonzero(mask))

#------------------------------------------------------------------------------

# Pick a random pick from the possible indices
def choose_random_index(indices):
  return indices[np.random.randint(indices.shape[0])]

#------------------------------------------------------------------------------

# Convert the pick place indices to env actions
def convert_indices_to_action(pick, place):
  action = np.concatenate([pick,place])/W
  return action

#------------------------------------------------------------------------------

def gaussian_heatmap(center, image_size = 10, sig = 5):
  """
  It produces single gaussian at expected center
  :param center:  the mean position (X, Y) - where high value expected
  :param image_size: The total image size
  :param sig: The sigma value
  :return:
  """
  x_axis = np.linspace(0, image_size-1, image_size) - center[0]
  y_axis = np.linspace(0, image_size-1, image_size) - center[1]
  xx, yy = np.meshgrid(x_axis, y_axis)
  kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
  return kernel

#------------------------------------------------------------------------------

def translate(image, x, y):
  # define the translation matrix and perform the translation
  M = np.float32([[1, 0, x], [0, 1, y]])
  shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)

  # return the translated image
  return shifted

#------------------------------------------------------------------------------

def rotate(image, angle, scale=1.0):
  # grab the dimensions of the image
  (h, w) = image.shape[:2]

  # if the center is None, initialize it as the center of
  # the image
  center = (w // 2, h // 2)

  # perform the rotation
  M = cv2.getRotationMatrix2D(center, angle, scale)
  rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

  # return the rotated image
  return rotated

#------------------------------------------------------------------------------

class DepthNorm:
  def __init__(self, dataset):
    # Calculate min and max depth for dataset
    # Min and max of training dataset depth
    min_depth = np.inf
    max_depth = 0
    for episode in dataset:
      for i in range(len(episode)):
        depth = episode[i]['obs']['depth']
        print(np.min(depth))
        depth = depth[depth > 0]
        n_depth = episode[i]['nobs']['depth']
        n_depth = n_depth[n_depth > 0]
        min_depth = min(min_depth, np.min(depth), np.min(n_depth))
        print(min_depth)
        max_depth = max(max_depth, np.max(depth), np.max(n_depth))
        print(max_depth)
    self.min_depth = min_depth
    self.max_depth = max_depth

  def __call__(self, depth):
    depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
    depth = 1-depth
    return depth

