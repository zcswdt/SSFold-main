import cv2
import numpy as np
from .model_utils import *
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def normalize_pickmap(pick_map):
  image_width = pick_map.shape[0]
  pick_map = np.expand_dims(pick_map, 2)
  pick_map[pick_map!=0] = cv2.normalize(pick_map[pick_map!=0], pick_map[pick_map!=0], 0, 1.0, cv2.NORM_MINMAX)
  pick_map = (plt.get_cmap('viridis')(pick_map[:,:,0]))[:,:,:3]
  pick_map = pick_map[:,:,::-1]
  pick_map = cv2.resize(pick_map, (image_width,image_width), interpolation=cv2.INTER_NEAREST)
  return pick_map

def normalize_placemap(place_map):
  image_width = place_map.shape[0]
  place_map = np.expand_dims(place_map, 2)
  place_map = cv2.normalize(place_map, place_map, 0, 1.0, cv2.NORM_MINMAX)
  place_map = (plt.get_cmap('viridis')(place_map[:,:,0]))[:,:,:3]
  place_map = place_map[:,:,::-1]
  place_map = cv2.resize(place_map, (image_width, image_width), interpolation=cv2.INTER_NEAREST)
  return place_map

#------------------------------------------------------------------------------

def viz_images(imgs, name="viz"):
  img = np.vstack(imgs)

  cv2.namedWindow(name, cv2.WINDOW_NORMAL)
  cv2.imshow(name, img)
  cv2.waitKey(1)
  return img

#------------------------------------------------------------------------------
#
# def get_viz_img(pick, place, pred_pick, pred_place, obs_rgb, goal_rgb, pick_map, place_map):
#   pick_map = normalize_pickmap(pick_map)
#   place_map = normalize_placemap(place_map)
#
#   pick_map_obs = 0.5*obs_rgb + 0.5*pick_map
#   place_map_goal = 0.5*goal_rgb + 0.5*place_map
#   true_action_img = 0.5*obs_rgb + 0.5*goal_rgb
#   pred_action_img = 0.5*obs_rgb + 0.5*goal_rgb
#
#   pick = int(pick[0]), int(pick[1])
#   place = int(place[0]), int(place[1])
#   pred_pick = int(pred_pick[0]), int(pred_pick[1])
#   pred_place = int(pred_place[0]), int(pred_place[1])
#
#   cv2.arrowedLine(true_action_img, tuple(pick), tuple(place), (1, 0, 0.0), 1)
#   cv2.circle(true_action_img, tuple(pick), 4, (0.0,1.0,1.0), 1)
#   cv2.arrowedLine(pred_action_img, tuple(pred_pick), tuple(pred_place), (0.5,0.0,1.0), 1)
#   cv2.circle(pred_action_img, tuple(pred_pick), 4, (1.0, 0.0, 0.0), 1)
#
#   #img = np.hstack([obs_rgb,goal_rgb,pick_map_obs,place_map_goal, true_action_img, pred_action_img])
#   img = np.hstack([obs_rgb, goal_rgb, pick_map_obs, place_map_goal, true_action_img, pred_action_img])
#   return img

#------------------------------------------------------------------------------

def get_viz_img(pick, place, pred_pick, pred_place, obs_rgb, goal_rgb, pick_map, place_map):
  pick_map = normalize_pickmap(pick_map)
  place_map = normalize_placemap(place_map)

  pick_map_obs = 0.5 * obs_rgb + 0.5 * pick_map
  place_map_goal = 0.5 * goal_rgb + 0.5 * place_map
  true_action_img = 0.5 * obs_rgb + 0.5 * goal_rgb
  pred_action_img = 0.5 * obs_rgb + 0.5 * goal_rgb

  pick = (int(pick[0]), int(pick[1]))
  place = (int(place[0]), int(place[1]))
  pred_pick = (int(pred_pick[0]), int(pred_pick[1]))
  pred_place = (int(pred_place[0]), int(pred_place[1]))

  cv2.arrowedLine(true_action_img, pick, place, (0, 0, 255), 1)  # Red arrow with width 2
  cv2.circle(true_action_img, pick, 1, (255, 255, 0), -1)  # Cyan circle with radius 4

  cv2.arrowedLine(pred_action_img, pred_pick, pred_place, (255, 0, 255), 1)  # Purple arrow with width 2
  cv2.circle(pred_action_img, pred_pick, 1, (0, 0, 255), -1)  # Red circle with radius 4

  img = np.hstack([obs_rgb, goal_rgb, pick_map_obs, place_map_goal, true_action_img, pred_action_img])
  return img