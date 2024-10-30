from .utils import *
import numpy as np
import torch

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------

# def list_of_imgs_to_torch(list_of_imgs):
#   return torch.FloatTensor(np.array(list_of_imgs)).cuda().permute(0,3,1,2)

#------------------------------------------------------------------------------
#
# def get_action(model, obs, goal, args):
#   # Get heatmaps
#   pick_gauss, pick_indices = get_all_pick_gauss(obs, args)
#   q_maps = F.sigmoid(get_qmaps(model, obs, goal, pick_gauss))
#   pick_map , best_pick_num = get_pickmap(q_maps, pick_indices)
#   pick, place = get_pick_place_from_maps(pick_map, q_maps, best_pick_num)
#
#   cv2.arrowedLine(obs, tuple(pick), tuple(place), (1.0,0,0), 2)
#   cv2.circle(obs, tuple(pick), 5, (1.0,0,0), 2)
#   cv2.imshow("Action", obs)
#
#   # pick = pick[::-1]
#   # place= place[::-1]
#   action = np.hstack([pick, place])/args.image_width
#   return action
#
# #------------------------------------------------------------------------------


