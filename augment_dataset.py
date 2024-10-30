import torch
import math
import numpy as np
import scipy.ndimage as nd
from tqdm import tqdm
import cv2
import argparse
import copy
import opensimplex



def cv2_clipped_zoom(img, zoom_factor=0):

    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.          
    """
    if zoom_factor == 0:
        return img


    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(dtype=int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    
    result = cv2.resize(cropped_img, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def augment_elem(elem, angle, flipx, flipy, transx, transy, scale_factor=1.0, thickness_factor=1.0):
    obs_aug_rgb = augment_image(elem["obs"]["rgb"], angle, flipx, flipy, transx, transy, scale_factor)
    obs_aug_dep = augment_image(elem["obs"]["depth"], angle, flipx, flipy, transx, transy, scale_factor, apply_noise=True, thickness_factor=thickness_factor)
    obs_aug = {"rgb": obs_aug_rgb, "depth": obs_aug_dep}

    nobs_aug_rgb = augment_image(elem["nobs"]["rgb"], angle, flipx, flipy, transx, transy, scale_factor)
    nobs_aug_dep = augment_image(elem["nobs"]["depth"], angle, flipx, flipy, transx, transy, scale_factor, apply_noise=True, thickness_factor=thickness_factor)
    nobs_aug = {"rgb": nobs_aug_rgb, "depth": nobs_aug_dep}

    pick_aug = augment_coord(elem["pick"], angle, flipx, flipy, transx, transy, scale_factor)
    place_aug = augment_coord(elem["place"], angle, flipx, flipy, transx, transy, scale_factor)


    draw_action(elem["obs"]["rgb"], elem["nobs"]["rgb"], elem["pick"], elem["place"], "original")
    draw_action(obs_aug_rgb, nobs_aug_rgb, pick_aug, place_aug, "augmented")

    # elem = Experience(obs_aug, goal_aug, pick_aug, place_aug, elem.rew, nobs_aug)
    elem = {"obs": obs_aug, "nobs": nobs_aug, "pick": pick_aug, "place": place_aug}
    return elem


def augment_image(og_img, angle, flipx, flipy, transx, transy, scale_factor=1, apply_noise=False, thickness_factor=1.0):
    img = copy.deepcopy(og_img)

    if apply_noise:
        # Apply random noise
        noise = np.random.normal(0, 0.01, img.shape)
        img[img>0] = img[img>0] + noise[img>0]

        # Apply perlin noise
        # n = np.zeros((img.shape[0], img.shape[1]))
        # for y in range(0, img.shape[0]):
        #     for x in range(0, img.shape[1]):
        #         value = opensimplex.noise2(x / 12., y / 12.)
        #         color = int((value + 1) * 128)/255.0
        #         print(color)
        #         n[y, x] = color
        # img[img>0] = img[img>0] + n[img>0]
        # cv2.imshow("noise", n)
        # cv2.waitKey(0)

    if len(img.shape) == 2 and thickness_factor != 1.0:
        img = img*thickness_factor

    if len(img.shape) == 3:
        trans = [transx, transy, 0]
    else:
        trans = [transx, transy]

    # Apply small translation
    img = nd.shift(img, trans, order=0)



    # Apply angle rotation
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1), img.shape[:2], flags=cv2.INTER_NEAREST)


    # Apply flip in x and y axis
    if flipx:
        img = np.fliplr(img)
    if flipy:
        img = np.flipud(img)


    # Apply scale, padding or cropping to maintain original size
    img = cv2_clipped_zoom(img, scale_factor)


    disp_img = np.hstack([img.copy(), og_img])
    cv2.imshow('img', disp_img)
    cv2.waitKey(1)

    return img


def rotate_coord(point, angle):
    """
    Rotate a point counterclockwise by a given angle around center

    The angle should be given in radians.
    """
    angle = math.radians(angle)
    origin = (64/2, 64/2)
    ox, oy = origin
    py, px = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([qy, qx]).astype(int)

def augment_coord(coord, angle, flipx, flipy, transx, transy, scale_factor=1):
    # Translate coord
    coord = coord + np.array([transx, transy])

    # Apply random angle rotation
    aug_coord = rotate_coord(coord, angle)

    # Apply random flip in x and y axis
    if flipx:
        aug_coord[0] = args.image_width - aug_coord[0]
    if flipy:
        aug_coord[1] = args.image_width - aug_coord[1]

    # Scale coord from center of image (32,32)
    aug_coord = int((aug_coord[0] - 32) * scale_factor + 32), int((aug_coord[1] - 32) * scale_factor + 32)


    return aug_coord

def draw_action(obs, nobs, pick, place, name):
    obs = obs.copy()/255.0
    nobs = nobs.copy()/255.0
    act_img = 0.5*obs + 0.5 * nobs
    cv2.circle(act_img, (int(pick[0]), int(pick[1])), 3, (1, 0, 1), 1)
    cv2.arrowedLine(act_img, (int(pick[0]), int(pick[1])), (int(place[0]), int(place[1])), (0, 0, 1), 1)
    cv2.imshow(name, act_img)
    cv2.waitKey(1)

def augment_dataset(dataset):
    print("Augmenting dataset...")
    new_dataset = []
    for elem in tqdm(dataset):
        for _ in range(args.num_augmentations):
            # angle = np.random.randint(0,3) * 90
            trans_max = 0
            flipx = np.random.randint(0, 2)
            flipy = np.random.randint(0, 2)
            # scale_factor = 1
            # thickness_factor = 1
            scale_factor = np.random.uniform(0.8, 1.2)
            thickness_factor = np.random.uniform(0.8, 1.2)

            # Choose angle that doesnt rotate the pick and place points out of the image, for any transition in the episode
            angle = np.random.uniform(0, 360)
            transx = 0#np.random.randint(-trans_max, trans_max)
            transy = 0#np.random.randint(-trans_max, trans_max)
            valid = False
            attempts = 0
            while not valid:
                pick = elem["pick"]
                place = elem["place"]
                pick_aug = augment_coord(pick, angle, flipx, flipy, transx, transy, scale_factor)
                place_aug = augment_coord(place, angle, flipx, flipy, transx, transy, scale_factor)
                valid = True
                if pick_aug[0] < 0 or pick_aug[0] >= 64 or pick_aug[1] < 0 or pick_aug[1] >= 64:
                    valid = False
                if place_aug[0] < 0 or place_aug[0] >= 64 or place_aug[1] < 0 or place_aug[1] >= 64:
                    valid = False
                if not valid:
                    angle = np.random.uniform(0, 360)
                    scale_factor = np.random.uniform(0.8, 1.2)
                    transx = 0#np.random.randint(-trans_max, trans_max)
                    transy = 0#np.random.randint(-trans_max, trans_max)
                if attempts > 100:
                    angle = np.random.randint(0,3) * 90
                    scale_factor = np.random.uniform(0.8, 1.2)
                    transx = 0#np.random.randint(-trans_max, trans_max)
                    transy = 0#np.random.randint(-trans_max, trans_max)

                    print(pick, place, pick_aug, place_aug)
                    print("trying 90")

                attempts += 1

            new_elem = augment_elem(elem, angle, flipx, flipy, scale_factor, thickness_factor)
            new_dataset.append(new_elem)
    return new_dataset



def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_path", type=str, required=False, help="Path to dataset",default="/home/zcs/work/github/folding-by-hand/data_my/test_buf.pt")
  parser.add_argument("--num_augmentations", default=20, type=int, required=False, help="Number of augmentations per element")
  parser.add_argument("--image_width", default=64, type=int, required=False, help="Image width")
  return parser.parse_args()


TRANS = 5
if __name__ == "__main__":
    args = parse_args()
    print(args.num_augmentations)
    print(args.dataset_path)
    # Load data
    data = torch.load(args.dataset_path)

    # Augment
    new_dataset = augment_dataset(data)

    # Save
    print(len(new_dataset))
    torch.save(new_dataset, args.dataset_path.replace(".pt", "_augmented.pt"))


