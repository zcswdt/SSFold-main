import cv2
import torch
import glob
import json
import numpy as np
import argparse
from tqdm import tqdm


def correct_action(pick, place, obs_mask, nobs_mask):
    kernel = np.ones((5, 5), np.uint8)

    # Convert bool to uint8
    obs_mask = obs_mask.astype(np.uint8)
    nobs_mask = nobs_mask.astype(np.uint8)

    obs_mask = cv2.erode(obs_mask, kernel, iterations=1)
    nobs_mask = cv2.erode(nobs_mask, kernel, iterations=1)

    # # Move pick to nearest non-zero pixel
    if obs_mask[pick[1], pick[0]] == False:
        pick = move_pick_to_nonzero(pick, obs_mask)
    if nobs_mask[place[1], place[0]] == 0:
        place = move_pick_to_nonzero(place, nobs_mask)

    pick = np.clip(pick, 0, 299)
    place = np.clip(place, 0, 299)
    return pick, place


def move_pick_to_nonzero(pos, mask):
    pos = np.array(pos)
    inds = np.argwhere(mask)
    # reverse xy of inds
    inds = inds[:, [1, 0]]

    dists = np.linalg.norm(inds - pos, axis=1)
    return inds[np.argmin(dists)]


def get_mask(img, ir):
    # set img to 0 where ir < 0.5
    ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    mask = ir > 20
    return mask


def draw_action(obs, nobs, pick, place, name):
    # cv2.imshow('obs', obs)
    # cv2.imshow('nobs', nobs)
    # cv2.waitKey(0)
    obs = obs.copy()/255.
    nobs = nobs.copy()/255.
    act_img = 0.5*obs + 0.5 * nobs
    cv2.circle(act_img, (int(pick[0]), int(pick[1])), 5, (0, 0, 1), 2)
    cv2.arrowedLine(act_img, (int(pick[0]), int(pick[1])), (int(place[0]), int(place[1])), (0, 0, 1), 2)
    cv2.imshow(name, act_img)
    # cv2.waitKey(0)


def normalize_depth(depth, mask):
    # depth = cv2.resize(depth, (64, 64))



    depth[mask==False] = 0

    # viz = depth.copy()
    # min_depth = np.min(depth)
    # print("MIN DEPTH", min_depth)
    # max_depth = np.max(depth)
    # print("MAX DEPTH", max_depth)
    # depth[mask==False] = 0
    # viz = (viz - min_depth) / (max_depth - min_depth)
    # viz = 1-viz
    # viz = (viz - 1854.)/(2210. - 1854.)
    # print(np.max(viz))
    # viz[mask==False] = 0
    # viz = cv2.resize(viz, (64, 64))
    # cv2.imwrite("viz.png", viz/255.)
    # print(np.max(depth))
    # print(np.min(depth))

    # cv2.imshow('viz', depth)
    # cv2.waitKey(0)

    return depth


def get_min_max(dataset_paths):
    min_depth = np.inf
    max_depth = -np.inf
    for episode_path in dataset_paths:
        for depth_file in glob.glob("{}/*_depth.npy".format(episode_path)):
            depth = np.load(depth_file)
            depth = cv2.warpPerspective(depth, h_ir, (300, 300))

            valid_depth = depth[depth > 0]
            if not valid_depth.size:
                continue

            min_depth = min(min_depth, np.min(valid_depth))
            max_depth = max(max_depth, np.max(valid_depth))

    #         # 可视化处理
    #         if min_depth < max_depth:  # 确保不会发生除以零的错误
    #             viz_img = depth.copy()
    #             viz_img = (viz_img - min_depth) / (max_depth - min_depth)
    #             viz_img = 1 - viz_img
    #             viz_img = (viz_img * 255).astype(np.uint8)  # 转换为8位图像
    #             cv2.imshow('Depth Visualization', viz_img)
    #             cv2.waitKey(1)  # 使用 1ms 延迟
    #
    # cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
    print("MIN DEPTH", min_depth)
    print("MAX DEPTH", max_depth)
    return float(min_depth), float(max_depth)




def parse_args():
  parser = argparse.ArgumentParser()
  # parser.add_argument("--datasets", type=str, required=False, help="Paths to process", nargs='+',default=["data/red_triangle_val"])
  # parser.add_argument("--output_name", type=str, required=False, help="Name of output dataset",default="red_triangle_val")
  parser.add_argument("--datasets", type=str, required=False, help="Paths to process", nargs='+',default=["data_my/huatu"])
  parser.add_argument("--output_name", type=str, required=False, help="Name of output dataset",default="test"
                                                                                                       "")
  return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()


    with open("./{}/homograph_data.json".format(args.datasets[0]), "r") as f:
        homograph_data = json.load(f)
        crop_keypoints = np.array(homograph_data["crop"])
        ir_keypoints   = np.array(homograph_data["ir"])
        rgb_keypoints  = np.array(homograph_data["rgb"])


    h_ir, _ = cv2.findHomography(ir_keypoints, crop_keypoints)
    h_rgb, _ = cv2.findHomography(rgb_keypoints, crop_keypoints)

    buf = []

    dataset_paths = []
    for dataset in args.datasets:
        dataset_paths += glob.glob("./{}/*".format(dataset))

    min_depth, max_depth = get_min_max(dataset_paths)
    #min_depth = 0.8080161
    #max_depth = 0.92840046
    for episode_path in tqdm(dataset_paths):
        timestep = 0
        next_timestep_depth_path = "{}/{}_depth.npy".format(episode_path, timestep+1)
        while next_timestep_depth_path in glob.glob("{}/*_depth.npy".format(episode_path)):
            try:
                o_file = "{}/{}_".format(episode_path, timestep)
                o_rgb = cv2.imread(o_file + "rgb.png")
                o_ir = cv2.imread(o_file + "ir.png")
                o_dep = np.load(o_file + "depth.npy")

                n_file = "{}/{}_".format(episode_path, timestep+1)
                n_rgb = cv2.imread(n_file + "rgb.png")
                n_ir = cv2.imread(n_file + "ir.png")
                n_dep = np.load(n_file + "depth.npy")
                
                o_ir = cv2.warpPerspective(o_ir, h_ir, (300, 300))
                o_dep = cv2.warpPerspective(o_dep, h_rgb, (300, 300))  #和RGB对应
                n_ir = cv2.warpPerspective(n_ir, h_ir, (300, 300))
                n_dep = cv2.warpPerspective(n_dep, h_rgb, (300, 300))  ##和RGB对应

                o_rgb = cv2.warpPerspective(o_rgb, h_rgb, (300, 300))
                n_rgb = cv2.warpPerspective(n_rgb, h_rgb, (300, 300))

                action = json.load(open(n_file + "action.json"))
                pick = action["pick"]
                place = action["place"]

                # Transform index to crop
                pick = np.array(pick)
                place = np.array(place)
                pick = np.dot(h_rgb, np.array([pick[0], pick[1], 1]))
                place = np.dot(h_rgb, np.array([place[0], place[1], 1]))
                pick = (pick[:2]/pick[2]).astype(int)
                place = (place[:2]/place[2]).astype(int)
                # Clip to 300
                pick = np.clip(pick, 0, 299)
                place = np.clip(place, 0, 299)



                #使用红外光来获得mask
                o_mask = get_mask(o_rgb, o_ir)
                n_mask = get_mask(n_rgb, n_ir)

                o_rgb[o_mask==False] = 0
                n_rgb[n_mask==False] = 0
                # cv2.imshow('viz', o_rgb)
                cv2.waitKey(0)


                # norm_o_dep = normalize_depth(o_dep, o_mask)
                # norm_n_dep = normalize_depth(n_dep, n_mask)

                # max_dep.append(np.max(norm_o_dep))
                # min_dep.append(np.min(norm_o_dep))

                draw_action(o_rgb, n_rgb, pick, place, "rgb")
                cv2.waitKey(0)
                pick, place = correct_action(pick, place, o_mask, n_mask)
                draw_action(o_rgb, n_rgb, pick, place, "rgb")

                # Downscale pick + place inds from 300x300 to 64x64
                pick = list((pick.astype(float) / 300 * 64).astype(int))
                place = list((place.astype(float) / 300 * 64).astype(int))
                print(pick, place)

                #
                # pick = pick[0]/(300//64), pick[1]/(300//64)
                # place = place[0]/(300//64), place[1]/(300//64)
                # small_depth =cv2.resize(norm_o_dep, (64, 64))#, cv2.INTER_NEAREST)
                # cv2.imshow('viz', small_depth)
                # cv2.waitKey(0)
                # print("----")
                # print("MIN SMALL", np.min(small_depth[small_depth>0]))

                # obs = {"rgb": cv2.resize(o_rgb, (64, 64)), "depth": cv2.resize(norm_o_dep, (64, 64))}
                # nobs = {"rgb": cv2.resize(n_rgb, (64, 64)), "depth": cv2.resize(norm_n_dep, (64, 64))}

                obs = {
                    "rgb": cv2.resize(o_rgb, (64, 64)),
                    "depth": cv2.resize(o_dep, (64, 64)),
                    "mask": cv2.resize(o_mask.astype(np.uint8), (64, 64))  # 添加mask到obs字典中，并转换为uint8类型
                }
                nobs = {
                    "rgb": cv2.resize(n_rgb, (64, 64)),
                    "depth": cv2.resize(n_dep, (64, 64)),
                    "mask": cv2.resize(n_mask.astype(np.uint8), (64, 64))  # 添加mask到nobs字典中，并转换为uint8类型
                }


                buf.append({"obs": obs, "nobs": nobs, "pick": pick, "place": place})

                    # print("global min and max")
                    # print(np.min(min_dep), np.max(max_dep))
                timestep += 1
                next_timestep_depth_path = "{}/{}_depth.npy".format(episode_path, timestep+1)
            except Exception as e:
                print(e)
                break
    print("Training dataset size: {}".format(len(buf)))
    # Save data to output name path
    torch.save(buf, "./data_my/huatu/{}_buf.pt".format(args.output_name))







