import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
from collections import defaultdict

from sci_model import Finetunemodel

# ==========================================
# [新增] 导入 evo 相关库
# ==========================================
try:
    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core import sync
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core.metrics import PoseRelation, Unit
except ImportError:
    print("未检测到 evo 库，请先在终端运行: pip install evo --upgrade --no-binary evo")
    sys.exit(1)

from pathlib import Path
from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync


import torch.nn.functional as F

def simulate_low_light(image_np, low_light_factor=0.1, noise_variance=0.03):
    """
    1. 逆 Gamma 校正回线性空间
    2. 降低物理照度 (low_light_factor)
    3. 添加近似泊松-高斯噪声 (传感器暗光噪声)
    4. Gamma 校正回 sRGB 空间
    """
    img_float = image_np.astype(np.float32) / 255.0
    img_linear = img_float ** 2.2
    
    # 降低照度
    img_dark = img_linear * low_light_factor
    
    # 注入传感器噪声
    noise = np.random.normal(scale=noise_variance, size=img_dark.shape)
    img_noisy = np.clip(img_dark + noise, 0.0, 1.0)
    
    img_final = img_noisy ** (1/2.2)
    return (img_final * 255).astype(np.uint8)


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(datapath, image_size=[320, 512], stereo=False, stride=1, sci_net=None):
    """ image generator """

    K_l = np.array([458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
    R_l = np.array([
         0.999966347530033, -0.001422739138722922, 0.008079580483432283, 
         0.001365741834644127, 0.9999741760894847, 0.007055629199258132, 
        -0.008089410156878961, -0.007044357138835809, 0.9999424675829176
    ]).reshape(3,3)
    
    P_l = np.array([435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0]).reshape(3,4)
    map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3,:3], (752, 480), cv2.CV_32F)
    
    K_r = np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(3,3)
    d_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]).reshape(5)
    R_r = np.array([
         0.9999633526194376, -0.003625811871560086, 0.007755443660172947, 
         0.003680398547259526, 0.9999684752771629, -0.007035845251224894, 
        -0.007729688520722713, 0.007064130529506649, 0.999945173484644
    ]).reshape(3,3)
    
    P_r = np.array([435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0]).reshape(3,4)
    map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (752, 480), cv2.CV_32F)

    intrinsics_vec = [435.2046959714599, 435.2046959714599, 367.4517211914062, 252.2008514404297]
    ht0, wd0 = [480, 752]

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(datapath, 'mav0/cam0/data/*.png')))[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]

    data_list = []

    save_dir = "enhanced_images"
    os.makedirs(save_dir, exist_ok=True)

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if stereo and not os.path.isfile(imgR):
            continue
        tstamp = float(imgL.split('/')[-1][:-4])        
        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        if stereo:
            images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]

        images = [simulate_low_light(img, low_light_factor=0.1, noise_variance=0.01) for img in images]

        # ==========================================
        # [新增] SCI 图像增强预处理
        # ==========================================
        if sci_net is not None:
            enhanced_images = []
            for img in images:
                # 1. BGR 转 RGB，并转为 0~1 的 Tensor 送入 GPU
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0).cuda() / 255.0
                
                with torch.no_grad():
                    # 2. SCI 推理：返回 i(光照) 和 r(反射/增强图)
                    i_illumination, r_enhanced = sci_net(img_tensor)
                
                # 3. 提取增强后的图像 r_enhanced，限制溢出，转回 uint8
                enhanced_numpy = np.clip(r_enhanced[0].permute(1,2,0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                
                # 4. 转回 BGR
                enhanced_img = cv2.cvtColor(enhanced_numpy, cv2.COLOR_RGB2BGR)

                # =========================================================
                # 🌟【新增】：高精度保存模块
                # =========================================================
                tstamp_str = "{:.6f}".format(tstamp)
                
                # 直接将文件名拼接到 enhanced_images 文件夹下
                save_path = os.path.join(save_dir, f"{tstamp_str}.png")
                
                cv2.imwrite(save_path, enhanced_img)
                # =========================================================

                enhanced_images.append(enhanced_img)
            
            images = enhanced_images
        # ==========================================
        
        images = [cv2.resize(image, (image_size[1], image_size[0])) for image in images]
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to(dtype=torch.float32)
        
        intrinsics = torch.as_tensor(intrinsics_vec)
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        data_list.append((stride*t, images, intrinsics))

    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence")
    parser.add_argument("--gt", help="path to gt file")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[320,512])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stereo", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--keyframe_thresh", type=float, default=3.0)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)

    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")

    parser.add_argument("--sci_weights", type=str, default="weights_1_3500.pt", help="SCI weights path")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    # ==========================================
    # [新增] 实例化并加载 SCI 网络
    # ==========================================
    print(f"Loading SCI model from {args.sci_weights} ...")
    sci_net = Finetunemodel(args.sci_weights).cuda()
    sci_net.eval()
    print("SCI Model Loaded Successfully!")
    # ==========================================

    droid = DroidAsync(args) if args.asynchronous else Droid(args)
    scene = Path(args.datapath).name

    images = image_stream(args.datapath, stereo=args.stereo, stride=1, sci_net=sci_net)

    # run with stride 2
    for (t, image, intrinsics) in tqdm(images[::2], desc=scene):
        droid.track(t, image, intrinsics=intrinsics)
    
    torch.cuda.empty_cache()

    # fill in missing poses with stride 1
    traj_est = droid.terminate(images)

    ### run evaluation ###

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    images_list = sorted(glob.glob(os.path.join(args.datapath, 'mav0/cam0/data/*.png')))
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    quat_est_wxyz = traj_est[:, [6, 3, 4, 5]]

    # 2. 构建正确的 Evo 轨迹对象
    traj_est = PoseTrajectory3D(
        positions_xyz=1.10 * traj_est[:, :3],
        orientations_quat_wxyz=quat_est_wxyz,
        timestamps=np.array(tstamps)
    )

    os.makedirs("euroc_trajectories", exist_ok=True)
    save_path = os.path.join("euroc_trajectories", f"{scene}.txt")
    file_interface.write_tum_trajectory_file(save_path, traj_est)
    print(f"\n[INFO] 序列 {scene} 的轨迹已成功保存至: {save_path}")

    traj_ref = file_interface.read_tum_trajectory_file(args.gt)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)

    print("\n>>> 2. RPE (平移误差 / 米) - Delta = 1 帧:")
    result_rpe_trans = main_rpe.rpe(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True,
        delta=1, delta_unit=Unit.frames, all_pairs=False)
    print(result_rpe_trans)

    # ------------------------------------------------
    # 3. 计算 RPE - 旋转误差 (Rotation)
    # ------------------------------------------------
    print("\n>>> 3. RPE (旋转误差 / 角度或弧度) - Delta = 1 帧:")
    result_rpe_rot = main_rpe.rpe(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.rotation_angle_deg, align=True, correct_scale=True,
        delta=1, delta_unit=Unit.frames, all_pairs=False)
    # 如果想输出角度 (degree) 可以用 PoseRelation.rotation_angle_deg
    print(result_rpe_rot)