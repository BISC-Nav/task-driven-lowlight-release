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
from pathlib import Path
from collections import defaultdict

from zero_dce_model import enhance_net_nopool

import torch.nn.functional as F

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

import torch.nn.functional as F
from droid import Droid
from droid_async import DroidAsync

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
    image = image[0].permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(datapath, dce_net=None):
    """ image generator """

    fx, fy, cx, cy = 596.1950425915236, 593.1411835433107, 327.0463454538411, 245.16142133264628

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([0.07561607662577634, -0.2088729975389971, -0.0023347441072184756, 0.004010356017731083])

    # read all png images in folder
    images_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))
    
    data_list = []
    for t, imfile in enumerate(images_list):
        image = cv2.imread(imfile)

        image = simulate_low_light(image, low_light_factor=0.1, noise_variance=0.01)

        # ==========================================
        # [新增] Zero-DCE 图像增强
        # ==========================================
        if dce_net is not None:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0).cuda() / 255.0
            
            with torch.no_grad():
                _, enhanced_tensor, _ = dce_net(img_tensor)
            
            # 使用 np.clip 防止溢出，转回 BGR
            enhanced_numpy = np.clip(enhanced_tensor[0].permute(1,2,0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(enhanced_numpy, cv2.COLOR_RGB2BGR)
        # ==========================================

        ht0, wd0, _ = image.shape
        image = cv2.undistort(image, K_l, d_l)
        image = cv2.resize(image, (320+32, 240+16))
        image = torch.from_numpy(image).permute(2,0,1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0] *= image.shape[2] / 640.0
        intrinsics[1] *= image.shape[1] / 480.0
        intrinsics[2] *= image.shape[2] / 640.0
        intrinsics[3] *= image.shape[1] / 480.0

        # crop image to remove distortion boundary
        intrinsics[2] -= 16
        intrinsics[3] -= 8
        image = image[:, 8:-8, 16:-16]

        data_list.append((t, image[None], intrinsics))

    return data_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=6000)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=1.5)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0) # 
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    parser.add_argument("--upsample", action="store_true")
    
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    parser.add_argument("--motion_damping", type=float, default=0.5)

    parser.add_argument("--dce_weights", type=str, default="Epoch99.pth", help="Zero-DCE weights path")

    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    # ==========================================
    # [新增] 实例化并加载 Zero-DCE 网络
    # ==========================================
    print(f"Loading Zero-DCE model from {args.dce_weights} ...")
    dce_net = enhance_net_nopool().cuda()
    dce_net.load_state_dict(torch.load(args.dce_weights))
    dce_net.eval()
    print("Zero-DCE Loaded Successfully!")
    # ==========================================

    droid = DroidAsync(args) if args.asynchronous else Droid(args)
    scene = Path(args.datapath).name

    tstamps = []
    images = image_stream(args.datapath, dce_net=dce_net)

    images_half = images[::2]

    for (t, image, intrinsics) in tqdm(images_half, desc=scene):
        if not args.disable_vis:
            show_image(image)
        droid.track(t, image, intrinsics=intrinsics)

    torch.cuda.empty_cache()
    traj_est = droid.terminate(images_half)

    ### run evaluation ###

    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    image_path = os.path.join(args.datapath, 'rgb')
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::2]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    if len(tstamps) != traj_est.shape[0]:
        print(f"[警告] 时间戳 ({len(tstamps)}) 与 轨迹 ({traj_est.shape[0]}) 长度不匹配，强行截断！")
        min_len = min(len(tstamps), traj_est.shape[0])
        tstamps = tstamps[:min_len]
        traj_est = traj_est[:min_len]

    quat_est_wxyz = traj_est[:, [6, 3, 4, 5]]

    # 构建 Evo 对象，传入正确的四元数
    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=quat_est_wxyz,
        timestamps=np.array(tstamps))

    os.makedirs("rover_trajectories", exist_ok=True)
    save_path = os.path.join("rover_trajectories", f"{scene}.txt")
    file_interface.write_tum_trajectory_file(save_path, traj_est)
    print(f"\n[INFO] 序列 {scene} 的轨迹已成功保存至: {save_path}")

    gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

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