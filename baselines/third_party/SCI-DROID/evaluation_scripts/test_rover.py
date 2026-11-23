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

# ==========================================
# [新增] 导入 SCI 图像增强模型
# ==========================================
from sci_model import Finetunemodel

# ==========================================
# 导入 evo 相关库
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

# ==========================================
# [修改] 加入 sci_net 参数，并使用 yield 生成器节约内存
# ==========================================
def image_stream(datapath, sci_net=None):
    """ image generator """

    fx, fy, cx, cy = 596.1950425915236, 593.1411835433107, 327.0463454538411, 245.16142133264628

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([0.07561607662577634, -0.2088729975389971, -0.0023347441072184756, 0.004010356017731083])

    # 统一在这里跳帧读取 [::2]，避免在下面引发“双重跳帧”导致的时间戳错位
    images_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::2]
    
    data_list = []
    for t, imfile in enumerate(images_list):
        image = cv2.imread(imfile)

        # 模拟弱光
        image = simulate_low_light(image, low_light_factor=0.1, noise_variance=0.01)

        # ------------------------------------------
        # [核心] SCI 图像增强预处理
        # ------------------------------------------
        if sci_net is not None:
            # BGR 转 RGB，转为 0~1 的 Tensor 送入 GPU
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0).cuda() / 255.0
            
            with torch.no_grad():
                # SCI 前向推理：获取光照图 i 和 反射增强图 r
                _, r_enhanced = sci_net(img_tensor)
            
            # 取出 r_enhanced，防溢出限制，转回 numpy uint8
            enhanced_numpy = np.clip(r_enhanced[0].permute(1,2,0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(enhanced_numpy, cv2.COLOR_RGB2BGR)
        # ------------------------------------------

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
    parser.add_argument("--buffer", type=int, default=5000)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=1.5)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.0)
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

    # ==========================================
    # [新增] SCI 权重参数
    # ==========================================
    parser.add_argument("--sci_weights", type=str, default="weights_1_3500.pt", help="SCI weights path")

    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    # ==========================================
    # [新增] 加载 SCI 模型
    # ==========================================
    print(f"Loading SCI model from {args.sci_weights} ...")
    sci_net = Finetunemodel(args.sci_weights).cuda()
    sci_net.eval()
    print("SCI Model Loaded Successfully!")

    droid = DroidAsync(args) if args.asynchronous else Droid(args)
    scene = Path(args.datapath).name

    tstamps = []
    images = image_stream(args.datapath)

    for (t, image, intrinsics) in tqdm(images[::2], desc=scene):
        if not args.disable_vis:
            show_image(image)
        droid.track(t, image, intrinsics=intrinsics)

    torch.cuda.empty_cache()
    traj_est = droid.terminate(images)

    ### run evaluation ###

    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    image_path = os.path.join(args.datapath, 'rgb')
    
    # [必须有 [::2]]：提取时间戳时也必须跳帧，保持与轨迹长度一致
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::2]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    # [新增] 安全防线：万一有一两帧的误差，自动截断对齐，防止 Evo 崩溃
    if len(tstamps) != traj_est.shape[0]:
        print(f"[警告] 时间戳 ({len(tstamps)}) 与 轨迹 ({traj_est.shape[0]}) 长度不匹配，执行自动对齐！")
        min_len = min(len(tstamps), traj_est.shape[0])
        tstamps = tstamps[:min_len]
        traj_est = traj_est[:min_len]

    # 修正四元数顺序为 [qw, qx, qy, qz]
    quat_est_wxyz = traj_est[:, [6, 3, 4, 5]]

    # 构建统一的 Evo 轨迹对象 (traj_est_evo)
    traj_est_evo = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=quat_est_wxyz,
        timestamps=np.array(tstamps))

    os.makedirs("rover_trajectories", exist_ok=True)
    save_path = os.path.join("rover_trajectories", f"{scene}.txt")
    file_interface.write_tum_trajectory_file(save_path, traj_est_evo)
    print(f"\n[INFO] 序列 {scene} 的轨迹已成功保存至: {save_path}")

    gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    # 使用统一的 traj_est_evo 去计算和关联误差
    traj_ref, traj_est_evo = sync.associate_trajectories(traj_ref, traj_est_evo)
    
    result = main_ape.ape(traj_ref, traj_est_evo, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)

    print("\n>>> 2. RPE (平移误差 / 米) - Delta = 1 帧:")
    result_rpe_trans = main_rpe.rpe(traj_ref, traj_est_evo, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True,
        delta=1, delta_unit=Unit.frames, all_pairs=False)
    print(result_rpe_trans)

    # ------------------------------------------------
    # 3. 计算 RPE - 旋转误差 (Rotation)
    # ------------------------------------------------
    print("\n>>> 3. RPE (旋转误差 / 角度或弧度) - Delta = 1 帧:")
    result_rpe_rot = main_rpe.rpe(traj_ref, traj_est_evo, est_name='traj', 
        pose_relation=PoseRelation.rotation_angle_deg, align=True, correct_scale=True,
        delta=1, delta_unit=Unit.frames, all_pairs=False)
    print(result_rpe_rot)