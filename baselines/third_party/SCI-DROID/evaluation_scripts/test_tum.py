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

import torch.nn.functional as F
from droid import Droid
from droid_async import DroidAsync

import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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


# =========================================================
# 【新增 1】：引入 TPAMI 版本的 SCI 快速推理类 Finetunemodel
# =========================================================
from sci_model import Finetunemodel

# =========================================================
# 【新增 2】：弱光与噪点仿真模块 (噪点 0.002 保障光流存活)
# =========================================================
def simulate_low_light(image_np, low_light_factor=0.1, noise_variance=0.002):
    img_float = image_np.astype(np.float32) / 255.0
    img_linear = img_float ** 2.2
    
    img_dark = img_linear * low_light_factor
    noise = np.random.normal(scale=noise_variance, size=img_dark.shape)
    img_noisy = np.clip(img_dark + noise, 0.0, 1.0)
    
    img_final = img_noisy ** (1/2.2)
    return (img_final * 255).astype(np.uint8)


def show_image(image):
    # 【修复】：加上 [0] 消除 Batch 维度，防止 permute 崩溃
    img_vis = image[0].permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', img_vis / 255.0)
    cv2.waitKey(1)


# =========================================================
# 【修改 1】：将函数改为生成器 (yield)，并传入 sci_net
# =========================================================
def image_stream(datapath, sci_net=None, lpips_fn=None, log_file=None):
    """ image generator """
    fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])

    images_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::2]
    
    # === [新增] 确保存图文件夹存在 ===
    save_dir = "enhanced_images"
    os.makedirs(save_dir, exist_ok=True)

    for t, imfile in enumerate(images_list):
        # 提取时间戳作为文件名
        tstamp_str = Path(imfile).stem
        
        image = cv2.imread(imfile)
        gt_img = image.copy() # === [新增] 保留无损的 GT 图像用于计算指标 ===

        # 1. 制造弱光环境 + 噪点
        image = simulate_low_light(image, low_light_factor=0.1, noise_variance=0.01)

        # 2. SCI 图像增强
        if sci_net is not None:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0).cuda() / 255.0
            
            with torch.no_grad():
                _, r = sci_net(img_tensor) 
            
            enhanced_numpy = (r[0].permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
            image = cv2.cvtColor(enhanced_numpy, cv2.COLOR_RGB2BGR)

        # =========================================================
        # === [新增] 按时间戳保存增强图并实时计算指标 ===
        # =========================================================
        cv2.imwrite(os.path.join(save_dir, f"{tstamp_str}.png"), image)

        if lpips_fn is not None and log_file is not None:
            # 1. PSNR / SSIM (灰度空间)
            gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
            enh_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            cur_psnr = psnr(gt_gray, enh_gray)
            cur_ssim = ssim(gt_gray, enh_gray)
            
            # 2. LPIPS (RGB 张量空间)
            gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            enh_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            t_gt = lpips.im2tensor(gt_rgb).cuda()
            t_enh = lpips.im2tensor(enh_rgb).cuda()
            with torch.no_grad():
                cur_lpips = lpips_fn(t_gt, t_enh).item()
                
            # 3. 图像特征熵 (灰度空间概率分布)
            hist = cv2.calcHist([enh_gray], [0], None, [256], [0, 256]).ravel()
            hist = hist / hist.sum()
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            
            # 写入日志
            log_file.write(f"{tstamp_str}, {cur_psnr:.4f}, {cur_ssim:.4f}, {cur_lpips:.4f}, {entropy:.4f}\n")
        # =========================================================

        # 3. DROID-SLAM 预处理
        ht0, wd0, _ = image.shape
        image = cv2.undistort(image, K_l, d_l)
        image = cv2.resize(image, (320+32, 240+16))
        image = torch.from_numpy(image).permute(2,0,1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0] *= image.shape[2] / 640.0
        intrinsics[1] *= image.shape[1] / 480.0
        intrinsics[2] *= image.shape[2] / 640.0
        intrinsics[3] *= image.shape[1] / 480.0

        intrinsics[2] -= 16
        intrinsics[3] -= 8
        image = image[:, 8:-8, 16:-16]

        yield t, image[None], intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
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

    # 【新增】：接收 SCI 的统一权重参数
    parser.add_argument("--sci_weights", type=str, default="weights_1_3500.pt", help="SCI weights path")

    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    # =========================================================
    # 【新增】：实例化 SCI 极速推理模型并加载权重
    # =========================================================
    print(f"Loading SCI model from {args.sci_weights} ...")
    sci_net = Finetunemodel(args.sci_weights).cuda() 
    sci_net.eval()
    print("SCI Loaded Successfully!")
    # =========================================================
    
    # === [新增] 初始化 LPIPS 和 日志文件 ===
    print("Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    
    metrics_log = open("enhancement_metrics.txt", "w")
    metrics_log.write("Timestamp, PSNR, SSIM, LPIPS, Entropy\n")
    open("droid_confidence_ratio.txt", "w").close() # 清空上一轮光流置信度

    droid = DroidAsync(args) if args.asynchronous else Droid(args)
    scene = Path(args.datapath).name

    # 初始化包含增强网络的生成器
    images_gen = image_stream(args.datapath, sci_net=sci_net, lpips_fn=lpips_fn, log_file=metrics_log)
    
    image_path = os.path.join(args.datapath, 'rgb')
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::2]
    total_frames = len(images_list)

    # 带着 total_frames 跑 tqdm 进度条，不会再假死
    for (t, image, intrinsics) in tqdm(images_gen, total=total_frames, desc=scene):
        if not args.disable_vis:
            show_image(image)
        droid.track(t, image, intrinsics=intrinsics)
        

    metrics_log.close()

    # 【修复】：重新实例化生成器收尾，防止 terminate 空解包报错
    filler_gen = image_stream(args.datapath, sci_net=sci_net)
    traj_est = droid.terminate(filler_gen)

    ### run evaluation ###

    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))

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

    # ------------------------------------------------
    # [新增] 计算图像增强与前端跟踪指标的平均值并打印
    # ------------------------------------------------
    print("\n" + "="*50)
    print(">>> 图像增强质量指标 (Sequence Averages)")
    print("="*50)
    
    psnr_list, ssim_list, lpips_list, entropy_list = [], [], [], []
    if os.path.exists("enhancement_metrics.txt"):
        with open("enhancement_metrics.txt", "r") as f:
            lines = f.readlines()[1:] # 跳过表头
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    psnr_list.append(float(parts[1]))
                    ssim_list.append(float(parts[2]))
                    lpips_list.append(float(parts[3]))
                    entropy_list.append(float(parts[4]))
        
        if psnr_list:
            print(f"Average PSNR    : {np.mean(psnr_list):.4f} dB")
            print(f"Average SSIM    : {np.mean(ssim_list):.4f}")
            print(f"Average LPIPS   : {np.mean(lpips_list):.4f}")
            print(f"Average Entropy : {np.mean(entropy_list):.4f}")

    print("\n" + "="*50)
    print(">>> 前端跟踪稳定性 (等效内点率 / Confidence Ratio)")
    print("="*50)
    
    conf_list = []
    if os.path.exists("droid_confidence_ratio.txt"):
        with open("droid_confidence_ratio.txt", "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    conf_list.append(float(parts[1]))
        
        if conf_list:
            print(f"Average High Confidence Ratio: {np.mean(conf_list) * 100:.2f} %")
    print("="*50 + "\n")