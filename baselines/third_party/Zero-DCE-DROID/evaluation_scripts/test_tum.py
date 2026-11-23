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

import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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

def simulate_low_light(image_np, low_light_factor=0.1, noise_variance=0.01):
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

def image_stream(datapath, dce_net=None, lpips_fn=None, log_file=None):
    """ image generator """
    fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])

    images_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))
    
    # 确保存图文件夹存在
    save_dir = "enhanced_images"
    os.makedirs(save_dir, exist_ok=True)
    
    data_list = []
    total_enhance_time = 0.0
    
    for t, imfile in enumerate(images_list):
        # 提取时间戳作为文件名 (例如: "1305031102.175304")
        tstamp_str = Path(imfile).stem
        
        image = cv2.imread(imfile)
        gt_img = image.copy() # 保存原始 GT 图像用于计算指标

        image = simulate_low_light(image, low_light_factor=0.1, noise_variance=0.01)

        # ==========================================
        # 2. Zero-DCE 图像增强
        # ==========================================
        if dce_net is not None:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0).cuda() / 255.0
            
            torch.cuda.synchronize()
            t_enh_start = time.perf_counter()

            with torch.no_grad():
                _, enhanced_tensor, _ = dce_net(img_tensor)
            
            torch.cuda.synchronize()
            t_enh_end = time.perf_counter()
            total_enhance_time += (t_enh_end - t_enh_start)
            
            enhanced_numpy = np.clip(enhanced_tensor[0].permute(1,2,0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(enhanced_numpy, cv2.COLOR_RGB2BGR)
        # ==========================================

        # === [新增] 按时间戳保存增强图并计算指标 ===
        cv2.imwrite(os.path.join(save_dir, f"{tstamp_str}.png"), image)

        if lpips_fn is not None and log_file is not None:
            # 1. PSNR / SSIM (转为灰度计算)
            gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
            enh_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            cur_psnr = psnr(gt_gray, enh_gray)
            cur_ssim = ssim(gt_gray, enh_gray)
            
            # 2. LPIPS (需要 RGB Tensor, 归一化到 [-1, 1])
            gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            enh_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            t_gt = lpips.im2tensor(gt_rgb).cuda()
            t_enh = lpips.im2tensor(enh_rgb).cuda()
            with torch.no_grad():
                cur_lpips = lpips_fn(t_gt, t_enh).item()
                
            # 3. 图像特征熵 (基于增强后的灰度图)
            hist = cv2.calcHist([enh_gray], [0], None, [256], [0, 256]).ravel()
            hist = hist / hist.sum() # 归一化为概率分布
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            
            # 写入日志
            log_file.write(f"{tstamp_str}, {cur_psnr:.4f}, {cur_ssim:.4f}, {cur_lpips:.4f}, {entropy:.4f}\n")
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

        intrinsics[2] -= 16
        intrinsics[3] -= 8
        image = image[:, 8:-8, 16:-16]

        data_list.append((t, image[None], intrinsics))

    return data_list, total_enhance_time

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
    # =========================================

    # === [新增] 初始化 LPIPS 和 日志文件 ===
    print("Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    
    metrics_log = open("enhancement_metrics.txt", "w")
    metrics_log.write("Timestamp, PSNR, SSIM, LPIPS, Entropy\n")

    open("droid_confidence_ratio.txt", "w").close()
    # =====================================

    droid = DroidAsync(args) if args.asynchronous else Droid(args)
    scene = Path(args.datapath).name

    tstamps = []
    images, tot_enh_time = image_stream(args.datapath, dce_net=dce_net, lpips_fn=lpips_fn, log_file=metrics_log)

    metrics_log.close()

    total_track_time = 0.0
    num_frames = len(images)

    for (t, image, intrinsics) in tqdm(images, desc=scene):
        if not args.disable_vis:
            show_image(image)
        torch.cuda.synchronize()
        t_trk_start = time.perf_counter()
        
        droid.track(t, image, intrinsics=intrinsics)
        
        torch.cuda.synchronize()
        t_trk_end = time.perf_counter()
        total_track_time += (t_trk_end - t_trk_start)

    torch.cuda.empty_cache()

    # ========================================================
    # [新增] 计算耗时均值并输出 Efficiency Analysis
    # ========================================================
    avg_enh_ms = (tot_enh_time / num_frames) * 1000.0 if num_frames > 0 else 0.0
    avg_trk_ms = (total_track_time / num_frames) * 1000.0 if num_frames > 0 else 0.0
    
    # 系统的整体吞吐时间 = 增强时间 + 追踪时间
    total_pipeline_time = tot_enh_time + total_track_time
    overall_sys_fps = num_frames / total_pipeline_time if total_pipeline_time > 0 else 0.0

    print("\n" + "="*50)
    print(f"[*] 序列 {scene} (Zero-DCE Variant) 运行效率分析:")
    print(f"  >>> 纯增强网络耗时 (Zero-DCE Enhancement): {avg_enh_ms:.2f} ms/frame")
    print(f"  >>> 纯前端跟踪耗时 (DROID Tracking)      : {avg_trk_ms:.2f} ms/frame")
    print(f"  >>> 系统端到端总帧率 (System FPS)        : {overall_sys_fps:.2f} Hz")
    print("="*50 + "\n")
    
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
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    quat_est_wxyz = traj_est[:, [6, 3, 4, 5]]

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=quat_est_wxyz,
        timestamps=np.array(tstamps))

    os.makedirs("tum_trajectories", exist_ok=True)
    save_path = os.path.join("tum_trajectories", f"{scene}.txt")
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

    # ------------------------------------------------
    # 4. 计算图像增强与前端跟踪指标的平均值并打印
    # ------------------------------------------------
    print("\n" + "="*50)
    print(">>> 4. 图像增强质量指标 (Sequence Averages)")
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
    print(">>> 5. 前端跟踪稳定性 (等效内点率 / Confidence Ratio)")
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