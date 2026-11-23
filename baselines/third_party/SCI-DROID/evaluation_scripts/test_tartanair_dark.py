import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')

from evaluation.tartanair_evaluator import TartanAirEvaluator

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import yaml
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

from droid import Droid
from droid_async import DroidAsync

# camera baseline hardcoded to 0.1m
STEREO_SCALE_FACTOR = 2.5

MONO_TEST_SCENES = [f"M{s}{i:03d}" for s in ["E", "H"] for i in range(8)]
STEREO_TEST_SCENES = [f"S{s}{i:03d}" for s in ["E", "H"] for i in range(8)]

def simulate_low_light(image_np, low_light_factor=0.1, noise_variance=0.015):
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
# ==========================================


def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[320.0, 320.0, 320.0, 240.0], stereo=False, sci_net=None):
    """ image generator """

    # read all png images in folder
    ht0, wd0 = [480, 640]

    if stereo:
        images_left = sorted(glob.glob(os.path.join(datapath, 'image_left/*.png')))
        images_right = sorted(glob.glob(os.path.join(datapath, 'image_right/*.png')))

    else:
        if os.path.exists(os.path.join(datapath, "image_left")):
            images_left = sorted(glob.glob(os.path.join(datapath, 'image_left/*.png')))
        else:
            images_left = sorted(glob.glob(os.path.join(datapath, '*.png')))

    data = []
    for t in range(len(images_left)):
        images = [ cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])) ]
        if stereo:
            images += [ cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0])) ]

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
                
                # 4. 转回 BGR 给 SLAM 使用
                enhanced_img = cv2.cvtColor(enhanced_numpy, cv2.COLOR_RGB2BGR)
                enhanced_images.append(enhanced_img)
            
            images = enhanced_images
        # ==========================================

        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)
        intrinsics = .8 * torch.as_tensor(intrinsics_vec)

        data.append((t, images, intrinsics))

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--gt_path")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--scene", type=str)

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.5)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.0)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    # damped linear velocity model
    parser.add_argument("--motion_damping", type=int, default=0.5)

    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")

    parser.add_argument("--sci_weights", type=str, default="weights_1_3500.pt", help="SCI weights path")

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    if not os.path.isdir("figures"):
        os.mkdir("figures")

    test_scenes = STEREO_TEST_SCENES if args.stereo else MONO_TEST_SCENES

    if args.scene is not None:
        test_scenes = [args.scene]

    # 用于动态收集所有评价指标
    all_metrics = defaultdict(list) 
    
    for scene in test_scenes:
        print("\n" + "="*60)
        print("Performing evaluation on {}".format(scene))
        print("="*60)
        torch.cuda.empty_cache()

        # ==========================================
        # [新增] 实例化并加载 SCI 网络
        # ==========================================
        print(f"Loading SCI model from {args.sci_weights} ...")
        sci_net = Finetunemodel(args.sci_weights).cuda()
        sci_net.eval()
        # ==========================================

        droid = DroidAsync(args) if args.asynchronous else Droid(args)

        scenedir = os.path.join(args.datapath, scene)
        gt_file = os.path.join(args.gt_path, f"{scene}.txt")

        images_data = image_stream(scenedir, stereo=args.stereo, sci_net=sci_net)

        for (tstamp, image, intrinsics) in tqdm(images_data, desc=scene):
            droid.track(tstamp, image, intrinsics=intrinsics)

        torch.cuda.empty_cache()

        # fill in non-keyframe poses + global BA
        traj_est = droid.terminate(images_data)

        os.makedirs("tartanair_trajectories", exist_ok=True)
        seq_name = scene.replace('/', '_')
        save_path = os.path.join("tartanair_trajectories", f"{seq_name}.txt")
        
        fake_timestamps_est = np.arange(len(traj_est)).reshape(-1, 1).astype(np.float64)
        
        if traj_est.shape[1] == 7:
            tum_traj = np.hstack((fake_timestamps_est, traj_est))
        elif traj_est.shape[1] == 8:
            # 如果已经有8列，强制把第一列替换为对齐的假时间戳
            traj_est[:, 0] = fake_timestamps_est.flatten()
            tum_traj = traj_est
            
        np.savetxt(save_path, tum_traj, fmt='%.6f', delimiter=' ')

        if args.stereo:
            traj_est[:, :3] *= STEREO_SCALE_FACTOR

        # 加载真值并把 ned 转换为 xyz (x, y, z, qx, qy, qz, qw)
        traj_ref = np.loadtxt(gt_file, delimiter=' ')[:, [1, 2, 0, 4, 5, 3, 6]] 

        gt_save_path = os.path.join("tartanair_trajectories", f"{seq_name}_gt_aligned.txt")
        fake_timestamps_gt = np.arange(len(traj_ref)).reshape(-1, 1).astype(np.float64)
        
        # 拼接时间戳和转换坐标系后的真值 (共 8 列)
        tum_traj_gt = np.hstack((fake_timestamps_gt, traj_ref[:, :7]))
        np.savetxt(gt_save_path, tum_traj_gt, fmt='%.6f', delimiter=' ')

        # ========================================================
        # 1. TartanAir 原版测评逻辑
        # ========================================================
        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(
            traj_ref, traj_est, scale=not args.stereo, title=scenedir[-20:].replace('/', '_'))
        
        print(f"\n[{scene}] TartanAir 原始评估结果:")
        for metric_name, value in results.items():
            try:
                print(f"  - {metric_name}: {value:.4f}")
            except (TypeError, ValueError):
                pass
            all_metrics[metric_name].append(value)


        # ========================================================
        # 2. Evo 库完整评估逻辑 (APE & RPE)
        # ========================================================
        print(f"\n[{scene}] Evo 专业评估结果:")
        
        # 关键修正：evo要求四元数格式为 [w, x, y, z]，而 numpy 里的是 [x, y, z, w]
        # w在第6列(索引)，x, y, z 分别在 3, 4, 5 列
        quat_est_wxyz = traj_est[:, [6, 3, 4, 5]]
        quat_ref_wxyz = traj_ref[:, [6, 3, 4, 5]]

        # 构建 Evo 专用的 Trajectory 对象
        traj_est_evo = PoseTrajectory3D(
            positions_xyz=traj_est[:, :3],
            orientations_quat_wxyz=quat_est_wxyz,
            timestamps=np.arange(len(traj_est), dtype=np.float64)
        )
        
        traj_ref_evo = PoseTrajectory3D(
            positions_xyz=traj_ref[:, :3],
            orientations_quat_wxyz=quat_ref_wxyz,
            timestamps=np.arange(len(traj_ref), dtype=np.float64)
        )

        # 时间戳对齐 (截齐长短不一的序列)
        traj_ref_evo, traj_est_evo = sync.associate_trajectories(traj_ref_evo, traj_est_evo)

        # ========================================================
        # 3. 计算 Hessian 条件数与 NEES 并保存 (论文 D 小节专用)
        # ========================================================
        print(f"\n[{scene}] 计算 Hessian Condition Number & NEES 动力学指标...")
        
        cond_list = []
        nees_list = []
        
        # 【关键修复】从 evo 轨迹中提取位置 (Nx3) 和 四元数 (Nx4)
        pos_est = traj_est_evo.positions_xyz
        pos_ref = traj_ref_evo.positions_xyz
        
        # evo 的四元数格式是 wxyz，而 lietorch 严格要求 xyzw 格式，需要做一次列调换
        quat_est_wxyz = traj_est_evo.orientations_quat_wxyz
        quat_ref_wxyz = traj_ref_evo.orientations_quat_wxyz
        
        quat_est_xyzw = np.concatenate([quat_est_wxyz[:, 1:], quat_est_wxyz[:, 0:1]], axis=1)
        quat_ref_xyzw = np.concatenate([quat_ref_wxyz[:, 1:], quat_ref_wxyz[:, 0:1]], axis=1)
        
        # 拼接成 Nx7 的标准格式: [x, y, z, qx, qy, qz, qw]
        vec_est = np.concatenate([pos_est, quat_est_xyzw], axis=1)
        vec_ref = np.concatenate([pos_ref, quat_ref_xyzw], axis=1)
        
        window_size = 8 # 模拟 DROID 后端 BA 的局部滑动窗口大小
        
        # 以滑动窗口的方式遍历轨迹
        for i in range(0, len(vec_est) - window_size, 2):
            window_est = vec_est[i : i+window_size]
            window_ref = vec_ref[i : i+window_size]
            
            H_approx = np.zeros((window_size * 6, window_size * 6))
            error_vec = np.zeros((window_size * 6, 1))
            
            for j in range(window_size):
                # 此时 window_est[j] 是严格的长度为 7 的向量，lietorch C++ 后端安全解析
                T_est = lietorch.SE3(torch.tensor(window_est[j]).unsqueeze(0).float())
                T_ref = lietorch.SE3(torch.tensor(window_ref[j]).unsqueeze(0).float())
                
                # 在李代数切空间上计算误差: e = log(T_ref^-1 * T_est)
                e_se3 = (T_ref.inv() * T_est).log().squeeze().numpy()  # shape: (6,)
                error_vec[j*6 : (j+1)*6, 0] = e_se3
                
                # 构建近似 Hessian (信息矩阵) H ≈ J^T \Sigma^{-1} J
                confidence = 1.0 / (1.0 + np.linalg.norm(e_se3)) 
                H_approx[j*6 : (j+1)*6, j*6 : (j+1)*6] = np.eye(6) * confidence
                
                # 添加 BA 优化中的相邻帧平滑约束（非对角块）
                if j > 0:
                    H_approx[(j-1)*6 : j*6, j*6 : (j+1)*6] = -np.eye(6) * (confidence * 0.4)
                    H_approx[j*6 : (j+1)*6, (j-1)*6 : j*6] = -np.eye(6) * (confidence * 0.4)
            
            # 添加 LM (Levenberg-Marquardt) 阻尼项，防止矩阵完全奇异
            H_damped = H_approx + np.eye(window_size * 6) * 1e-6
            
            # 1. 计算条件数 Condition Number (最大特征值 / 最小特征值)
            cond_val = np.linalg.cond(H_damped)
            cond_list.append(cond_val)
            
            # 2. 计算 NEES (归一化估计误差平方): e^T * H * e
            nees_val = error_vec.T @ H_damped @ error_vec
            nees_list.append(nees_val[0, 0])
            
        # 将当前序列的动力学数据保存到本地
        np.save(os.path.join("tartan_trajectories", f"{seq_name}_cond.npy"), np.array(cond_list))
        np.save(os.path.join("tartan_trajectories", f"{seq_name}_nees.npy"), np.array(nees_list))
        print(f"  >>> 已保存动力学数据至 tartan_trajectories/{seq_name}_cond.npy 等文件")
        
        # 判断是否需要尺度对齐 (单目需要缩放，双目通常不需要)
        do_scale_correction = not args.stereo

        # --- A. 计算 APE (绝对平移误差) ---
        result_ape = main_ape.ape(traj_ref_evo, traj_est_evo, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=do_scale_correction)
        ape_rmse = result_ape.stats["rmse"]
        print(f"  >>> APE (平移) RMSE: {ape_rmse:.4f} m")
        all_metrics["evo_APE_Trans_RMSE"].append(ape_rmse)

        # --- B. 计算 RPE (相对平移误差) Delta = 1 帧 ---
        result_rpe_trans = main_rpe.rpe(traj_ref_evo, traj_est_evo, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=do_scale_correction,
            delta=1, delta_unit=Unit.frames, all_pairs=False)
        rpe_trans_rmse = result_rpe_trans.stats["rmse"]
        print(f"  >>> RPE (平移) RMSE: {rpe_trans_rmse:.4f} m/frame")
        all_metrics["evo_RPE_Trans_RMSE"].append(rpe_trans_rmse)

        # --- C. 计算 RPE (相对旋转误差) Delta = 1 帧 ---
        result_rpe_rot = main_rpe.rpe(traj_ref_evo, traj_est_evo, est_name='traj', 
            pose_relation=PoseRelation.rotation_angle_deg, align=True, correct_scale=do_scale_correction,
            delta=1, delta_unit=Unit.frames, all_pairs=False)
        rpe_rot_rmse = result_rpe_rot.stats["rmse"]
        print(f"  >>> RPE (旋转) RMSE: {rpe_rot_rmse:.4f} deg/frame")
        all_metrics["evo_RPE_Rot_RMSE"].append(rpe_rot_rmse)

        # --- D. 计算 RPE (完整 SE3 误差) Delta = 1 帧 ---
        result_rpe_se3 = main_rpe.rpe(traj_ref_evo, traj_est_evo, est_name='traj', 
            pose_relation=PoseRelation.full_transformation, align=True, correct_scale=do_scale_correction,
            delta=1, delta_unit=Unit.frames, all_pairs=False)
        rpe_se3_rmse = result_rpe_se3.stats["rmse"]
        print(f"  >>> RPE (SE3) RMSE:  {rpe_se3_rmse:.4f} /frame")
        all_metrics["evo_RPE_SE3_RMSE"].append(rpe_se3_rmse)


    # ==========================================
    # 打印最终所有序列的平均结果
    # ==========================================
    print("\n" + "#"*60)
    print("FINAL MEAN RESULTS OVER ALL SCENES")
    print("#"*60)
    
    for metric_name, values in all_metrics.items():
        try:
            mean_val = np.mean(values, axis=0) 
            if isinstance(mean_val, (int, float, np.number)):
                print(f"Mean {metric_name:25}: {mean_val:.4f}")
            else:
                pass # 忽略非数字的输出
        except Exception:
            pass
            
    print("#"*60 + "\n")

    if args.plot_curve and "ate_score" in all_metrics:
        import matplotlib.pyplot as plt
        ate = np.array(all_metrics["ate_score"])
        xs = np.linspace(0.0, 1.0, 512)
        ys = [np.count_nonzero(ate < t) / ate.shape[0] for t in xs]

        plt.plot(xs, ys)
        plt.xlabel("ATE [m]")
        plt.ylabel("% runs")
        plt.title("ATE Cumulative Distribution")
        plt.show()