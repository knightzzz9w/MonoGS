import rospy
from std_msgs.msg import Header
import numpy as np
from collections import defaultdict
from sensor_msgs.msg import Image
from utils.convert_stamp import stamp2seconds , TicToc
import torch
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge, CvBridgeError
from utils.math_utils import inverse_Tmatrix
from gui import gui_utils, slam_gui
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
import cv2
import time
from sklearn.neighbors import NearestNeighbors
from munch import munchify
import open3d as o3d
from utils.slam_backend import BackEnd
from utils.multiprocessing_utils import FakeQueue
import torch.multiprocessing as mp
from utils.camera_utils import Camera
from utils.multiprocessing_utils import clone_obj
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
import math
from utils.slam_utils import get_loss_tracking, get_median_depth

class Estimator:
    
    def __init__(self , config):

        self.config = config
        self.device = "cuda"

        T_matrix = np.array(config["T_BS"]["data"]).reshape(4,4)
        self.ric = T_matrix[ :3 ,  :3 ]
        self.tic = T_matrix[ :3 ,  3 ]
        self.Tic = T_matrix
        
        self.deltatime = 1e-2
        
        self.width = config["resolution"][0]
        self.height = config["resolution"][1]
        
        self.K = np.array([[config["intrinsics"][0], 0, config["intrinsics"][2]],
                           [0, config["intrinsics"][1], config["intrinsics"][3]],
                           [0, 0, 1]])
        #param in gaussian
        self.fx = self.K[0 , 0]
        self.fy = self.K[1 , 1]
        self.cx = self.K[0 , 2]
        self.cy = self.K[1 , 2]
        
        self.fovx = 2*math.atan(self.width/(2*self.fx)) ; self.fovy =  2*math.atan(self.height/(2*self.fy))
        
        self.projection_matrix = getProjectionMatrix2(
                znear=0.01,
                zfar=100.0,
                fx=self.K[0 , 0],
                fy=self.K[1 , 1],
                cx=self.K[0 , 2],
                cy=self.K[1 , 2],
                W=self.width,
                H=self.height,
            ).transpose(0, 1).to(self.device)  #反过来的
        
        self.dist_coeffs = np.array(config["distortion_coefficients"])
        
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
            
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        
        self.kf_indices = []
        self.cameras = dict()
        self.dtype = torch.float32
        self.reset = True
        
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()
        
    
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()
        self.config["Training"]["monocular"] = self.monocular
        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        self.initialized = False
        
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main

        self.backend = BackEnd(self.config)

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.q_main2vis = q_main2vis
        self.backend.q_vis2main = q_vis2main
        self.backend.live_mode = True

        self.backend.set_hyperparams()
        
        self.frontend_queue = frontend_queue
        self.backend_queue = backend_queue  # 与上面的保持一样
        self.requested_init = False
        self.requested_keyframe = 0

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        
        self.frame_id = 0
        self.gs_keyframe_id = 0
        
        
        
    def dataalign(self , image_buf , keypose_buf , key_point_cloud_buf): #align the data k1 k2 p1 p2 k3
        
        measurements = []
        
        while(True):
            if(image_buf.empty() or keypose_buf.empty() or key_point_cloud_buf.empty()):
                return measurements
            
            # if(stamp2seconds(image_buf.queue[-1]) < stamp2seconds(keypose_buf.queue[0])  or  stamp2seconds(key_point_cloud_buf.queue[-1]) < stamp2seconds(keypose_buf.queue[0])):  #smaller than the first keypose
            #     return measurements
            
            # if(stamp2seconds(keypose_buf.queue[0]) < stamp2seconds(image_buf.queue[0])  and  stamp2seconds(keypose_buf.queue[0]) < stamp2seconds(key_point_cloud_buf.queue[0])):
            #     keypose_buf.get()
            #     continue
                
            imagetime_first =  stamp2seconds(image_buf.queue[0]) ; keyposetime_first = stamp2seconds(keypose_buf.queue[0]) ; keypointcloudtime_first = stamp2seconds(key_point_cloud_buf.queue[0])
            time_array = np.array([imagetime_first , keyposetime_first , keypointcloudtime_first])
            id_first = np.argmax(time_array)
            time_first = np.max(time_array)

            if id_first != 0:
                while (not image_buf.empty()) and  (time_first   - stamp2seconds(image_buf.queue[0]) > self.deltatime):  #pop the data before it
                    image_buf.get()
            if id_first != 1:
                while  (not keypose_buf.empty()) and  (time_first   - stamp2seconds(keypose_buf.queue[0]) > self.deltatime):  #pop the data before it
                    keypose_buf.get()
            if id_first != 2:
                while    (not key_point_cloud_buf.empty()) and  (time_first   - stamp2seconds(key_point_cloud_buf.queue[0]) > self.deltatime):  #pop the data before it
                    key_point_cloud_buf.get()

            if(image_buf.empty() or keypose_buf.empty() or key_point_cloud_buf.empty()):
                return measurements
            
            cur_image = image_buf.get()  ; cur_keypose = keypose_buf.get() ; cur_key_point_cloud = key_point_cloud_buf.get()
            
            time_cur_image = stamp2seconds(cur_image) ; time_cur_keypose = stamp2seconds(cur_keypose) ; time_cur_key_point_cloud = stamp2seconds(cur_key_point_cloud)
            #print("image time is " , time_cur_image , "keypose time is " , time_cur_keypose , "key_point_cloud time is " , time_cur_key_point_cloud)
            assert(abs(time_cur_image - time_cur_keypose) < self.deltatime) ; assert(abs(time_cur_image - time_cur_key_point_cloud) < self.deltatime)
            measurements.append([cur_image , cur_keypose , cur_key_point_cloud])
            
        return measurements
    
    
    def convert_ros_image_to_cv2(self, ros_image):
        try:
            cv_image = CvBridge().imgmsg_to_cv2(ros_image, "bgr8")
            return cv_image
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return None
        
    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()
        
    def getpicdepth(self ,  points_2D , points_3D_cam ):
        
        # 初始化深度图
        depth_map = np.full((self.height, self.width), np.nan)

        # 提取深度值
        depths = points_3D_cam[:, 2]

        # 将已知深度值填入深度图
        for idx, (x, y) in enumerate(points_2D):
            depth_map[y,x] = depths[idx]

        # 构建KNN模型
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(points_2D)

        # 创建一个新的深度图以存储结果
        new_depth_map = np.copy(depth_map)

        # 遍历所有像素点
        for i in range(self.height):
            for j in range(self.width):
                if np.isnan(new_depth_map[i, j]):  # 如果点没有初始深度信息
                    # 获取半径为3的邻域
                    top = max(0, i-4) ; bot = min(self.height, i+5) ; left = max(0, j-4) ; right = min(self.width, j+5)
                    neighborhood = new_depth_map[top:bot, left:right]
                    if np.all(np.isnan(neighborhood)):  #如果这个点没有被填充
                        # 使用KNN计算该点的深度值
                        distances, indices = knn.kneighbors([[j, i]])
                        neighbor_depths = depths[indices[0]]
                        weights = 1 / (distances[0] + 1e-8)  # 防止除零
                        weighted_sum = np.sum(weights * neighbor_depths)
                        sum_weights = np.sum(weights)
                        new_depth_map[top:bot, left:right] = 0  # 标记为这个区域已填充 但是这些点是没有深度的 只有一个地方有深度
                        new_depth_map[i, j] = weighted_sum / sum_weights

        # 使用 cv2.inpaint 修复深度图中的空缺点
        mask = np.isnan(new_depth_map).astype(np.uint8)
        new_depth_map[np.isnan(new_depth_map)] = 0  # 将 NaN 替换为 0
        #new_depth_map_filled = cv2.inpaint(new_depth_map.astype(np.float32), mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        u = u.flatten()
        v = v.flatten()
        homogeneous_pixel_coords = np.vstack((u, v, np.ones_like(u)))
        
        depths = new_depth_map.flatten()
        valid_mask = depths > 0
        valid_pixel_coords = homogeneous_pixel_coords[:, valid_mask]
        valid_depths = depths[valid_mask]
        
        
        cam_coords = np.linalg.inv(self.K) @ (valid_pixel_coords * valid_depths)
        new_points_2D = valid_pixel_coords[:2].T
        new_points_3D_cam = cam_coords.T

        return new_points_2D , new_points_3D_cam

    def getnewpoints(self ,  points_2D , points_3D_cam ):

        depth_map = np.full((self.height, self.width), np.nan)

        depths = points_3D_cam[:, 2]
        depth_median = np.mean(depths)  #mean depth 
        self.last_depth = depth_median
        depth_std = 0.4
        
        # 将已知深度值填入深度图
        x_coords = points_2D[:, 0]
        y_coords = points_2D[:, 1]
        depth_map[y_coords, x_coords] = depths
        
        
        
        
        invalid_depth_mask = np.isnan(depth_map)
        depth_map[invalid_depth_mask] = depth_median
        depth_map += np.random.standard_normal(depth_map.shape)*np.where(invalid_depth_mask , depth_std*0.5 , depth_std*0.1 )
        u, v = np.meshgrid(np.arange(0 , self.width,7), np.arange(0 , self.height , 7))
        u = u.flatten() ; v = v.flatten()
        
        homogeneous_pixel_coords = np.vstack((u, v, np.ones_like(u)))
        filterdepth_map = depth_map[::7 , ::7]
        filterdepth_map = filterdepth_map.flatten()
        
        cam_coords = np.linalg.inv(self.K) @ (homogeneous_pixel_coords * filterdepth_map)
        new_points_3D_cam = cam_coords.T
        
        print("new points sample after downsampling is " , new_points_3D_cam.shape)
        return homogeneous_pixel_coords[:2 , :].T , new_points_3D_cam
    
    
    
    def getallnewpoints(self,last_median_depth):

        depth_map = np.full((self.height, self.width), last_median_depth)

        depth_std = 0.6
        depth_map[: , :] += np.random.standard_normal(depth_map.shape)*depth_std*0.5
        u, v = np.meshgrid(np.arange(0 , self.width,7), np.arange(0 , self.height , 7))
        u = u.flatten() ; v = v.flatten()
        
        homogeneous_pixel_coords = np.vstack((u, v, np.ones_like(u)))
        filterdepth_map = depth_map[::7 , ::7]
        filterdepth_map = filterdepth_map.flatten()
        
        cam_coords = np.linalg.inv(self.K) @ (homogeneous_pixel_coords * filterdepth_map)
        new_points_3D_cam = cam_coords.T
        
        print("new points sample after downsampling is " , new_points_3D_cam.shape)
        return homogeneous_pixel_coords[:2 , :].T , new_points_3D_cam




    def pub_render(self , tensor_image):
        
        array = tensor_image.cpu().detach().numpy()
        # 由于sensor_msgs/Image期望图像数据为uint8，因此需要转换类型
        array = (array * 255).astype(np.uint8)

        # 如果tensor是3D的，将其转换为4D (H, W, C)
        if len(array.shape) == 3:
            array = array.transpose(1, 2, 0)
        array = array[:, :, ::-1]
        bridge = CvBridge()
        ros_image = bridge.cv2_to_imgmsg(array, encoding="passthrough")
        self.render_pub.publish(ros_image)
        
        
    def get_median_depth(self , depth, opacity=None, mask=None, return_std=False):
        depth = depth.detach().clone()
        opacity = opacity.detach()
        valid = depth > 0
        if opacity is not None:
            valid = torch.logical_and(valid, opacity > 0.95)
        if mask is not None:
            valid = torch.logical_and(valid, mask)
        valid_depth = depth[valid]
        if return_std:
            return valid_depth.median(), valid_depth.std(), valid
        return valid_depth.median()



    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap ):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        print("frontend put keyframe msg")
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        print("frontend put map msg")
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map ):
        msg = ["init", cur_frame_idx, viewpoint, depth_map ]
        print("frontend put init msg")
        self.backend_queue.put(msg)
        self.requested_init = True
        
        
    def sync_backend(self, data):   #keyframe and gaussians
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())  #没有变过 RT 

    
    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = self.get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = self.get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()
        

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check


    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        #viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)  这一帧不用了

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map )
        self.reset = False
        


    def tracking(self, cur_frame_idx, viewpoint):
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )  #recover the details in the image

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )

        self.median_depth = self.get_median_depth(depth, opacity)
        return render_pkg
        
    


    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame
    
    def project_and_get_colors(self, measurements):
    
        while True:
            if len(measurements) == 0:
                return 
            
            if self.frontend_queue.empty():

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                [cur_image, cur_keypose, cur_key_point_cloud] = measurements.pop(0)

                
                tracking_time = TicToc()
                tracking_time.tic()

                image = self.convert_ros_image_to_cv2(cur_image)  #图像去掉畸变
                #print(image.shape)
                image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)  #shape is [height , width , 3]  color is bgr
                image = image.transpose(2,0,1) ; image = image[::-1 , : , :] ; image = image.astype(float)/255  ;  image = torch.from_numpy(image).clamp(0.0 , 1.0).to(self.device , dtype = self.dtype)
                
                position = cur_keypose.pose.pose.position  #R_wbk
                orientation = cur_keypose.pose.pose.orientation   #T_wbk
                
                

                translation = np.array([position.x, position.y, position.z])
                quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
                rotation_matrix = R.from_quat(quaternion).as_matrix()
                
                T_Wbk = np.eye(4)
                T_Wbk[ :3 ,  :3 ] = rotation_matrix ; T_Wbk[ :3 , 3 ] = translation
                
                T_Wck = torch.from_numpy((T_Wbk@self.Tic).astype(np.float)).to("cuda")
                
                T_ckW = inverse_Tmatrix(T_Wck)


                points_3D_ext = torch.tensor([[point.x, point.y, point.z , 1] for point in cur_key_point_cloud.points]) 
                
                depth_matrix = None
                depth_mask = None
                

                if points_3D_ext.shape[0] > 0:
                    # depth_matrix = torch.zeros((1 , self.height , self.width)).cuda()
                    # depth_mask = torch.zeros_like(depth_matrix, dtype=torch.bool).cuda()
                    points_3D_ext = points_3D_ext.to("cuda")

                    points_3D_ext_cam = points_3D_ext@T_ckW.T
                    points_3D_cam = points_3D_ext_cam[: , :3]  #得到在当前相机坐标系下表示的点云 N*3
                    points_3D_cam = points_3D_cam[points_3D_cam[: , 2] >= 0.01]

                    point_depth = torch.norm(points_3D_cam , dim = -1)
                    point_depth2 = points_3D_cam[: , 2]  #Z depth


                    points_2D_cam = points_3D_cam@(torch.from_numpy(self.K.T).to("cuda")).to(torch.float) 
                    points_2D = points_2D_cam[:, :2] / (points_2D_cam[:, 2].reshape(-1, 1))  # Nx2
                    points_2D = points_2D.int()
                    # points_2D = torch.clamp(points_2D , min = torch.tensor([0,0]).to("cuda") , max = torch.tensor([self.width-1 , self.height-1]).to("cuda"))


                    # # Populate gt_depth_matrix and depth_mask
                    # depth_matrix[0, points_2D[:, 1], points_2D[:, 0]] = point_depth2
                    # depth_mask[0, points_2D[:, 1], points_2D[:, 0]] = True


                viewpoint = Camera(
                    uid = self.frame_id , color = image, depth = None, gt_T = T_Wck  , projection_matrix=self.projection_matrix,
                    fx = self.fx , fy = self.fy , cx = self.cx , cy = self.cy, fovx = self.fovx , fovy = self.fovy , image_height = self.height,
                    image_width = self.width,   trian_depthmap=depth_matrix , trian_depthmask=depth_mask   , T = T_Wck  , device = self.device
                )
                viewpoint.compute_grad_mask(self.config)
                self.cameras[self.frame_id] = viewpoint


                print("Frontend !!! Prepare viewpoint and 3D point cost " , tracking_time.toc()  ,"ms")

                if self.reset:
                    init_time = TicToc()
                    init_time.tic()
                    self.initialize(self.frame_id, viewpoint)
                    self.current_window.append(self.frame_id)
                    self.frame_id += 1   #get init unsync
                    continue  #这次循环加一


                self.initialized = self.initialized or (len(self.current_window) == self.window_size)

                render_pkg = self.tracking(self.frame_id , viewpoint)
                 
                print("Frontend !!! Tracking and refining exposure cost " , tracking_time.toc()  ,"ms")


                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                
                
                
                self.q_main2vis.put(  #回传了结果 可以往下进行
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
            
                print("Frontend !!! Push to UI cost " , tracking_time.toc()  ,"ms")

                if self.requested_keyframe > 0:    #wait for the backend
                    self.cleanup(self.frame_id)
                    self.frame_id += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (self.frame_id - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    self.frame_id,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf


                print("Frontend !!! Judge Kf cost " , tracking_time.toc()  ,"ms")
                
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        self.frame_id,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        print(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    
                    depth_map = self.add_new_keyframe(
                        self.frame_id,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(  #need gaussian point to get depth loss
                        self.frame_id, viewpoint, self.current_window, depth_map
                    )
                    
                else:
                    self.cleanup(self.frame_id)
                

                print("Frontend !!! Create Kf cost " , tracking_time.toc()  ,"ms")
                print("current window is " , [cur_win for cur_win in self.current_window])
                self.frame_id += 1

                # if (
                #     self.save_results
                #     and self.save_trj
                #     and create_kf
                #     and len(self.kf_indices) % self.save_trj_kf_intv == 0
                # ):
                #     Log("Evaluating ATE at frame: ", cur_frame_idx)
                #     eval_ate(
                #         self.cameras,
                #         self.kf_indices,
                #         self.save_dir,
                #         cur_frame_idx,
                #         monocular=self.monocular,
                #     )
                # toc.record()
                torch.cuda.synchronize()
                # if create_kf:
                #     # throttle at 3fps when keyframe is added
                #     duration = tic.elapsed_time(toc)
                #     time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))

            
            
            
                
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    print("receive data from backend , and requested_keyframe is "  , self.requested_keyframe)
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    print("receive init information from backend ")
                    self.sync_backend(data)
                    self.requested_init = False


          
        #     # 假设 cur_key_point_cloud 是 Nx3 的点云数组
        #     points_3D_ext = np.array([[point.x, point.y, point.z , 1] for point in cur_key_point_cloud.points])  # Nx4  扩展的
        #     print("cur point size is %d"  , points_3D_ext.shape[0])
            
        #     t_pre1 = time.time()
        #     # 从cur_keypose获取旋转和平移矩阵
        #     position = cur_keypose.pose.pose.position  #R_wbk
        #     orientation = cur_keypose.pose.pose.orientation   #T_wbk
            
            

        #     translation = np.array([position.x, position.y, position.z])
        #     quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        #     rotation_matrix = R.from_quat(quaternion).as_matrix()
            
        #     T_Wbk = np.eye(4)
        #     T_Wbk[ :3 ,  :3 ] = rotation_matrix ; T_Wbk[ :3 , 3 ] = translation
            
        #     T_Wck = T_Wbk@self.Tic
            
        #     T_ckW = inverse_Tmatrix(T_Wck)
        #     assert( np.sum(np.abs((T_ckW@T_Wck - np.eye(4)))) < 1e-3  )
        #     if points_3D_ext.shape[0] > 0:                
                
        #         t_pre_part1_1 = time.time()
                
        #         points_3D_ext_cam = points_3D_ext@T_ckW.T
        #         points_3D_cam = points_3D_ext_cam[: , :3]  #得到在当前相机坐标系下表示的点云
                
                
        #         points_2D_cam = points_3D_cam@self.K.T #还要必上Z
        #         points_2D = points_2D_cam[:, :2] / (points_2D_cam[:, 2].reshape(-1, 1))  # Nx2
        #         print("num of point is " , points_2D.shape[0])
                
        #         t_pre_part1_2 = time.time()
        #         print("perpare 2D points time  is " , (t_pre_part1_2 - t_pre_part1_1)*1e3, " ms")
                
        #         #print(points_2D)
                
        #         t_pre_part2_1 = time.time()
        #         points_2D = np.round(points_2D).astype(int)
        #         filtered_ids = (
        #             (points_2D[:, 0] >= 0) & (points_2D[:, 0] < self.width) &
        #             (points_2D[:, 1] >= 0) & (points_2D[:, 1] < self.height)
        #         )
        #         points_2D = points_2D[filtered_ids] ; points_3D_cam = points_3D_cam[filtered_ids] ; points_3D_ext = points_3D_ext[filtered_ids]
        #         print("num of point in screen is " , points_2D.shape[0])
        #         #self.getpicdepth(points_2D , points_3D_cam )  时间太慢了 去掉
        #         t_pre_part2_2 = time.time()
        #         print("perpare legal 2D points time  is " , (t_pre_part2_2 - t_pre_part2_1)*1e3, " ms")
                
        #         t_addpoints = time.time()
        #         new_point2D , new_point3D_cam = self.getnewpoints(points_2D , points_3D_cam)  #new point in cam cor
        #         t_addpoints2 = time.time()
        #         print("add points time is " , (t_addpoints2 - t_addpoints)*1e3 , "ms")
            
            
        #     else:
        #         new_point2D , new_point3D_cam = self.getallnewpoints(self.last_depth)
            
            
        #     t_pre_part3_1 = time.time()
        #     image = self.convert_ros_image_to_cv2(cur_image)  #图像去掉畸变
        #     #print(image.shape)
        #     image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)  #shape is [height , width , 3]  color is bgr
        #     # colors = []
        #     # for point in points_2D:
        #     #     x, y = point[0], point[1] #分别代表横向的和纵向的
        #     #     cur_color = image[y, x]
        #     #     rgb_colors = [float(cur_color[2]/255) , float(cur_color[1]/255), float(cur_color[0]/255)]
        #     #     colors.append(rgb_colors)
            
        #     x_coords = new_point2D[:, 0]
        #     y_coords = new_point2D[:, 1]
        #     selected_colors = image[y_coords, x_coords]
        #     normalized_colors = selected_colors.astype(float) / 255.0

        #     # 将颜色顺序从 BGR 转换为 RGB
        #     normalized_colors = normalized_colors[:, ::-1].copy()
        #     print(normalized_colors.shape)
            
        #     t_pre_part3_2 = time.time()
        #     print("perpare 2D points color time  is " , (t_pre_part3_2 - t_pre_part3_1)*1e3, " ms")
            
            
        #     t_pre2 = time.time()
        #     print("perpare time is " , (t_pre2 - t_pre1)*1e3, " ms")
            
            
            
            
            
        #     cam_model = loadCam(self.frame_id , image , T_Wck[:3 , :3] ,T_Wck[:3 , 3] , self.K  )
            
        #     t_add_gaussian = time.time()
        #     R_Wck = T_Wck[:3 , :3] ; t_Wck = T_Wck[:3 , 3]
        #     points_3D = new_point3D_cam@R_Wck.T + t_Wck
        #     self.gaussians.extend_from_pcd_seq(cam_info = cam_model , point_3D = points_3D  , colors = normalized_colors , point_size = self.point_size , kf_id = self.frame_id)
        #     t_add_gaussian2 = time.time()
        #     print("add gaussian time is " , (t_add_gaussian2 - t_add_gaussian)*1e3 , "ms" )
            
            
            
        #     t_render = time.time()
        #     for i in range(20):
        #         render_pkg = render(
        #             cam_model, self.gaussians, self.pipeline_params, torch.tensor(self.background).to("cuda")
        #         )
        #         image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]            
                
        #         gt_image = cam_model.original_image.cuda()
        #         Ll1 = l1_loss(image, gt_image)
        #         loss = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
        #         loss.backward()
        #         #iter_end.record()
                
        #         self.gaussians.optimizer.step()
        #         self.gaussians.optimizer.zero_grad(set_to_none=True)
                
                
                
                
        #         with torch.no_grad():
        #             if i % self.update_freq == self.update_offset:
        #                 self.gaussians.densify_and_prune(
        #                 self.opt_params.densify_grad_threshold,
        #                 self.gaussian_th,
        #                 self.gaussian_extent,
        #                 self.size_threshold,
        #                     )
                    
        #     t_render2 = time.time()
        #     print("rendering time is " , (t_render2 - t_render)*1e3 , "ms" )
        #     self.pub_render(image)
        #     print("cur gaussian num is " , (self.gaussians._xyz.shape))

            
            

            
    
    
    def get_measurements(self , image_buf , keypose_buf , key_point_cloud_buf):
        return self.dataalign( image_buf , keypose_buf , key_point_cloud_buf)
    
    
    
    
            
        


