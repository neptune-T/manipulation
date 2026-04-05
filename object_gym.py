
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
import imageio
import open3d as o3d
import cv2
import math
import numpy as np
import torch
import time
import trimesh as tm
from utils import images_to_video, orientation_error, \
    get_downsampled_pc, get_point_cloud_from_rgbd_GPU
import os, json
import yaml
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("../")
sys.path.append("../vision")
sys.path.append("./gym")
import torch
# import numpy as np
import json
from fast_contact_calc import FastContactCalculator

import trimesh

import plotly.graph_objects as go
import os

# if True: use grounded dino, gsam, sudoai
if False:
    from vision.grounded_sam_demo import prepare_GroundedSAM_for_inference, inference_one_image
    from sudoai import SudoAI
# if True: use curobo, otherwise use ik
if True:
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState, RobotConfig
    from curobo.util.logger import setup_curobo_logger
    from curobo.util_file import (
        get_robot_configs_path,
        get_world_configs_path,
        join_path,
        load_yaml,
        )
    from curobo.geom.types import Mesh, WorldConfig, Cuboid
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.util_file import get_robot_path, join_path, load_yaml

class ObjectGym():
    def __init__(
            self, 
            cfgs,
            grounded_dino_model = None, 
            sam_predictor = None,
            save_root = None
        ):
        self.cfgs = cfgs
        self.debug = cfgs["debug"]
        self.use_cam = cfgs["cam"]["use_cam"]
        self.steps = cfgs["steps"]
        
        # configure env grid
        self.num_envs = cfgs["num_envs"]
        self.num_per_row = int(math.sqrt(self.num_envs))
        self.spacing = cfgs["env_spacing"]
        self.env_lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
        print("Creating %d environments" % self.num_envs)
        
        if self.use_cam:
            self.cam_w = cfgs["cam"]["cam_w"]
            self.cam_h = cfgs["cam"]["cam_h"]
            self.cam_far_plane = cfgs["cam"]["cam_far_plane"]
            self.cam_near_plane = cfgs["cam"]["cam_near_plane"]
            self.horizontal_fov = cfgs["cam"]["cam_horizontal_fov"]
            self.cam_poss = cfgs["cam"]["cam_poss"]
            self.cam_targets = cfgs["cam"]["cam_targets"]
            self.num_cam_per_env = len(self.cam_poss)
            self.point_cloud_bound = cfgs["cam"]["point_cloud_bound"]
            
        # segmentation
        self.franka_seg_id = cfgs["asset"]["franka_seg_id"]
        self.asset_seg_ids = cfgs["asset"]["asset_seg_ids"]
        self.table_seg_id = cfgs["asset"]["table_seg_id"]    
        
        # headless
        self.headless = cfgs["HEADLESS"]
    
        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        
        # Add custom arguments
        self.args = gymutil.parse_arguments(description="Placement",
            custom_parameters=[
                {"name": "--mode", "type": str, "default": ""},
                {"name": "--task_root", "type": str, "default": "gym_outputs_task_gen_ycb_0229"},
                {"name": "--config", "type": str, "default": "config_render_api2"},
                {"name": "--device", "type": str, "default": "cuda"},
                {"name": "--headless", "action": 'store_true', "default": False},
                ]
            )

        # set torch device
        self.device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        
        # unused for now, be careful, otherwise it will cause error
        # sim_params.dt = 1.0 / 60.0 # haoran: 60
        # sim_params.substeps = 2 # default 2
        # assert self.args.physics_engine == gymapi.SIM_PHYSX
        # sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024 # default 1024 * 1024
        # sim_params.physx.max_depenetration_velocity = 1000
        # sim_params.physx.solver_type = 1
        # sim_params.physx.num_position_iterations = 8
        # sim_params.physx.num_velocity_iterations = 1
        # sim_params.physx.rest_offset = 0.0
        # sim_params.physx.contact_offset = 0.02
        # sim_params.physx.friction_offset_threshold = 0.001
        # sim_params.physx.friction_correlation_distance = 0.0005
        # sim_params.physx.num_threads = self.args.num_threads
        # sim_params.physx.use_gpu = self.args.use_gpu
        
        # Grab controller
        self.controller_name = cfgs["controller"]
        assert self.controller_name in {"ik", "osc", "curobo"}, f"Invalid controller specified -- options are (ik, osc). Got: {self.controller_name}"
        
        # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        self.sim_params = sim_params
        self.sim_dt = float(getattr(sim_params, "dt", 1.0 / 60.0))

        # create viewer
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
        
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        # save root
        self.asset_root = self.cfgs["asset"]["asset_root"]
        self.save_root = save_root
        
        # prepare assets
        self.prepare_mano_asset()
        self.prepare_obj_assets()
        if self.cfgs["USE_ARTI"]:
            self.prepare_arti_obj_assets()
        self.load_env(load_cam=self.use_cam)
        
        self.init_observation()
        
        # init run, warm up, not necessary
        self.run_steps(pre_steps = 5)
        self.refresh_observation(get_visual_obs=False)
        
        # some off-the-shelf models
        if self.cfgs["INFERENCE_GSAM"] and grounded_dino_model is None and \
            sam_predictor is None:
            self.prepare_groundedsam()
        else:
            self.grounded_dino_model = grounded_dino_model
            self.sam_predictor = sam_predictor
            self.box_threshold = 0.3
            self.text_threshold = 0.25
        if self.cfgs["USE_CUROBO"]:
            self.prepare_curobo(use_mesh=self.cfgs["USE_MESH_COLLISION"])
        if self.cfgs["USE_GRASPNET"]:
            self.prepare_graspnet() 
        if self.cfgs["USE_SUDOAI"]:
            self.prepare_sudo_ai(self.save_root)
    
    # not used
    def prepare_sudo_ai(self, save_root):
        self.sudoai_api = SudoAI(output_dir=save_root)
        
    # not used
    def inference_sudo_ai(self, img_path):
        meshfile = self.sudoai_api.image_to_3d(img_path, save_root= self.save_root)
        print(f"Mesh file saved to {meshfile}")
        assert os.path.exists(meshfile), "BUG!"
        glb_path = meshfile
        mesh=trimesh.load(glb_path)
        # os.makedirs('')
        mesh.export(f'{self.save_root}/material.obj')
      
    # not used  
    def prepare_graspnet(self):
        
        from gym.test_files.infer_vis_grasp import MyGraspNet
        self.graspnet = MyGraspNet(self.cfgs["graspnet"])
    
    # not used
    def inference_graspnet(self, pcs, keep = 1000):
        gg = self.graspnet.inference(pcs)
        gg = gg.nms()
        gg = gg.sort_by_score()
        if self.cfgs["graspnet"]["vis"]:
            if gg.__len__() > keep:
                gg_vis = gg[:keep]
            else:
                gg_vis = gg
            grippers = gg_vis.to_open3d_geometry_list()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcs.astype(np.float32))
            o3d.visualization.draw_geometries([cloud, *grippers])   
        
        return gg

    # used when controlling with curobo
    def prepare_curobo(self, use_mesh = False):
        setup_curobo_logger("error")
        tensor_args = TensorDeviceType()
        robot_file = "franka.yml"  # 保持上一轮改好的这个文件名
        
        if not use_mesh:
            from curobo.geom.types import WorldConfig, Cuboid
            dummy_floor = Cuboid(name="floor", dims=[5.0, 5.0, 0.1], pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0])
            world_model = WorldConfig(cuboid=[dummy_floor])

            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_file,
                world_model,  
                tensor_args,
                interpolation_dt=0.01,
            )
        else:
            asset_files = self.cfgs["asset"]["asset_files"]
            asset_obj_files = [os.path.join(self.cfgs["asset"]["asset_root"],  "/".join(asset_file.split("/")[:-1]), "textured.obj") for asset_file in self.cfgs["asset"]["asset_files"]]
            # import pdb; pdb.set_trace()
            object_meshes = [tm.load(asset_obj_file) for asset_obj_file in asset_obj_files]
            states = self.root_states[2:, :7].cpu().numpy()
            assert len(states) == len(object_meshes), "BUG!"
            obstables = [
                Mesh(
                    name=f'object_{object_meshes_i}', 
                    pose=states[object_meshes_i],
                    vertices=object_meshes[object_meshes_i].vertices,
                    faces=object_meshes[object_meshes_i].faces
                    ) 
                for object_meshes_i in range(len(object_meshes))
                ]
            
            # import pdb; pdb.set_trace()
            table = Cuboid(
                name='table',
                dims=[self.table_scale[0], self.table_scale[1], self.table_scale[2]],
                # dims=[0, 0, 0],
                pose=[self.table_pose.p.x, self.table_pose.p.y, self.table_pose.p.z, self.table_pose.r.x, self.table_pose.r.y, self.table_pose.r.z, self.table_pose.r.w],
                scale=1.0
            )
            world_model = WorldConfig(
                mesh=obstables,
                cuboid=[table],
            )
            world_model = WorldConfig.create_collision_support_world(world_model)
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_file,
                world_model,
                tensor_args,
                # interpolation_dt=0.1,
                # trajopt_tsteps=8,
                collision_checker_type=CollisionCheckerType.MESH,
                use_cuda_graph=False,
                # num_trajopt_seeds=12,
                # num_graph_seeds=12,
                # interpolation_dt=0.03,
                collision_cache={"obb": 30, "mesh": 10},
                # collision_activation_distance=0.01,
                # acceleration_scale=1.0,
                self_collision_check=True,
                # maximum_trajectory_dt=0.25,
                # fixed_iters_trajopt=True,
                # finetune_dt_scale=1.05,
                # velocity_scale=None,
                # interpolation_type=InterpolateType.CUBIC,
                # use_gradient_descent=True,
                store_debug_in_result=False,
            )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True)
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
        
    # not used
    def prepare_groundedsam(self):
        
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        sam_version = "vit_h"
        sam_checkpoint = "../assets/ckpts/sam_vit_h_4b8939.pth"
        grounded_checkpoint = "../assets/ckpts/groundingdino_swint_ogc.pth"
        config = "../vision/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

        self.grounded_dino_model, self.sam_predictor = prepare_GroundedSAM_for_inference(sam_version=sam_version, sam_checkpoint=sam_checkpoint,
                grounded_checkpoint=grounded_checkpoint, config=config, device=self.device)
        
    # used
    def get_gapartnet_anno(self):
        '''
        Get gapartnet annotation
        '''
        self.gapart_cates = []
        self.gapart_init_bboxes = []
        self.gapart_link_names = []
        self.gapart_raw_valid_annos = []
        for gapartnet_id in self.gapartnet_ids:
            # load object annotation
            annotation_path = f"{self.asset_root}/{self.gapartnet_root}/{gapartnet_id}/link_annotation_gapartnet.json"
            anno = json.loads(open(annotation_path).read())
            num_link_anno = len(anno)
            gapart_raw_valid_anno = []
            for link_i in range(num_link_anno):
                anno_i = anno[link_i]
                if anno_i["is_gapart"]:
                    gapart_raw_valid_anno.append(anno_i)
            self.gapart_raw_valid_annos.append(gapart_raw_valid_anno)
            self.gapart_cates.append([anno_i["category"] for anno_i in gapart_raw_valid_anno])
            self.gapart_init_bboxes.append(np.array([np.asarray(anno_i["bbox"]) for anno_i in gapart_raw_valid_anno]))
            self.gapart_link_names.append([anno_i["link_name"] for anno_i in gapart_raw_valid_anno])

    # # used
    # def prepare_franka_asset(self):
    #     '''
    #     Prepare franka asset
    #     '''
    #     # load franka asset
    #     franka_asset_file = self.cfgs["asset"]["franka_asset_file"]
    #     asset_options = gymapi.AssetOptions()
    #     # asset_options.armature = 0.01
    #     # asset_options.fix_base_link = True
    #     # asset_options.disable_gravity = True
    #     # asset_options.flip_visual_attachments = True
    #     asset_options.fix_base_link = True
    #     asset_options.disable_gravity = True
    #     # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
    #     asset_options.flip_visual_attachments = True
    #     asset_options.collapse_fixed_joints = False 
    #     asset_options.thickness = 0.001 # default 0.02
    #     asset_options.vhacd_enabled = True
    #     asset_options.vhacd_params = gymapi.VhacdParams()
    #     asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    #     asset_options.vhacd_params.resolution = 100000 # 1000000
    #     asset_options.use_mesh_materials = True
    #     self.franka_asset = self.gym.load_asset(self.sim, self.asset_root, franka_asset_file, asset_options)

    #     # configure franka dofs
    #     self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
    #     franka_lower_limits = self.franka_dof_props["lower"]
    #     franka_upper_limits = self.franka_dof_props["upper"]
    #     franka_ranges = franka_upper_limits - franka_lower_limits
    #     franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)
        
    #     # Set controller parameters
    #     # use position drive for all dofs
    #     if self.controller_name == "ik" or self.controller_name == "curobo":
    #         self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    #         self.franka_dof_props["stiffness"][:7].fill(400.0)
    #         self.franka_dof_props["damping"][:7].fill(40.0)
    #     else:       # osc
    #         self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    #         self.franka_dof_props["stiffness"][:7].fill(0.0)
    #         self.franka_dof_props["damping"][:7].fill(0.0)
    #     # grippers
    #     self.franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
    #     self.franka_dof_props["stiffness"][7:].fill(1.0e6)
    #     self.franka_dof_props["damping"][7:].fill(100.0)

    #     # default dof states and position targets
    #     self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
    #     self.franka_default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
    #     self.franka_default_dof_pos[:7] = franka_mids[:7]
    #     self.franka_default_dof_pos[:7] = np.array([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469], dtype=np.float32)
    #     # grippers open
    #     self.franka_default_dof_pos[7:] = franka_upper_limits[7:]

    #     self.franka_default_dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
    #     self.franka_default_dof_state["pos"] = self.franka_default_dof_pos

    #     # send to torch
    #     self.default_dof_pos_tensor = to_torch(self.franka_default_dof_pos, device=self.device)

    #     # get link index of panda hand, which we will use as end effector
    #     franka_link_dict = self.gym.get_asset_rigid_body_dict(self.franka_asset)
    #     self.franka_num_links = len(franka_link_dict)
    #     # print("franka dof:", self.franka_num_dofs, "franka links:", self.franka_num_links)
    #     self.franka_hand_index = franka_link_dict["panda_hand"]

    def _resolve_mano_urdf_path(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        default_mano_root = os.path.abspath(os.path.join(module_dir, "..", "urdf"))
        mano_asset_root = self.cfgs.get("asset", {}).get("mano_asset_root", default_mano_root)
        mano_asset_file = self.cfgs.get("asset", {}).get("mano_asset_file", "mano.urdf")
        if not os.path.isabs(mano_asset_root):
            mano_asset_root = os.path.abspath(os.path.join(module_dir, mano_asset_root))
        mano_asset_path = os.path.join(mano_asset_root, mano_asset_file)
        return mano_asset_root, mano_asset_file, mano_asset_path


    def prepare_mano_asset(self):
        mano_asset_root, mano_asset_file, mano_asset_path = self._resolve_mano_urdf_path()
        if not os.path.exists(mano_asset_path):
            raise FileNotFoundError(f"MANO URDF not found: {mano_asset_path}")
        
        asset_options = gymapi.AssetOptions()
        asset_options.override_inertia = True 
        asset_options.override_com = True
        
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True 
        
        asset_options.collapse_fixed_joints = False
        asset_options.thickness = 0.001
        asset_options.vhacd_enabled = True 
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 100000 
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS 

        # self.mano_asset = self.gym.load_asset(self.sim, self.asset_root, mano_asset_file, asset_options)
        self.mano_asset = self.gym.load_asset(self.sim, mano_asset_root, mano_asset_file, asset_options)


        self.mano_dof_props = self.gym.get_asset_dof_properties(self.mano_asset)
        self.mano_num_dofs = self.gym.get_asset_dof_count(self.mano_asset)
        print(f"成功加载 MANO 手，检测到 {self.mano_num_dofs} 个自由度。")

        self.mano_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        mano_stiffness = float(self.cfgs.get("asset", {}).get("mano_dof_stiffness", 100.0))
        mano_damping = float(self.cfgs.get("asset", {}).get("mano_dof_damping", 10.0))
        mano_effort = self.cfgs.get("asset", {}).get("mano_dof_effort", None)
        self.mano_dof_props["stiffness"].fill(mano_stiffness)
        self.mano_dof_props["damping"].fill(mano_damping)
        # Effort limit caps the joint motor torque/force; too small => weak grasp.
        if mano_effort is not None:
            try:
                self.mano_dof_props["effort"].fill(float(mano_effort))
            except Exception:
                pass

        self.mano_default_dof_pos = np.zeros(self.mano_num_dofs, dtype=np.float32)
        
        # 构建初始状态张量
        self.mano_default_dof_state = np.zeros(self.mano_num_dofs, gymapi.DofState.dtype)
        self.mano_default_dof_state["pos"] = self.mano_default_dof_pos
        self.default_dof_pos_tensor = to_torch(self.mano_default_dof_pos, device=self.device)

        mano_link_dict = self.gym.get_asset_rigid_body_dict(self.mano_asset)
        self.mano_num_links = len(mano_link_dict)

    # used
    def prepare_obj_assets(self):
        '''
        Prepare object assets, some ycb or objaverse objects
        '''
        table_pose_p = self.cfgs["asset"]["table_pose_p"]
        table_scale = self.cfgs["asset"]["table_scale"]
        self.table_scale = self.cfgs["asset"]["table_scale"]
        table_dims = gymapi.Vec3(table_scale[0], table_scale[1], table_scale[2])
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(table_pose_p[0], table_pose_p[1], table_pose_p[2])

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        obj_asset_files = self.cfgs["asset"]["asset_files"]
        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000000
        self.num_asset_per_env = len(obj_asset_files)
        self.obj_assets = [self.gym.load_asset(self.sim, self.asset_root, obj_asset_file, asset_options) for obj_asset_file in obj_asset_files]
        self.obj_num_links_dict = [self.gym.get_asset_rigid_body_dict(asset_i) for asset_i in self.obj_assets]
        self.obj_num_links = sum([len(obj_num_links) for obj_num_links in self.obj_num_links_dict])
        self.obj_num_dofs = sum([self.gym.get_asset_dof_count(asset_i) for asset_i in self.obj_assets])
        self.table_num_links = 1
    
    # used
    def prepare_arti_obj_assets(self):
        '''
        Prepare articulated object assets
        '''
        ### TODO: support multiple loading
        self.gapartnet_ids = self.cfgs["asset"]["arti_gapartnet_ids"]
        self.gapartnet_root = self.cfgs["asset"]["arti_obj_root"]
        arti_obj_paths = [f"{self.gapartnet_root}/{gapartnet_id}/mobility_annotation_gapartnet.urdf" for gapartnet_id in self.gapartnet_ids]

        arti_obj_asset_options = gymapi.AssetOptions()
        # arti_obj_asset_options.disable_gravity = True     # if not disabled, it will need a very initial large force to open a drawer
        arti_obj_asset_options.fix_base_link = True 
        arti_obj_asset_options.collapse_fixed_joints = True # default False
        # arti_obj_asset_options.convex_decomposition_from_submeshes = True
        arti_obj_asset_options.armature = 0.005 # default 0.0
        arti_obj_asset_options.vhacd_enabled = True
        arti_obj_asset_options.vhacd_params = gymapi.VhacdParams()
        arti_obj_asset_options.vhacd_params.resolution = 100000 # 1000000
        arti_obj_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        arti_obj_asset_options.disable_gravity = False
        arti_obj_asset_options.flip_visual_attachments = False

        # GAPartNet URDFs often omit inertial tags; computing inertia/COM from geometry
        # prevents links from becoming effectively immovable due to invalid inertial data.
        if bool(self.cfgs.get("asset", {}).get("arti_obj_override_inertia", True)):
            arti_obj_asset_options.override_inertia = True
        if bool(self.cfgs.get("asset", {}).get("arti_obj_override_com", True)):
            arti_obj_asset_options.override_com = True
        arti_density = self.cfgs.get("asset", {}).get("arti_obj_density", None)
        if arti_density is not None and hasattr(arti_obj_asset_options, "density"):
            arti_obj_asset_options.density = float(arti_density)
        
        # unused settings, be careful, otherwise it will cause error
        # arti_obj_asset_options.thickness = 0.02
        # arti_obj_asset_options.use_mesh_materials = True
        # arti_obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # arti_obj_asset_options.override_inertia = True
        # arti_obj_asset_options.override_com = True
        # arti_obj_asset_options.fix_base_link = True
        # obj_asset_options.vhacd_enabled = True
        # obj_asset_options.vhacd_params = gymapi.VhacdParams()
        # obj_asset_options.vhacd_params.resolution = 1000000
        self.arti_obj_assets = [self.gym.load_asset(self.sim, self.asset_root, arti_obj_path, arti_obj_asset_options)
                                for arti_obj_path in arti_obj_paths]

        ### TODO: support multiple loading from here
        self.arti_obj_asset = self.arti_obj_assets[0]
        self.arti_obj_num_dofs = self.gym.get_asset_dof_count(self.arti_obj_asset)
        self.arti_obj_dof_dict = self.gym.get_asset_dof_dict(self.arti_obj_asset)
        self.arti_obj_dof_names = list(self.arti_obj_dof_dict.keys())
        arti_obj_link_dict = self.gym.get_asset_rigid_body_dict(self.arti_obj_asset)
        self.arti_obj_num_links = len(arti_obj_link_dict)
        print("obj dof:", self.arti_obj_num_dofs, "obj links:", self.arti_obj_num_links)
        print("obj dof names:", self.arti_obj_dof_names)
        
        # set physical props
        self.arti_obj_dof_props = self.gym.get_asset_dof_properties(self.arti_obj_asset)
        arti_obj_dof_stiffness = float(self.cfgs.get("asset", {}).get("arti_obj_dof_stiffness", 0.0))
        arti_obj_dof_damping = float(self.cfgs.get("asset", {}).get("arti_obj_dof_damping", 10.0))
        arti_obj_dof_friction = float(self.cfgs.get("asset", {}).get("arti_obj_dof_friction", 0.0))
        self.arti_obj_dof_props["stiffness"][:] = arti_obj_dof_stiffness
        self.arti_obj_dof_props["damping"][:] = arti_obj_dof_damping
        self.arti_obj_dof_props["friction"][:] = arti_obj_dof_friction
        self.arti_obj_dof_props["driveMode"][:] = gymapi.DOF_MODE_NONE
        
        
        init_pos = self.arti_obj_dof_props["lower"]
        self.arti_obj_default_dof_pos = np.zeros(self.arti_obj_num_dofs, dtype=np.float32)
        self.arti_obj_default_dof_state = np.zeros(self.arti_obj_num_dofs, gymapi.DofState.dtype)
        self.arti_obj_default_dof_state["pos"] = init_pos
        self.arti_default_dof_pos_tensor = to_torch(self.arti_obj_default_dof_pos, device=self.device)
        
    # used
    def load_env(self, load_cam = True):
        '''
        Load environment
        '''
        self.envs = []
        self.obj_actor_idxs = []
        self.hand_idxs = []
        self.init_franka_pos_list = []
        self.init_franka_rot_list = []
        self.init_obj_pos_list = []
        self.init_obj_rot_list = []
        self.arti_init_obj_pos_list = []
        self.arti_init_obj_rot_list = []
        self.env_offsets = []
        self.arti_obj_actor_idxs = []

        self.mano_actor_idxs = []

        # init pose
        franka_pose = gymapi.Transform()
        franka_pose_p = self.cfgs["asset"]["franka_pose_p"]
        franka_pose.p = gymapi.Vec3(franka_pose_p[0], franka_pose_p[1], franka_pose_p[2])
        
        # object pose
        obj_pose_ps = [self.cfgs["asset"]["obj_pose_ps"][obj_i] for obj_i in range(self.num_asset_per_env)]
        if self.cfgs["asset"]["obj_pose_rs"] is not None:
            obj_pose_rs = [self.cfgs["asset"]["obj_pose_rs"][obj_i] for obj_i in range(self.num_asset_per_env)]
        else:
            obj_pose_rs = None
            
        # noise
        position_noise = self.cfgs["asset"]["position_noise"]
        rotation_noise = self.cfgs["asset"]["rotation_noise"]
        
        # arti obj pose
        arti_obj_pose_ps = self.cfgs["asset"]["arti_obj_pose_ps"]
        arti_obj_pose_p = arti_obj_pose_ps[0]
        arti_position_noise = self.cfgs["asset"]["arti_position_noise"]
        arti_rotation_noise = self.cfgs["asset"]["arti_rotation_noise"]
        arti_rotation = self.cfgs["asset"]["arti_rotation"]
        
        # load camera
        if load_cam:
            self.cams = []
            self.rgb_tensors = []
            self.depth_tensors = []
            self.seg_tensors = []
            self.cam_vinvs = []
            self.cam_projs = []
            

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)
            origin = self.gym.get_env_origin(env)
            self.env_offsets.append([origin.x, origin.y, origin.z])

            # add franka
            # franka_handle = self.gym.create_actor(env, self.franka_asset, franka_pose, "franka", i, 2, 0) #self.franka_seg_id
            
            mano_pose = gymapi.Transform()
            mano_pose.p = gymapi.Vec3(franka_pose_p[0], franka_pose_p[1], franka_pose_p[2])
            mano_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            # set dof properties
            # self.gym.set_actor_dof_properties(env, franka_handle, self.franka_dof_props)

            # set initial dof states
            # self.gym.set_actor_dof_states(env, franka_handle, self.franka_default_dof_state, gymapi.STATE_ALL)
            mano_handle = self.gym.create_actor(env, self.mano_asset, mano_pose, "mano_hand", i, 2, 0)
            mano_actor_idx = self.gym.find_actor_index(env, "mano_hand", gymapi.DOMAIN_SIM)
            self.mano_actor_idxs.append(mano_actor_idx)

            # 3. 应用物理属性和初始状态
            self.gym.set_actor_dof_properties(env, mano_handle, self.mano_dof_props)
            self.gym.set_actor_dof_states(env, mano_handle, self.mano_default_dof_state, gymapi.STATE_ALL)
            self.gym.set_actor_dof_position_targets(env, mano_handle, self.mano_default_dof_pos)

            mano_contact_offset = self.cfgs.get("asset", {}).get("mano_shape_contact_offset", None)
            mano_friction = self.cfgs.get("asset", {}).get("mano_shape_friction", None)
            mano_thickness = self.cfgs.get("asset", {}).get("mano_shape_thickness", None)
            if mano_contact_offset is not None or mano_friction is not None or mano_thickness is not None:
                mano_shape_props = self.gym.get_actor_rigid_shape_properties(env, mano_handle)
                for prop in mano_shape_props:
                    if mano_contact_offset is not None:
                        prop.contact_offset = float(mano_contact_offset)
                    if mano_friction is not None:
                        prop.friction = float(mano_friction)
                    if mano_thickness is not None:
                        prop.thickness = float(mano_thickness)
                self.gym.set_actor_rigid_shape_properties(env, mano_handle, mano_shape_props)

            # 记录根节点索引，以后如果你想动态平移手掌会用到
            hand_idx = self.gym.find_actor_rigid_body_index(env, mano_handle, "palm", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            # set initial position targets
            # self.gym.set_actor_dof_position_targets(env, franka_handle, self.franka_default_dof_pos)

            # get inital hand pose
            # hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            # hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            # self.init_franka_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            # self.init_franka_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # # get global index of hand in rigid body state tensor
            # hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            # self.hand_idxs.append(hand_idx)
            
            ### Table
            self.table_handle = self.gym.create_actor(env, self.table_asset, self.table_pose, "table", i, 0, self.table_seg_id)
            
            ## Object Assets
            self.init_obj_pos_list.append([])
            self.init_obj_rot_list.append([])
            self.obj_actor_idxs.append([])
            for asset_i in range(self.num_asset_per_env):
                initial_pose = gymapi.Transform()
                initial_pose.p.x = obj_pose_ps[asset_i][0] + np.random.uniform(-1.0, 1.0) * position_noise[0]
                initial_pose.p.y = obj_pose_ps[asset_i][1] + np.random.uniform(-1.0, 1.0) * position_noise[1]
                initial_pose.p.z = obj_pose_ps[asset_i][2]
                if obj_pose_rs is None:
                    initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), rotation_noise/180.0*np.random.uniform(-math.pi, math.pi))
                else:
                    initial_pose.r.x =  obj_pose_rs[asset_i][0]
                    initial_pose.r.y =  obj_pose_rs[asset_i][1]
                    initial_pose.r.z =  obj_pose_rs[asset_i][2]
                    initial_pose.r.w =  obj_pose_rs[asset_i][3]
                # initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), rotation_noise/180.0*np.random.uniform(-math.pi, math.pi))

                self.init_obj_pos_list[-1].append([initial_pose.p.x, initial_pose.p.y, initial_pose.p.z])
                self.init_obj_rot_list[-1].append([initial_pose.r.x, initial_pose.r.y, initial_pose.r.z, initial_pose.r.w])
                
                obj_actor_handle = self.gym.create_actor(env, self.obj_assets[asset_i], initial_pose, f'actor_{asset_i}', i, 0, self.asset_seg_ids[asset_i])
                
                obj_actor_idx = self.gym.get_actor_rigid_body_index(env, obj_actor_handle, 0, gymapi.DOMAIN_SIM)
                self.obj_actor_idxs[i].append(obj_actor_idx)
                self.gym.set_actor_scale(env, obj_actor_handle, self.cfgs["asset"]["obj_scale"])
            
            if self.cfgs["USE_ARTI"]:
                ### Articulated Object
                arti_initial_pose = gymapi.Transform()
                arti_initial_pose.p.x = arti_obj_pose_p[0] + np.random.uniform(-1.0, 1.0) * arti_position_noise
                arti_initial_pose.p.y = arti_obj_pose_p[1] + np.random.uniform(-1.0, 1.0) * arti_position_noise
                arti_initial_pose.p.z = arti_obj_pose_p[2]
                arti_initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), arti_rotation/180.0*math.pi + arti_rotation_noise/180.0*np.random.uniform(-math.pi, math.pi))
                self.arti_init_obj_pos_list.append([arti_initial_pose.p.x, arti_initial_pose.p.y, arti_initial_pose.p.z])
                self.arti_init_obj_rot_list.append([arti_initial_pose.r.x, arti_initial_pose.r.y, arti_initial_pose.r.z, arti_initial_pose.r.w])
                arti_obj_actor_handle = self.gym.create_actor(env, self.arti_obj_asset, arti_initial_pose, 'arti_actor', i, 1, 0) #1, self.asset_seg_ids[-1] + 1
                
                self.gym.set_actor_dof_properties(env, arti_obj_actor_handle, self.arti_obj_dof_props)
                # set initial dof states
                ### TODO check
                # self.arti_obj_default_dof_state["pos"][:3] = 2 + np.random.uniform(-1.0, 1.0) * 0.5
                self.gym.set_actor_dof_states(env, arti_obj_actor_handle, self.arti_obj_default_dof_state, gymapi.STATE_ALL)
                # Only set dof targets if the articulation is actually driven.
                # For DOF_MODE_NONE, setting targets is unnecessary and can be confusing.
                try:
                    driven = np.any(self.arti_obj_dof_props["driveMode"] != gymapi.DOF_MODE_NONE)
                except Exception:
                    driven = False
                if driven:
                    self.gym.set_actor_dof_position_targets(env, arti_obj_actor_handle, self.arti_obj_default_dof_state["pos"])
                arti_obj_actor_idx = self.gym.get_actor_rigid_body_index(env, arti_obj_actor_handle, 0, gymapi.DOMAIN_SIM)
                self.arti_obj_actor_idxs.append(arti_obj_actor_idx)
                self.gym.set_actor_scale(env, arti_obj_actor_handle, self.cfgs["asset"]["arti_obj_scale"])
                
                agent_shape_props = self.gym.get_actor_rigid_shape_properties(env, arti_obj_actor_handle)
                arti_contact_offset = float(self.cfgs.get("asset", {}).get("arti_shape_contact_offset", 0.02))
                arti_friction = float(self.cfgs.get("asset", {}).get("arti_shape_friction", 5.0))
                arti_thickness = float(self.cfgs.get("asset", {}).get("arti_shape_thickness", 0.2))
                for agent_shape_prop in agent_shape_props:
                    # agent_shape_prop.compliance = agent.rigid_shape_compliance
                    agent_shape_prop.contact_offset = arti_contact_offset  # 0.001
                    # agent_shape_prop.filter = agent.rigid_shape_filter
                    agent_shape_prop.friction = arti_friction
                    # agent_shape_prop.rest_offset = agent.rigid_shape_rest_offset
                    # agent_shape_prop.restitution = agent.rigid_shape_restitution
                    # agent_shape_prop.rolling_friction = agent.rigid_shape_rolling_friction
                    agent_shape_prop.thickness = arti_thickness
                    # agent_shape_prop.torsion_friction = agent.rigid_shape_torsion_friction
                self.gym.set_actor_rigid_shape_properties(env, arti_obj_actor_handle, agent_shape_props)  
            
            if load_cam:
                # add camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cam_w
                cam_props.height = self.cam_h
                cam_props.far_plane = self.cam_far_plane
                cam_props.near_plane = self.cam_near_plane 
                cam_props.horizontal_fov = self.horizontal_fov
                cam_props.enable_tensors = True
                self.cams.append([])
                self.depth_tensors.append([])
                self.rgb_tensors.append([])
                self.seg_tensors.append([])
                self.cam_vinvs.append([])
                self.cam_projs.append([])
                for i in range(self.num_cam_per_env):
                    cam_handle = self.gym.create_camera_sensor(env, cam_props)
                    self.gym.set_camera_location(cam_handle, env, 
                        gymapi.Vec3(self.cam_poss[i][0], self.cam_poss[i][1], self.cam_poss[i][2]), 
                        gymapi.Vec3(self.cam_targets[i][0], self.cam_targets[i][1], self.cam_targets[i][2]))
                    self.cams[-1].append(cam_handle)
                
                    proj = self.gym.get_camera_proj_matrix(self.sim, env, cam_handle)
                    # view_matrix_inv = torch.inverse(torch.tensor(self.gym.get_camera_view_matrix(self.sim, env, cam_handle))).to(self.device)
                    vinv = np.linalg.inv(np.matrix(self.gym.get_camera_view_matrix(self.sim, env, cam_handle)))
                    self.cam_vinvs[-1].append(vinv)
                    self.cam_projs[-1].append(proj)

                    # obtain rgb tensor
                    rgb_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
                    # wrap camera tensor in a pytorch tensor
                    torch_rgb_tensor = gymtorch.wrap_tensor(rgb_tensor)
                    self.rgb_tensors[-1].append(torch_rgb_tensor)
                    
                    # obtain depth tensor
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env, cam_handle, gymapi.IMAGE_DEPTH)
                    # wrap camera tensor in a pytorch tensor
                    torch_depth_tensor = gymtorch.wrap_tensor(depth_tensor)
                    self.depth_tensors[-1].append(torch_depth_tensor)
        

                    # obtain depth tensor
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
                    # wrap camera tensor in a pytorch tensor
                    torch_seg_tensor = gymtorch.wrap_tensor(seg_tensor)
                    self.seg_tensors[-1].append(torch_seg_tensor)
                    
                    
            
        self.env_offsets = np.array(self.env_offsets)
        
        # point camera at middle env
        if not self.headless:
            
            viewer_cam_pos = gymapi.Vec3(self.cam_poss[0][0], self.cam_poss[0][1], self.cam_poss[0][2])
            viewer_cam_target = gymapi.Vec3(self.cam_targets[0][0], self.cam_targets[0][1], self.cam_targets[0][2])
            middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, viewer_cam_pos, viewer_cam_target)

        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)

    def control_ik(self, dpose):
        # global damping, j_eef, num_envs
        damping = 0.05
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u
    
    def plan_to_pose_ik(self, goal_position, goal_roation, close_gripper = True, save_video = False, save_root = "", start_step = 0, control_steps = 10):
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        hand_rot_now = self.hand_rot
        goal_roation = goal_roation.to(self.device).reshape(1,-1)
        goal_position = goal_position.to(self.device).reshape(1,-1)
        if goal_roation.shape[1] != 0:
            orn_err = orientation_error(goal_roation, hand_rot_now)
        else:
            orn_err = hand_rot_now[...,:3].clone()
            orn_err[...] = 0
        pos_err = goal_position - self.hand_pos
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        pos_action[:, :7] = self.dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)
        if close_gripper:
            grip_acts = torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)
        else:
            grip_acts = torch.Tensor([[1, 1]] * self.num_envs).to(self.device)
        # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device), 
        #                         torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
        pos_action[:, 7:9] = grip_acts
        for step_i in range(control_steps):
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))            
            self.run_steps(pre_steps = 1)
            if save_video:
                self.gym.render_all_camera_sensors(self.sim)
                step_str = str(start_step + step_i).zfill(4)
                os.makedirs(f"{save_root}/video", exist_ok=True)
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.cams[0][0], gymapi.IMAGE_COLOR, f"{save_root}/video/step-{step_str}.png")

    def init_observation(self):
        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "mano_hand")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        # self.j_eef = self.jacobian[:, self.franka_hand_index - 1, :, :7]

        # get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "mano_hand")
        self.mm = gymtorch.wrap_tensor(_massmatrix)
        # self.mm = self.mm[:, :7, :7]          # only need elements corresponding to the franka arm
        
        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        num_rb = int(self.rb_states.shape[0]/self.num_envs)
        
        self.root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.gym.refresh_actor_root_state_tensor(self.sim)
            
        
        if self.cfgs["USE_ARTI"]:
            assert num_rb == self.mano_num_links + self.obj_num_links + self.table_num_links + self.arti_obj_num_links, "Number of rigid bodies in tensor does not match franka & obj asset"
        else:
            assert num_rb == self.mano_num_links + self.obj_num_links + self.table_num_links, "Number of rigid bodies in tensor does not match franka & obj asset"
        
        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)

        num_dof = int(self.dof_states.shape[0]/self.num_envs)
        if self.cfgs["USE_ARTI"]:
            assert num_dof == self.mano_num_dofs + self.obj_num_dofs + self.arti_obj_num_dofs
        else:
            assert num_dof == self.mano_num_dofs + self.obj_num_dofs

        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, num_dof, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, num_dof, 1)

    def _mano_actor_indices_tensor(self) -> torch.Tensor:
        """Global actor indices (DOMAIN_SIM) for MANO actors, as int32 tensor."""
        if not hasattr(self, "mano_actor_idxs") or len(self.mano_actor_idxs) == 0:
            return torch.zeros((0,), dtype=torch.int32, device=self.device)
        idx = getattr(self, "_mano_actor_idxs_tensor", None)
        if idx is None or idx.numel() != len(self.mano_actor_idxs):
            idx = torch.as_tensor(self.mano_actor_idxs, dtype=torch.int32, device=self.device)
            self._mano_actor_idxs_tensor = idx
        return idx

    def _set_mano_root_state_tensor(self, root_states: torch.Tensor) -> None:
        """
        Only set the root state for MANO actors.

        This avoids accidentally overwriting the root state of other dynamic actors
        (e.g. YCB objects), which can make them appear "stuck" during interaction.
        """
        mano_actor_idxs = self._mano_actor_indices_tensor()
        if mano_actor_idxs.numel() == 0:
            # Fallback: keep legacy behavior if MANO actor indices are missing.
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))
            return
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(mano_actor_idxs),
            int(mano_actor_idxs.numel()),
        )

    def refresh_observation(self, get_visual_obs = True):
        # refresh tensors
        # Keep root states in sync; several controllers "teleport" the MANO root.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        
        # state obs
        self.hand_pos = self.rb_states[self.hand_idxs, :3]
        self.hand_rot = self.rb_states[self.hand_idxs, 3:7]
        self.hand_vel = self.rb_states[self.hand_idxs, 7:]

        ### TODO: support different dof tensor shapes in different envs
        # self.robot_dof_qpos_qvel = self.dof_states.reshape(self.num_envs,-1,2)[:,:self.franka_num_dofs, :].view(self.num_envs, self.franka_num_dofs, 2)
        self.robot_dof_qpos_qvel = self.dof_states.reshape(self.num_envs,-1,2)[:,:self.mano_num_dofs, :].view(self.num_envs, self.mano_num_dofs, 2)
        
        # render sensors and refresh camera tensors
        if self.use_cam and get_visual_obs:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            points_envs = []
            colors_envs = []
            ori_points_envs = []
            ori_colors_envs = []
            rgb_envs = []
            depth_envs = []
            seg_envs = []
            # bbox_axis_aligned_envs = []
            # bbox_center_envs = []
            # import pdb; pdb.set_trace()
            for env_i in range(self.num_envs):
                points_env = []
                colors_env = []
                rgb_env = []
                depth_env = []
                seg_env = []
                for cam_i_per_env in range(self.num_cam_per_env):
                    # write tensor to image
                    cam_img = self.rgb_tensors[env_i][cam_i_per_env].cpu().numpy()
                    depth = self.depth_tensors[env_i][cam_i_per_env].cpu().numpy() # W * H
                    seg = self.seg_tensors[env_i][cam_i_per_env].cpu().numpy() # W * H

                    rgb_env.append(cam_img)
                    depth_env.append(depth)
                    seg_env.append(seg)
                    
                    # if self.cfgs["INFERENCE_GSAM"]:
                    #     masks = inference_one_image(cam_img[..., :3], self.grounded_dino_model, self.sam_predictor, box_threshold=self.box_threshold, 
                    #         text_threshold=self.text_threshold, text_prompt=text_prompt, device=self.device)
                    
                    #     if self.cfgs["SAVE_RENDER"]:
                    #         save_dir = self.cfgs["SAVE_ROOT"]
                    #         os.makedirs(save_dir, exist_ok=True)
                    #         import cv2
                    #         for i in range(masks.shape[0]):
                    #             cam_img_ = cam_img.copy()
                    #             cam_img_[masks[i][0].cpu().numpy()] = 0
                    #             fname = os.path.join(save_dir, text_prompt + "-mask-%04d-%04d-%04d-%04d.png" % (0, env_i, cam_i_per_env, i))
                    #             imageio.imwrite(fname, cam_img_)

                        
                    ### RGBD -> Point Cloud with CPU
                    # s = time.time()
                    # points, colors = get_point_cloud_from_rgbd(depth, cam_img, None, self.cam_vinvs[env_i][cam_i_per_env], self.cam_projs[env_i][cam_i_per_env], self.cam_w, self.cam_h)
                    # points = np.transpose(points(0, 2, 1))
                    # e = time.time()
                    # print("Time to get point cloud: ", e-s)
                    
                    ### RGBD -> Point Cloud with GPU
                    s = time.time()
                    pointclouds = get_point_cloud_from_rgbd_GPU(
                        self.depth_tensors[env_i][cam_i_per_env], 
                        self.rgb_tensors[env_i][cam_i_per_env],
                        None,
                        self.cam_vinvs[env_i][cam_i_per_env], 
                        self.cam_projs[env_i][cam_i_per_env], 
                        self.cam_w, self.cam_h
                    )
                    points = pointclouds[:, :3].cpu().numpy()
                    colors = pointclouds[:, 3:6].cpu().numpy()
                    i_indices, j_indices = np.meshgrid(np.arange(self.cam_w), np.arange(self.cam_h), 
                            indexing='ij')
                    pointid2pixel = np.stack((i_indices, j_indices), axis=-1).reshape(-1, 2)
                    pixel2pointid = np.arange(self.cam_w * self.cam_h).reshape(self.cam_w, self.cam_h)
                    pointid2pixel = None
                    pixel2pointid = None
                    points_env.append(points)
                    colors_env.append(colors)
                    
                    # e = time.time()
                    # print("Time to get point cloud: ", e-s)
                    
                    # if self.cfgs["INFERENCE_GSAM"]:
                    #     pc_mask = masks[0][0].cpu().numpy().reshape(-1)
                    #     target_points = points[pc_mask]
                    #     target_colors = colors[pc_mask]
                    #     point_cloud = o3d.geometry.PointCloud()
                    #     point_cloud.points = o3d.utility.Vector3dVector(target_points[:, :3])
                    #     point_cloud.colors = o3d.utility.Vector3dVector(target_colors[:, :3]/255.0)
                        
                    #     if self.cfgs["SAVE_RENDER"]:
                    #         # save_to ply
                    #         fname = os.path.join(save_dir, "point_cloud-%04d-%04d-target.ply" % (env_i, cam_i_per_env))
                    #         o3d.io.write_point_cloud(fname, point_cloud)
                    #     bbox_axis_aligned = np.array([target_points.min(axis=0), target_points.max(axis=0)])
                    #     bbox_center = bbox_axis_aligned.mean(axis=0)
                    #     bbox_axis_aligned_envs.append(bbox_axis_aligned)
                    #     bbox_center_envs.append(bbox_center)
                    #     masks_envs.append(masks)
                    

                ori_points_envs.append(points_env)
                ori_colors_envs.append(colors_env)
                rgb_envs.append(rgb_env)
                depth_envs.append(depth_env)
                seg_envs.append(seg_env)
                points_env = np.concatenate(points_env, axis=0) - self.env_offsets[env_i]
                colors_env = np.concatenate(colors_env, axis=0) - self.env_offsets[env_i]
                pc_mask_bound = (points_env[:, 0] > self.point_cloud_bound[0][0]) & (points_env[:, 0] < self.point_cloud_bound[0][1]) & \
                                (points_env[:, 1] > self.point_cloud_bound[1][0]) & (points_env[:, 1] < self.point_cloud_bound[1][1]) & \
                                (points_env[:, 2] > self.point_cloud_bound[2][0]) & (points_env[:, 2] < self.point_cloud_bound[2][1])
                points_env = points_env[pc_mask_bound]
                colors_env = colors_env[pc_mask_bound]

                s = time.time()
                points_env, colors_env, pcs_mask = get_downsampled_pc(points_env, colors_env, 
                    sampled_num=self.cfgs["cam"]["sampling_num"], sampling_method = self.cfgs["cam"]["sampling_method"])
                e = time.time()
                points_envs.append(points_env)
                colors_envs.append(colors_env)
                print("Time to get point cloud: ", e-s)

            self.gym.end_access_image_tensors(self.sim)

        
            return points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel

    def save_render(self, rgb_envs, depth_envs, ori_points_env, ori_colors_env, points, colors, save_dir, save_name = "render", save_pc = False, save_depth = False, save_single = False):
        for env_i in range(len(rgb_envs)):
            for cam_i in range(len(rgb_envs[0])):
                fname = os.path.join(save_dir, f"{save_name}-rgb-{env_i}-{cam_i}.png")
                os.makedirs(save_dir, exist_ok=True)
                imageio.imwrite(fname, rgb_envs[env_i][cam_i].astype(np.uint8))
                if save_single:
                    return
        
                if depth_envs is not None and save_depth:
                    depth = depth_envs[env_i][cam_i]
                    # depth clip to 0.1m - 10m and scale to 0-255
                    depth_clip = np.clip(depth, -1, 1)
                    depth_rgb = (depth_clip + 1) / 2 * 255.0
                    # W * H * 3
                    depth_img = np.zeros((depth_rgb.shape[0], depth_rgb.shape[1], 3))
                    depth_img[:, :, 0] = depth_rgb
                    depth_img[:, :, 1] = depth_rgb
                    depth_img[:, :, 2] = depth_rgb
                    fname = os.path.join(save_dir, f"{save_name}-depth-{env_i}-{cam_i}.png")
                    os.makedirs(save_dir, exist_ok=True)
                    imageio.imwrite(fname, depth_img.astype(np.uint8))
            
                if ori_points_env is not None and save_pc:
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(ori_points_env[env_i][cam_i][:, :3])
                    point_cloud.colors = o3d.utility.Vector3dVector(ori_colors_env[env_i][cam_i][:, :3]/255.0)
                    # save_to ply
                    fname = os.path.join(save_dir, f"{save_name}-partial-point_cloud--{env_i}-{cam_i}.ply")
                    o3d.io.write_point_cloud(fname, point_cloud)
            # o3d.visualization.draw_geometries([point_cloud])
            if points is not None and save_pc:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points[env_i][:, :3])
                point_cloud.colors = o3d.utility.Vector3dVector(colors[env_i][:, :3]/255.0)
                # save_to ply
                fname = os.path.join(save_dir, f"{save_name}-{env_i}-all-point_cloud.ply")
                o3d.io.write_point_cloud(fname, point_cloud)

    def inference_gsam(self, rgb_img, points, colors, text_prompt, save_dir, save_name = "gsam"):
        
        bbox_axis_aligned_envs = []
        bbox_center_envs = []
        
        assert self.cfgs["INFERENCE_GSAM"]
        masks = inference_one_image(rgb_img[..., :3], self.grounded_dino_model, self.sam_predictor, box_threshold=self.box_threshold, text_threshold=self.text_threshold, text_prompt=text_prompt, device=self.device)

        if masks is None:
            # import pdb; pdb.set_trace()
            return None, None, None
        if self.cfgs["SAVE_RENDER"]:
            os.makedirs(save_dir, exist_ok=True)
            for i in range(masks.shape[0]):
                cam_img_ = rgb_img.copy()
                cam_img_[masks[i][0].cpu().numpy()] = 0
                fname = os.path.join(save_dir, f"{save_name}-gsam-mask-{text_prompt}-{i}.png")
                imageio.imwrite(fname, cam_img_)
                np.save(fname.replace(".png", ".npy"), masks[i][0].cpu().numpy())
                
        
        for i in range(masks.shape[0]):
            pc_mask = masks[i][0].cpu().numpy().reshape(-1)
            target_points = points[pc_mask]
            target_colors = colors[pc_mask]
            
            if self.cfgs["SAVE_RENDER"]:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(target_points[:, :3])
                point_cloud.colors = o3d.utility.Vector3dVector(target_colors[:, :3]/255.0)
                # save_to ply
                fname = os.path.join(save_dir, f"{save_name}-gsam-mask-{text_prompt}-{i}.ply")
                o3d.io.write_point_cloud(fname, point_cloud)
                
            bbox_axis_aligned = np.array([target_points.min(axis=0), target_points.max(axis=0)])
            bbox_center = bbox_axis_aligned.mean(axis=0)
            bbox_axis_aligned_envs.append(bbox_axis_aligned)
            bbox_center_envs.append(bbox_center)

        return masks, bbox_axis_aligned_envs, bbox_center_envs

    def plan_to_pose_curobo(self, position, quaternion, max_attempts=100, start_state= None):
        '''
        start_state: JointState
            if None, use current state as start state
            else, use given start_state
            
        position: list or np.array
            target position
        quaternion: list or np.array
            target orientation
        '''
        if start_state == None:
            start_state = JointState.from_position(self.robot_dof_qpos_qvel[:,:7,0].clone())
            # start_state = JointState.from_position(self.robot_dof_qpos_qvel[:,:7,0])
        goal_pos = torch.tensor(position, device=self.device, dtype=torch.float64) - torch.tensor(self.cfgs["asset"]["franka_pose_p"], device=self.device, dtype=torch.float64)
        goal_quat = torch.tensor(quaternion, device=self.device, dtype=torch.float64)
        goal_state = Pose(goal_pos, quaternion=goal_quat)
        
        result = self.motion_gen.plan_single(start_state, goal_state, MotionGenPlanConfig(max_attempts=max_attempts))

        traj = result.get_interpolated_plan()
        # if result.optimized_dt == None or result.success[0] == False:
        #     return None
        try:
            print("Trajectory Generated: ", result.success, result.optimized_dt.item(), traj.position.shape)
        except:
            print("Trajectory Generated: ", result.success)
        return traj

    def move_to_traj(self, traj, close_gripper = True, save_video = False, save_root = "", start_step = 0):
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        #import pdb; pdb.set_trace()
        for step_i in range(len(traj)):
            # print("Step: ", step_i)
            # Deploy actions
            pos_action[:, :7] = traj.position.reshape(-1, 7)[step_i]
            if close_gripper:
                grip_acts = torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)
            else:
                grip_acts = torch.Tensor([[1, 1]] * self.num_envs).to(self.device)
            # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device), 
            #                         torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
            pos_action[:, 7:9] = grip_acts
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))
            self.run_steps(pre_steps = 1)
            if save_video:
                self.gym.render_all_camera_sensors(self.sim)
                # print("Saving video frame:", start_step + step_i)
                step_str = str(start_step + step_i).zfill(4)
                os.makedirs(f"{save_root}/video", exist_ok=True)
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.cams[0][0], gymapi.IMAGE_COLOR, f"{save_root}/video/step-{step_str}.png")
                # self.gym.write_viewer_image_to_file(self.viewer, f"{save_root}/step-{start_step + step_i}.png")
          
    def move_gripper(self, close_gripper = True, save_video = False, save_root = "", start_step = 0):
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        if close_gripper:
            grip_acts = torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)
        else:
            grip_acts = torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device)
        # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device), 
        #                         torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
        pos_action[:, :7] = self.robot_dof_qpos_qvel[:,:7,0]
        pos_action[:, 7:9] = grip_acts
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))
        self.run_steps(pre_steps = 5)
        if save_video:
            self.gym.render_all_camera_sensors(self.sim)
            # print("Saving video frame:", start_step)
            # start_step string, 4 digit
            step_str = str(start_step).zfill(4)
            os.makedirs(f"{save_root}/video", exist_ok=True)
            self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.cams[0][0], gymapi.IMAGE_COLOR, f"{save_root}/video/step-{step_str}.png")
        return start_step + 1
    

    def control_to_pose(self, pose_from_optim, qpos_from_optim, close_gripper=True, save_video=False, save_root="", step_num=0):
        """
        接收来自 PyTorch 优化的姿态，在物理引擎中直接执行并验证。
        """
        print(" 正在将优化姿态同步至 Isaac Gym...")
        
        # 1. 强制传送 Root 状态 (手腕飞到把手处)
        root_states = self.root_states.clone()
        for env_i in range(self.num_envs):
            mano_idx = self.mano_actor_idxs[env_i]
            # 设置位置 (xyz) 和旋转 (quaternion)
            root_states[mano_idx, :3] = torch.tensor(pose_from_optim[:3], dtype=torch.float32, device=self.device)
            root_states[mano_idx, 3:7] = torch.tensor(pose_from_optim[3:], dtype=torch.float32, device=self.device)
            root_states[mano_idx, 7:13] = 0.0 # 动量清零
            
        self._set_mano_root_state_tensor(root_states)

        # 获取当前环境中所有自由度 (手 + 物体) 的当前目标值，Shape: (num_envs, 23)
        pos_action = self.dof_pos.squeeze(-1).clone() 
        
        # 将我们优化好的 MANO 20个关节角度转成 tensor
        mano_targets = torch.tensor(qpos_from_optim, dtype=torch.float32, device=self.device)
        
        # 只覆盖前 20 个属于 MANO 的自由度，保留后 3 个属于柜门的自由度不变
        pos_action[:, :self.mano_num_dofs] = mano_targets 
        
        # 发送给 Isaac Gym
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
        
        # 3. 运行物理仿真进行“验证”
        print(" 等待手指闭合 (运行 50 帧)...")
        self.run_steps(pre_steps=50, refresh_obs=True, print_step=False)

        return step_num + 50, None
    
    def smooth_control(self, start_pose, end_pose, start_qpos, end_qpos, steps=60):
        """
        线性插值控制器：让手平滑地移动和抓取，展现整个物理过程
        steps: 插值的帧数，数值越大动作越慢
        """
        import time
        for step in range(steps):
            # 计算当前进度的比例 (0.0 到 1.0)
            alpha = step / float(steps)
            
            # 1. 插值计算当前的位置和手指角度
            current_pos = start_pose[:3] * (1 - alpha) + end_pose[:3] * alpha
            current_rot = end_pose[3:] # 旋转保持目标旋转
            current_qpos = start_qpos * (1 - alpha) + end_qpos * alpha
            
            # 2. 更新手腕位置 (Root State)
            root_states = self.root_states.clone()
            for env_i in range(self.num_envs):
                mano_idx = self.mano_actor_idxs[env_i]
                root_states[mano_idx, :3] = torch.tensor(current_pos, dtype=torch.float32, device=self.device)
                root_states[mano_idx, 3:7] = torch.tensor(current_rot, dtype=torch.float32, device=self.device)
                root_states[mano_idx, 7:13] = 0.0 # 消除惯性
            self._set_mano_root_state_tensor(root_states)
            
            # 3. 更新手指关节目标角度 (Target Tensor)
            pos_action = self.dof_pos.squeeze(-1).clone() 
            pos_action[:, :self.mano_num_dofs] = torch.tensor(current_qpos, dtype=torch.float32, device=self.device)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            
            # 4. 步进物理引擎1帧
            self.run_steps(pre_steps=1, refresh_obs=True, print_step=False)
            time.sleep(0.01) # 微微暂停，让你肉眼能看清动作过程

    def _get_current_arti_obj_urdf_path(self):
        if not self.cfgs.get("USE_ARTI", False):
            return None
        if not hasattr(self, "gapartnet_root") or not hasattr(self, "gapartnet_ids"):
            return None
        if len(self.gapartnet_ids) == 0:
            return None
        return os.path.join(
            self.asset_root,
            self.gapartnet_root,
            str(self.gapartnet_ids[0]),
            "mobility_annotation_gapartnet.urdf",
        )

    def _ensure_contact_calc(self, obj_urdf_path, points_per_link=1000, hand_points_per_link=200):
        if obj_urdf_path is None:
            return
        if hasattr(self, "contact_calc") and getattr(self, "_contact_calc_obj_urdf_path", None) == obj_urdf_path:
            return

        _, _, mano_urdf = self._resolve_mano_urdf_path()
        obj_scale = 1.0
        try:
            obj_scale = float(self.cfgs.get("asset", {}).get("arti_obj_scale", 1.0))
        except Exception:
            obj_scale = 1.0
        self.contact_calc = FastContactCalculator(
            mano_urdf,
            obj_urdf_path,
            device=self.device,
            obj_scale=obj_scale,
            points_per_link=points_per_link,
            hand_points_per_link=hand_points_per_link,
        )
        self._contact_calc_obj_urdf_path = obj_urdf_path

    def _compute_surface_contact_summary(
        self,
        hand_pose_6d,
        obj_urdf_path,
        surface_contact_thresh=0.015,
    ):
        self._ensure_contact_calc(obj_urdf_path=obj_urdf_path)
        if not hasattr(self, "contact_calc"):
            return 0, {}, float("inf")

        h_pos = torch.tensor(hand_pose_6d[:3], dtype=torch.float32, device=self.device).unsqueeze(0)
        h_rot = torch.tensor(hand_pose_6d[3:7], dtype=torch.float32, device=self.device).unsqueeze(0)
        h_qpos = self.dof_pos[0, : self.mano_num_dofs, 0].unsqueeze(0)

        o_pos = torch.tensor(self.arti_init_obj_pos_list[0], dtype=torch.float32, device=self.device).unsqueeze(0)
        o_rot = torch.tensor(self.arti_init_obj_rot_list[0], dtype=torch.float32, device=self.device).unsqueeze(0)
        o_qpos = self.dof_pos[
            0, self.mano_num_dofs : self.mano_num_dofs + self.arti_obj_num_dofs, 0
        ].unsqueeze(0)

        contact_mask, min_dists, link_counts = self.contact_calc.compute_batch_surface_contact(
            h_pos,
            h_rot,
            h_qpos,
            o_pos,
            o_rot,
            o_qpos,
            thresh=surface_contact_thresh,
        )
        contact_count = int(contact_mask.to(torch.int32).sum().item())
        link_counts_int = {k: int(v.item()) for k, v in link_counts.items()}

        # Compute actual signed distance so penetration (negative) is detectable.
        # The unsigned cdist min_dist is always >= 0 and cannot represent penetration.
        merged_hand_points, _ = self.contact_calc._compute_hand_surface_points_world(
            h_pos, h_rot, h_qpos
        )
        try:
            signed_dists, _, _ = self.contact_calc.compute_batch_signed_distance(
                merged_hand_points, o_pos, o_rot, o_qpos,
            )
            min_dist = float(signed_dists.min().item())
        except (RuntimeError, AttributeError):
            # Fallback: unsigned distance if normals unavailable
            min_dist = float(min_dists.min().item())

        return contact_count, link_counts_int, min_dist

    def stabilize_grasp_by_surface_contact(
        self,
        start_pose_6d,
        target_qpos,
        approach_dir=None,
        obj_urdf_path=None,
        surface_contact_thresh=0.015,
        min_contact_points=60,
        required_contact_links=None,
        min_points_per_link=5,
        settle_steps=8,
        max_iters=12,
        push_step=0.002,
    ):
        """
        基于“手部表面点 -> 物体表面点”的距离阈值生成二值接触标签，并在拉动前做稳定接触。

        策略：
        - 固定手腕姿态+目标手指 qpos，让物理引擎先结算若干帧；
        - 计算接触点数量与每个手指 link 的接触点数量；
        - 若不满足阈值，则沿 approach_dir 微推入把手，重复以上过程。
        """
        if obj_urdf_path is None:
            obj_urdf_path = self._get_current_arti_obj_urdf_path()
        if obj_urdf_path is None:
            return np.array(start_pose_6d), False, {"reason": "no_obj_urdf"}

        pose = np.array(start_pose_6d, dtype=np.float32).copy()
        target_qpos_tensor = torch.tensor(target_qpos, dtype=torch.float32, device=self.device)

        best_pose = pose.copy()
        best_count = -1
        best_info = {}

        if required_contact_links is None:
            required_contact_links = []

        for it in range(max_iters):
            root_states = self.root_states.clone()
            for env_i in range(self.num_envs):
                mano_idx = self.mano_actor_idxs[env_i]
                root_states[mano_idx, :3] = torch.tensor(pose[:3], dtype=torch.float32, device=self.device)
                root_states[mano_idx, 3:7] = torch.tensor(pose[3:7], dtype=torch.float32, device=self.device)
                root_states[mano_idx, 7:13] = 0.0
            self._set_mano_root_state_tensor(root_states)

            pos_action = self.dof_pos.squeeze(-1).clone()
            pos_action[:, : self.mano_num_dofs] = target_qpos_tensor
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))

            self.run_steps(pre_steps=settle_steps, refresh_obs=True, print_step=False)

            contact_count, link_counts, min_dist = self._compute_surface_contact_summary(
                hand_pose_6d=pose,
                obj_urdf_path=obj_urdf_path,
                surface_contact_thresh=surface_contact_thresh,
            )

            links_in_contact = sum(1 for v in link_counts.values() if v >= min_points_per_link)
            required_links_ok = True
            if len(required_contact_links) > 0:
                required_links_ok = all(
                    link_counts.get(link_name, 0) >= min_points_per_link
                    for link_name in required_contact_links
                )
            stable = (contact_count >= min_contact_points) and required_links_ok

            info = {
                "iter": it,
                "contact_count": contact_count,
                "links_in_contact": links_in_contact,
                "required_links_ok": required_links_ok,
                "min_dist": min_dist,
                "link_counts": link_counts,
            }

            if contact_count > best_count:
                best_count = contact_count
                best_pose = pose.copy()
                best_info = info

            if stable:
                return pose, True, info

            if approach_dir is None:
                break

            pose[:3] = pose[:3] + np.asarray(approach_dir, dtype=np.float32) * float(push_step)

        return best_pose, False, best_info

    def follow_trajectory_and_record(
        self,
        traj_poses,
        target_qpos,
        record_surface_contact=False,
        surface_contact_thresh=0.015,
        min_contact_points=None,
        required_contact_links=None,
        min_points_per_link=3,
        drive_dof_index=None,
        drive_dof_delta_thresh=None,
        set_root_velocities=False,
        max_root_lin_vel=3.0,
        max_root_ang_vel=12.0,
    ):
        """
        按照预计算好的 6D 轨迹列表，逐帧移动手腕并记录数据。
        """
        records_list = []
        target_qpos_tensor = torch.tensor(target_qpos, dtype=torch.float32, device=self.device)
        obj_urdf_path = None
        if record_surface_contact and self.cfgs.get("USE_ARTI", False):
            obj_urdf_path = self._get_current_arti_obj_urdf_path()
            self._ensure_contact_calc(obj_urdf_path=obj_urdf_path)

        if required_contact_links is None:
            required_contact_links = []

        init_drive_dof_val = None
        prev_pos = None
        prev_rot = None
        dt = float(getattr(self, "sim_dt", 1.0 / 60.0))
        for frame_idx, pose in enumerate(traj_poses):
            current_pos = pose[:3]
            current_rot = pose[3:]

            root_states = self.root_states.clone()
            for env_i in range(self.num_envs):
                mano_idx = self.mano_actor_idxs[env_i]
                root_states[mano_idx, :3] = torch.tensor(current_pos, dtype=torch.float32, device=self.device)
                root_states[mano_idx, 3:7] = torch.tensor(current_rot, dtype=torch.float32, device=self.device)
                if set_root_velocities and prev_pos is not None and prev_rot is not None and dt > 0:
                    lin_vel = (np.asarray(current_pos, dtype=np.float32) - np.asarray(prev_pos, dtype=np.float32)) / dt
                    try:
                        rel_rotvec = (R.from_quat(current_rot) * R.from_quat(prev_rot).inv()).as_rotvec()
                        ang_vel = np.asarray(rel_rotvec, dtype=np.float32) / dt
                    except Exception:
                        ang_vel = np.zeros(3, dtype=np.float32)

                    if max_root_lin_vel is not None:
                        lin_speed = float(np.linalg.norm(lin_vel))
                        if lin_speed > float(max_root_lin_vel) > 1e-8:
                            lin_vel = lin_vel / lin_speed * float(max_root_lin_vel)
                    if max_root_ang_vel is not None:
                        ang_speed = float(np.linalg.norm(ang_vel))
                        if ang_speed > float(max_root_ang_vel) > 1e-8:
                            ang_vel = ang_vel / ang_speed * float(max_root_ang_vel)

                    root_states[mano_idx, 7:10] = torch.tensor(lin_vel, dtype=torch.float32, device=self.device)
                    root_states[mano_idx, 10:13] = torch.tensor(ang_vel, dtype=torch.float32, device=self.device)
                else:
                    root_states[mano_idx, 7:13] = 0.0
            self._set_mano_root_state_tensor(root_states)

            pos_action = self.dof_pos.squeeze(-1).clone()
            pos_action[:, :self.mano_num_dofs] = target_qpos_tensor
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))

            self.run_steps(pre_steps=1, refresh_obs=True, print_step=False)

            current_hand_qpos = (
                self.dof_pos[0, : self.mano_num_dofs, 0].detach().cpu().numpy().tolist()
            )
            current_obj_dof = self.dof_pos[
                0, self.mano_num_dofs:self.mano_num_dofs + self.arti_obj_num_dofs, 0
            ].cpu().numpy().tolist()

            record = {
                "frame": frame_idx,
                "hand_pos": current_pos.tolist(),
                "hand_rot": current_rot.tolist(),
                # Store the *actual* qpos from simulation (used later for contact labels).
                "hand_qpos": current_hand_qpos,
                "hand_qpos_target": target_qpos_tensor.detach().cpu().numpy().tolist(),
                "obj_dof": current_obj_dof
            }

            if drive_dof_index is not None and isinstance(drive_dof_index, (int, np.integer)):
                if 0 <= int(drive_dof_index) < len(current_obj_dof):
                    drive_val = float(current_obj_dof[int(drive_dof_index)])
                    if init_drive_dof_val is None:
                        init_drive_dof_val = drive_val
                    drive_delta = drive_val - float(init_drive_dof_val)
                    record["drive_dof_index"] = int(drive_dof_index)
                    record["drive_dof_val"] = float(drive_val)
                    record["drive_dof_delta"] = float(drive_delta)
                    if drive_dof_delta_thresh is not None:
                        moved = abs(drive_delta) >= float(drive_dof_delta_thresh)
                        record["obj_moved"] = bool(moved)

            records_list.append(record)

            if record_surface_contact and obj_urdf_path is not None and hasattr(self, "contact_calc"):
                contact_count, link_counts, min_dist = self._compute_surface_contact_summary(
                    hand_pose_6d=pose,
                    obj_urdf_path=obj_urdf_path,
                    surface_contact_thresh=surface_contact_thresh,
                )
                records_list[-1]["surface_contact_count"] = int(contact_count)
                records_list[-1]["surface_contact_link_counts"] = link_counts
                records_list[-1]["surface_contact_min_dist"] = float(min_dist)
                if min_contact_points is not None or len(required_contact_links) > 0:
                    required_links_ok = True
                    if len(required_contact_links) > 0:
                        required_links_ok = all(
                            link_counts.get(link_name, 0) >= int(min_points_per_link)
                            for link_name in required_contact_links
                        )
                    stable = (contact_count >= int(min_contact_points or 0)) and required_links_ok
                    records_list[-1]["surface_contact_stable"] = bool(stable)
                    records_list[-1]["surface_required_links_ok"] = bool(required_links_ok)
                    if "obj_moved" in records_list[-1]:
                        records_list[-1]["can_drive_object"] = bool(stable and records_list[-1]["obj_moved"])

            prev_pos = current_pos
            prev_rot = current_rot

        return records_list
    
    def closed_loop_interactive_open(self, start_pose_6d, target_qpos, world_origin, world_axis, joint_type, target_amount, obj_dof_index, steps=100):
        """
        闭环牵引控制：根据物体的实时物理反馈来更新手腕位置
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        import time
        
        records_list = []
        
        # 拆解初始手部姿态
        hand_pos_init = start_pose_6d[:3]
        hand_rot_init = R.from_quat(start_pose_6d[3:7])
        
        # 每一帧的超前引导量（类似弹簧被拉伸的长度）
        # 保证正/负目标都能平滑推进
        lead_delta = abs(target_amount / steps) * 1.5 
        
        for step_i in range(steps):
            # 1. 【核心】获取物体的真实物理状态
            # 读取柜门/抽屉当前的真实位置或角度
            current_obj_dof = self.dof_pos[0, self.mano_num_dofs + obj_dof_index, 0].item()
            
            # 2. 计算引导目标（永远比真实状态超前一点点，形成牵引力）
            # 注意不要超过最终的 target_amount；兼容正向/反向旋转
            if target_amount >= current_obj_dof:
                guided_amount = min(current_obj_dof + lead_delta, target_amount)
            else:
                guided_amount = max(current_obj_dof - lead_delta, target_amount)
            
            # 3. 实时计算手腕此刻应该在的绝对正确位置
            if joint_type in ['revolute', 'continuous']:
                # 完美的圆弧旋转几何计算
                rot_vec = world_axis * guided_amount
                delta_R = R.from_rotvec(rot_vec)
                vec_to_hand = hand_pos_init - world_origin
                
                target_pos = world_origin + delta_R.apply(vec_to_hand)
                target_rot = (delta_R * hand_rot_init).as_quat()
                
            elif joint_type == 'prismatic':
                # 直线平移计算
                target_pos = hand_pos_init + world_axis * guided_amount
                target_rot = start_pose_6d[3:7] # 姿态不变
            else:
                target_pos = hand_pos_init
                target_rot = start_pose_6d[3:7]
                
            # 4. 执行手腕的传送与手指的抓紧
            root_states = self.root_states.clone()
            for env_i in range(self.num_envs):
                mano_idx = self.mano_actor_idxs[env_i]
                root_states[mano_idx, :3] = torch.tensor(target_pos, dtype=torch.float32, device=self.device)
                root_states[mano_idx, 3:7] = torch.tensor(target_rot, dtype=torch.float32, device=self.device)
                root_states[mano_idx, 7:13] = 0.0 # 清除速度，避免物理引擎过度补偿
                
            self._set_mano_root_state_tensor(root_states)
            
            # 保持手指闭合
            pos_action = self.dof_pos.squeeze(-1).clone() 
            pos_action[:, :self.mano_num_dofs] = torch.tensor(target_qpos, dtype=torch.float32, device=self.device)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            
            # 5. 步进物理仿真
            self.run_steps(pre_steps=1, refresh_obs=True, print_step=False)
            
            # 6. 记录高仿真数据
            current_hand_qpos = (
                self.dof_pos[0, : self.mano_num_dofs, 0].detach().cpu().numpy().tolist()
            )
            current_obj_dof_all = self.dof_pos[0, self.mano_num_dofs : self.mano_num_dofs + self.arti_obj_num_dofs, 0].cpu().numpy().tolist()
            records_list.append({
                "frame": step_i,
                "hand_pos": target_pos.tolist(),
                "hand_rot": target_rot.tolist(),
                "hand_qpos": current_hand_qpos,
                "hand_qpos_target": np.asarray(target_qpos).tolist(),
                "obj_dof": current_obj_dof_all
            })
            
            # time.sleep(0.01) # 肉眼观察调试用
            
        return records_list


    def process_and_save_dataset(self, records_list, obj_world_pos, obj_world_rot, obj_urdf_path, save_path):
        """
        在轨迹录制完成后，将整个 records_list 送入 GPU 瞬间计算接触标签并保存。
        """
        # 初始化/复用接触计算器（按 URDF 路径区分，避免对象切换时复用错）
        self._ensure_contact_calc(obj_urdf_path=obj_urdf_path, points_per_link=1000, hand_points_per_link=200)

        B = len(records_list)
        
        # 批量打包张量
        h_pos = torch.tensor([r["hand_pos"] for r in records_list], dtype=torch.float32, device=self.device)
        h_rot = torch.tensor([r["hand_rot"] for r in records_list], dtype=torch.float32, device=self.device)
        h_qpos = torch.tensor([r["hand_qpos"] for r in records_list], dtype=torch.float32, device=self.device)
        
        o_pos = torch.tensor(obj_world_pos, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(B, 1)
        o_rot = torch.tensor(obj_world_rot, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(B, 1)
        o_qpos = torch.tensor([r["obj_dof"] for r in records_list], dtype=torch.float32, device=self.device)
        
        labels, dists = self.contact_calc.compute_batch_contact(h_pos, h_rot, h_qpos, o_pos, o_rot, o_qpos)
        
        # 将 GPU 张量转回 CPU 列表
        labels_np = labels.cpu().numpy().tolist()
        dists_np = dists.cpu().numpy().tolist()
        
        # 将接触信息写回字典
        for i in range(B):
            records_list[i]["contact_labels"] = labels_np[i]
            records_list[i]["contact_dists"] = dists_np[i]
            
        # 写入 JSON
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(records_list, f)
        print(f" 成功保存包含高精度接触标签的轨迹集: {save_path} (共 {B} 帧)")
        
    
    # not used
    def move_obj_to_pose(self, position, quaternion = None):
        
        root_states = self.root_states.clone()
        root_states[-1, :3] = torch.tensor(position, dtype=torch.float32, device=self.device)
        if quaternion is not None:
            root_states[-1, 3:7] = torch.tensor(quaternion, dtype=torch.float32, device=self.device)
        # self.rb_states[:, self.actor_id, :7] = target_pose
        root_reset_actors_indices = torch.unique(torch.tensor(np.arange(root_states.shape[0]), dtype=torch.float32, device=self.device)).to(dtype=torch.int32)
        res = self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(root_reset_actors_indices),len(root_reset_actors_indices))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        assert res == True
        self.run_steps(1)
        if False:
            points_envs, colors_envs, rgb_envs, depth_envs, seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation()
            cv2.imwrite(f"test.png", rgb_envs[0][0])

    # not used
    def add_obj_to_env(self, urdf_path, obj_pose_p, final_rotation):
        obj_asset_file = urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = False
        
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000000
        self.num_asset_per_env+=1
        self.obj_assets.append(self.gym.load_asset(self.sim, self.asset_root, obj_asset_file, asset_options))
        self.obj_num_links+=1
        
        for i in range(self.num_envs):
            env = self.envs[i]
            initial_pose = gymapi.Transform()
            initial_pose.p.x = obj_pose_p[0]
            initial_pose.p.y = obj_pose_p[1]
            initial_pose.p.z = obj_pose_p[2]
            rotation_noise = 0.0
            initial_pose.r.x = final_rotation[0]
            initial_pose.r.y = final_rotation[1]
            initial_pose.r.z = final_rotation[2]
            initial_pose.r.w = final_rotation[3]

            self.init_obj_pos_list[i].append([initial_pose.p.x, initial_pose.p.y, initial_pose.p.z])
            self.init_obj_rot_list[i].append([initial_pose.r.x, initial_pose.r.y, initial_pose.r.z, initial_pose.r.w])
            
            added_obj_actor_handle = self.gym.create_actor(env, self.obj_assets[-1], initial_pose, 'added_actor', i, 1, 0)
            
            obj_actor_idx = self.gym.get_actor_rigid_body_index(env, added_obj_actor_handle, 0, gymapi.DOMAIN_SIM)
            self.obj_actor_idxs[i].append(obj_actor_idx)
            self.gym.set_actor_scale(env, added_obj_actor_handle, self.cfgs["obj_scale"])
        self.gym.prepare_sim(self.sim)
        self.init_observation()

    def run_steps(self, pre_steps = 100, refresh_obs = True, refresh_visual_obs = False, print_step = False):
        # simulation loop
        for frame in range(pre_steps):
            if print_step:
                print("Step: ", frame)
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if refresh_obs:
                self.refresh_observation(get_visual_obs=refresh_visual_obs)
            
            # update viewer
            self.gym.step_graphics(self.sim)
            if not self.headless:
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
        self.refresh_observation(get_visual_obs=refresh_visual_obs)
    
    def clean_up(self):
        # cleanup
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
