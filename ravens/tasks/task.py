# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base Task class."""

import collections
import os
import random
import string
import tempfile

import cv2
import numpy as np
from ravens.tasks import cameras
from ravens.tasks import planners
from ravens.tasks import primitives
from ravens.tasks.grippers import Suction
from ravens.utils import utils

import pybullet as p


class Task():
  """Base Task class."""

  # Class variables.
  rolls = [0]
  pitchs = [0]

  @classmethod
  def get_rolls(cls):
      return cls.rolls

  @classmethod
  def get_pitchs(cls):
      return cls.pitchs

  def __init__(self, continuous = False):
    """Constructor.

    Args:
      continuous: Set to `True` if you want the continuous variant.
    """

    self.continuous = continuous
    self.ee = Suction
    self.mode = 'train'
    self.sixdof = False
    if continuous:
      self.primitive = primitives.PickPlaceContinuous()
    else:
      self.primitive = primitives.PickPlace()
    self.oracle_cams = cameras.Oracle.CONFIG

    # Evaluation epsilons (for pose evaluation metric).
    self.pos_eps = 0.01
    self.rot_eps = np.deg2rad(15)

    # Whether we need to match roll and pitch.
    self.match_rp = False

    # Workspace bounds.
    self.pix_size = 0.003125
    self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])

    self.goals = []
    self.progress = 0
    self._rewards = 0

    self.assets_root = None

  def reset(self, env):  # pylint: disable=unused-argument
    if not self.assets_root:
      raise ValueError('assets_root must be set for task, '
                       'call set_assets_root().')
    self.goals = []
    self.progress = 0  # Task progression metric in range [0, 1].
    self._rewards = 0  # Cumulative returned rewards.
    if self.continuous:
      self.primitive.reset()

  #-------------------------------------------------------------------------
  # Oracle Agent
  #-------------------------------------------------------------------------

  def oracle(self, env, **kwargs):
    """Oracle agent."""
    if self.continuous:
      return self._continuous_oracle(env, **kwargs)
    return self._discrete_oracle(env)

  def _continuous_oracle(self, env, **kwargs):
    """Continuous oracle agent.

    This oracle will generate the pick and place poses using the original
    discrete oracle. It will then interpolate intermediate actions using
    splines.

    Args:
      env: The environment instance.
      **kwargs: extra kwargs for the oracle.
    Returns:
      ContinuousOracle.
    """
    kwargs['env'] = env
    kwargs['base_oracle_cls'] = self._discrete_oracle
    kwargs['ee'] = self.ee
    return ContinuousOracle(**kwargs)

  def _discrete_oracle(self, env):
    """Discrete oracle agent."""
    OracleAgent = collections.namedtuple('OracleAgent', ['act'])

    def act(obs, info):  # pylint: disable=unused-argument
      """Calculate action."""

      # Oracle uses perfect RGB-D orthographic images and segmentation masks.
      _, hmap, obj_mask = self.get_true_image(env)

      # Unpack next goal step.
      objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]

      # Match objects to targets without replacement.
      if not replace:

        # Modify a copy of the match matrix.
        matches = matches.copy()

        # Ignore already matched objects.
        for i in range(len(objs)):
          object_id, (symmetry, _) = objs[i]
          pose = p.getBasePositionAndOrientation(object_id)
          targets_i = np.argwhere(matches[i, :]).reshape(-1)
          for j in targets_i:
            if self.is_match(pose, targs[j], symmetry):
              # print('{}th object is matched with {}th target'.format(i, j))
              matches[i, :] = 0
              matches[:, j] = 0

      # print('matches:\n', matches)

      # Get objects to be picked (prioritize farthest from nearest neighbor).
      nn_dists = []
      nn_targets = []
      for i in range(len(objs)):
        object_id, (symmetry, _) = objs[i]
        xyz, _ = p.getBasePositionAndOrientation(object_id)
        # print("object_id", object_id, "xyz", xyz)
        targets_i = np.argwhere(matches[i, :]).reshape(-1)
        if len(targets_i) > 0:  # pylint: disable=g-explicit-length-test
          targets_xyz = np.float32([targs[j][0] for j in targets_i])
          dists = np.linalg.norm(
              targets_xyz - np.float32(xyz).reshape(1, 3), axis=1)
          nn = np.argmin(dists)
          nn_dists.append(dists[nn])
          nn_targets.append(targets_i[nn])

        # Handle ignored objects.
        else:
          nn_dists.append(0)
          nn_targets.append(-1)
      order = np.argsort(nn_dists)[::-1]

      # Filter out matched objects.
      order = [i for i in order if nn_dists[i] > 0]

      ########################################
      # do not prioritize farthest from nearest neighbor, follow the order of 'objs'
      ########################################
      order = [i for i in range(len(objs)) if i in order]
      # print('order', order, 'from length of objs', len(objs))

      pick_mask = None
      for pick_i in order:
        pick_mask = np.uint8(obj_mask == objs[pick_i][0])

        # save pick_mask into binary image
        # cv2.imwrite('/home/max/Desktop/mira/pick_mask_{}.png'.format(pick_i), pick_mask * 255)

        # Erode to avoid picking on edges.
        # pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

        if np.sum(pick_mask) > 0:
          break

      # print('pick_mask', pick_mask.shape, np.sum(pick_mask))

      # Trigger task reset if no object is visible.
      if pick_mask is None or np.sum(pick_mask) == 0:
        self.goals = []
        print('Object for pick is not visible. Skipping demonstration.')
        return

      # Get picking pose.
      pick_prob = np.float32(pick_mask)
      pick_pix = utils.sample_distribution(pick_prob)
      # For "deterministic" demonstrations on insertion-easy, use this:
      # pick_pix = (160,80)
      pick_pos = utils.pix_to_xyz(pick_pix, hmap,
                                  self.bounds, self.pix_size)
      pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

      # Get placing pose.
      targ_pose = targs[nn_targets[pick_i]]  # pylint: disable=undefined-loop-variable
      obj_pose = p.getBasePositionAndOrientation(objs[pick_i][0])  # pylint: disable=undefined-loop-variable
      if not self.sixdof:
        obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
        obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
        obj_pose = (obj_pose[0], obj_quat)

        targ_euler = utils.quatXYZW_to_eulerXYZ(targ_pose[1])
        targ_quat = utils.eulerXYZ_to_quatXYZW((0, 0, targ_euler[2]))
        targ_pose = (targ_pose[0], targ_quat)
      world_to_pick = utils.invert(pick_pose)
      obj_to_pick = utils.multiply(world_to_pick, obj_pose)
      pick_to_obj = utils.invert(obj_to_pick)
      place_pose = utils.multiply(targ_pose, pick_to_obj)

      # Rotate end effector?
      if not rotations:
        place_pose = (place_pose[0], (0, 0, 0, 1))

      place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))

      return {'pose0': pick_pose, 'pose1': place_pose}

    return OracleAgent(act)

  #-------------------------------------------------------------------------
  # Reward Function and Task Completion Metrics
  #-------------------------------------------------------------------------

  def reward(self):
    """Get delta rewards for current timestep.

    Returns:
      A tuple consisting of the scalar (delta) reward, plus `extras`
        dict which has extra task-dependent info from the process of
        computing rewards that gives us finer-grained details. Use
        `extras` for further data analysis.
    """
    reward, info = 0, {}

    if self.goals:
      # Unpack next goal step.
      objs, matches, targs, _, _, metric, params, max_reward = self.goals[0]

      # Evaluate by matching object poses.
      if metric == 'pose':
        step_reward = 0
        for i in range(len(objs)):
          object_id, (symmetry, _) = objs[i]
          pose = p.getBasePositionAndOrientation(object_id)
          targets_i = np.argwhere(matches[i, :]).reshape(-1)
          for j in targets_i:
            target_pose = targs[j]
            if self.is_match(pose, target_pose, symmetry):
              step_reward += max_reward / len(objs)
              break

      # Evaluate by measuring object intersection with zone.
      elif metric == 'zone':
        zone_pts, total_pts = 0, 0
        obj_pts, zones = params
        for zone_pose, zone_size in zones:

          # Count valid points in zone.
          for obj_id in obj_pts:
            pts = obj_pts[obj_id]
            obj_pose = p.getBasePositionAndOrientation(obj_id)
            world_to_zone = utils.invert(zone_pose)
            obj_to_zone = utils.multiply(world_to_zone, obj_pose)
            pts = np.float32(utils.apply(obj_to_zone, pts))
            if len(zone_size) > 1:
              valid_pts = np.logical_and.reduce([
                  pts[0, :] > -zone_size[0] / 2, pts[0, :] < zone_size[0] / 2,
                  pts[1, :] > -zone_size[1] / 2, pts[1, :] < zone_size[1] / 2,
                  pts[2, :] < self.bounds[2, 1]])

            zone_pts += np.sum(np.float32(valid_pts))
            total_pts += pts.shape[1]
        step_reward = max_reward * (zone_pts / total_pts)

      # Get cumulative rewards and return delta.
      reward = self.progress + step_reward - self._rewards
      self._rewards = self.progress + step_reward

      # Move to next goal step if current goal step is complete.
      if np.abs(max_reward - step_reward) < 0.01:
        self.progress += max_reward  # Update task progress.
        self.goals.pop(0)

    else:
      # At this point we are done with the task but executing the last movements
      # in the plan. We should return 0 reward to prevent the total reward from
      # exceeding 1.0.
      reward = 0.0

    return reward, info

  def done(self):
    """Check if the task is done or has failed.

    Returns:
      True if the episode should be considered a success, which we
        use for measuring successes, which is particularly helpful for tasks
        where one may get successes on the very last time step, e.g., getting
        the cloth coverage threshold on the last alllowed action.
        However, for bag-items-easy and bag-items-hard (which use the
        'bag-items' metric), it may be necessary to filter out demos that did
        not attain sufficiently high reward in external code. Currently, this
        is done in `main.py` and its ignore_this_demo() method.
    """

    # # For tasks with self.metric == 'pose'.
    # if hasattr(self, 'goal'):
    # goal_done = len(self.goal['steps']) == 0  # pylint:
    # disable=g-explicit-length-test
    return (len(self.goals) == 0) or (self._rewards > 0.99)  # pylint: disable=g-explicit-length-test
    # return zone_done or defs_done or goal_done

  #-------------------------------------------------------------------------
  # Environment Helper Functions
  #-------------------------------------------------------------------------

  def is_match(self, pose0, pose1, symmetry):
    """Check if pose0 and pose1 match within a threshold."""

    # Get translational error.
    diff_pos = np.float32(pose0[0][:2]) - np.float32(pose1[0][:2])
    dist_pos = np.linalg.norm(diff_pos)

    ########################################
    # consider height error
    diff_height = np.float32(pose0[0][2]) - np.float32(pose1[0][2])
    dist_height = np.abs(diff_height)
    height_is_close = (dist_height < self.pos_eps)
    ########################################

    # Get rotational error around z-axis (account for symmetries).
    diff_rot = 0
    rot_is_close = (diff_rot < self.rot_eps)
    
    if symmetry > 0:
      rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[2]
      rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[2]
      diff_rot = np.abs(rot0 - rot1) % symmetry
      if diff_rot > (symmetry / 2):
        diff_rot = symmetry - diff_rot
      rot_is_close = (diff_rot < self.rot_eps)
    pos_is_close = (dist_pos < self.pos_eps)

    if self.match_rp:
      # Yaw
      if symmetry > 0:
        rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[2]
        rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[2]
        diff_rot_y = np.abs(rot0 - rot1) % symmetry
        if diff_rot_y > (symmetry / 2):
          diff_rot_y = symmetry - diff_rot
      else:
        diff_rot_y = 0

      # Roll, Pitch
      rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[:2]
      rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[:2]
      diff_rot_rp = np.abs(rot0 - rot1)
      
      # Increase `self.pos_eps` and `self.rot_eps` because 
      # hanging-disks-ood shows that disk could hit the floor 
      # and affect the positions.
      # TODO: Change the task to fix this.
      diff_rot = (diff_rot_rp[0], diff_rot_rp[1], diff_rot_y)
      rot_is_close = (diff_rot < self.rot_eps*1.5).all()
      pos_is_close = (dist_pos < self.pos_eps*2)

    
    return pos_is_close and rot_is_close and height_is_close

  def get_true_image(self, env):
    """Get RGB-D orthographic heightmaps and segmentation masks."""

    # Capture near-orthographic RGB-D images and segmentation masks.
    color, depth, segm = env.render_camera(self.oracle_cams[0])

    # save color image
    # color_cv2 = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('/home/max/Desktop/mira/color.png', color_cv2)

    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = utils.reconstruct_heightmaps(
        [color], [depth], self.oracle_cams, self.bounds, self.pix_size)

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
    return cmap, hmap, mask

  def get_random_pose(self, env, obj_size, constraints = False):
    """Get random collision-free object pose within workspace bounds."""

    # Get erosion size of object in pixels.
    max_size = np.sqrt(obj_size[0]**2 + obj_size[1]**2)
    erode_size = int(np.round(max_size / self.pix_size))

    _, hmap, obj_mask = self.get_true_image(env)

    # Randomly sample an object pose within free-space pixels.
    free = np.ones(obj_mask.shape, dtype=np.uint8)

    # for 'stacking-boxes' task, we need to avoid sampling on the right side of the table
    if constraints:
      # free image is 320x160
      free[160:, :] = 0

    for obj_ids in env.obj_ids.values():
      for obj_id in obj_ids:
        free[obj_mask == obj_id] = 0
    free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
    free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
    if np.sum(free) == 0:
      return None, None
    pix = utils.sample_distribution(np.float32(free))
    pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
    pos = (pos[0], pos[1], obj_size[2] / 2)
    theta = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
    return pos, rot

  #-------------------------------------------------------------------------
  # Helper Functions
  #-------------------------------------------------------------------------

  def fill_template(self, template, replace):
    """Read a file and replace key strings."""
    full_template_path = os.path.join(self.assets_root, template)
    with open(full_template_path, 'r') as file:
      fdata = file.read()
    for field in replace:
      for i in range(len(replace[field])):
        fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
    alphabet = string.ascii_lowercase + string.digits
    rname = ''.join(random.choices(alphabet, k=16))
    tmpdir = tempfile.gettempdir()
    template_filename = os.path.split(template)[-1]
    fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
    with open(fname, 'w') as file:
      file.write(fdata)
    return fname

  def get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
    """Get random box size."""
    size = np.random.rand(3)
    size[0] = size[0] * (max_x - min_x) + min_x
    size[1] = size[1] * (max_y - min_y) + min_y
    size[2] = size[2] * (max_z - min_z) + min_z
    return tuple(size)

  def get_box_object_points(self, obj):
    obj_shape = p.getVisualShapeData(obj)
    obj_dim = obj_shape[0][3]
    obj_dim = tuple(d for d in obj_dim)
    xv, yv, zv = np.meshgrid(
      np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
      np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
      np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
      sparse=False, indexing='xy')
    return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

  def get_mesh_object_points(self, obj):
    mesh = p.getMeshData(obj)
    mesh_points = np.array(mesh[1])
    mesh_dim = np.vstack((mesh_points.min(axis=0), mesh_points.max(axis=0)))
    xv, yv, zv = np.meshgrid(
          np.arange(mesh_dim[0][0], mesh_dim[1][0], 0.02),
          np.arange(mesh_dim[0][1], mesh_dim[1][1], 0.02),
          np.arange(mesh_dim[0][2], mesh_dim[1][2], 0.02),
          sparse=False, indexing='xy')
    return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

  def get_object_points(self, obj):
    obj_shape = p.getVisualShapeData(obj)
    obj_dim = obj_shape[0][3]
    xv, yv, zv = np.meshgrid(
        np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
        np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
        np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
        sparse=False, indexing='xy')
    return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

  def color_random_brown(self, obj):
    shade = np.random.rand() + 0.5
    color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
    p.changeVisualShape(obj, -1, rgbaColor=color)

  def set_assets_root(self, assets_root):
    self.assets_root = assets_root


class ContinuousOracle:
  """Continuous oracle."""

  def __init__(
      self,
      env,
      base_oracle_cls,
      ee,
      t_max = 10.0,
      steps_per_seg = 2,
      height = 0.32,
  ):
    """Constructor.

    Args:
      env:
      base_oracle_cls:
      ee:
      t_max:
      steps_per_seg:
      height:
    """
    self._env = env
    self._base_oracle = base_oracle_cls(env)
    self.steps_per_seg = steps_per_seg

    planner_cls = planners.PickPlacePlanner if ee == Suction else planners.PushPlanner
    self._planner = planner_cls(steps_per_seg, t_max, height)

    self._actions = []

  def act(self, obs, info):
    """Get oracle action from planner."""
    if not self._actions:
      # Query the base oracle for pick and place poses.
      act = self._base_oracle.act(obs, info)
      if act is None:
        return
      self._actions = self._planner(self._env.get_ee_pose(), act['pose0'],
                                    act['pose1'])
    act = self._actions.pop(0)
    act['acts_left'] = len(self._actions)
    return act

  @property
  def num_poses(self):
    return self._planner.NUM_POSES
