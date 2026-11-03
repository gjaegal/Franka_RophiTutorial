"""Model, LCM wire types, collision primitives and velocity IK shared by both processes."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import osqp
from scipy import sparse

# Reuse the tutorial's generated wire types, without importing its controller stack.
TUTORIAL = Path(__file__).resolve().parents[2] / "RoPhiTutorial"
sys.path.append(str(TUTORIAL))
from lcm_type.collision.col_info_t import col_info_t
from lcm_type.joint.joint_ctrl_t import joint_ctrl_t
from lcm_type.joint.joint_meas_t import joint_meas_t
from lcm_type.pose.pose3d_t import pose3d_t

STATE = "sw_flexiv_arm_joint_meas"
COMMAND = "sw_flexiv_arm_hand_joint_ctrl"
STATIC = "sw_flexiv_arm_hand_static_col_info"
ROBOT = "sw_flexiv_arm_hand_robot_col_info"
TARGET = "sw_ik_target_pose"
# A different port prevents accidental cross-talk with a running Flexiv tutorial.
URL = "udpm://239.255.76.67:7668?ttl=0"
HARDWARE_URL = "udpm://239.255.76.67:7669?ttl=0"
HOME = np.array([0., -.45, 0., -1.85, 0., 1.5, .75])
DT = .01
MARGIN = .04
SPEED = .4


def load_model():
    """Wrap the unmodified, vendored FR3 with an open-top square cell."""
    root = Path(__file__).resolve().parent / "assets/franka_fr3"
    tree = ET.parse(root / "fr3.xml")
    mj = tree.getroot()
    mj.find("compiler").set("meshdir", str(root / "assets"))
    mj.find("option").set("timestep", ".002")
    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "headlight", ambient=".45 .45 .45", diffuse=".8 .8 .8")
    ET.SubElement(mj.find("asset"), "texture", type="skybox", builtin="gradient",
                  rgb1=".55 .65 .8", rgb2=".9 .9 .9", width="512", height="3072")
    # Bias compensation is added once by the simulation environment.
    world = mj.find("worldbody")
    link7 = world.find(".//body[@name='fr3_link7']")
    tool = ET.SubElement(link7, "body", name="index_tip", pos="0 0 0.107")
    ET.SubElement(tool, "site", name="index_tip", pos="0 0 0",
                  size=".012", rgba="0 1 0 1", group="1")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", diffuse=".8 .8 .8")
    # Full inner width = 1.70 m, height = 1.45 m, wall thickness = 4 cm.
    # walls = {
    #     "floor": ([0, 0, -.04], [.89, .89, .04]),
    #     "wall_front": ([.87, 0, .725], [.02, .89, .725]),
    #     "wall_back": ([-.87, 0, .725], [.02, .89, .725]),
    #     "wall_left": ([0, .87, .725], [.85, .02, .725]),
    #     "wall_right": ([0, -.87, .725], [.85, .02, .725]),
    # }
    walls = {
        "floor": ([0, 0, -.04], [.89, .445, .04]),

        "wall_front": ([.87, 0, .725], [.02, .445, .725]),
        "wall_back":  ([-.87, 0, .725], [.02, .445, .725]),

        "wall_left":  ([0, .425, .725], [.85, .02, .725]),
        "wall_right": ([0, -.425, .725], [.85, .02, .725]),
    }
    for name, (pos, size) in walls.items():
        ET.SubElement(world, "geom", name=name, type="box",
                      pos=" ".join(map(str, pos)), size=" ".join(map(str, size)),
                      rgba=".45 .6 .8 .17" if name != "floor" else ".65 .65 .65 1")
    target = ET.SubElement(world, "body", name="keyboard_target", mocap="true", pos=".4 0 .5")
    ET.SubElement(target, "geom", type="sphere", size=".018", rgba="1 0 0 1",
                  contype="0", conaffinity="0", group="1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(mj, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qpos[:] = HOME
    data.ctrl[:] = HOME
    mujoco.mj_forward(model, data)
    data.mocap_pos[0] = data.site("index_tip").xpos
    return model, data


def collision_geometry(model):
    """Conservative link-local bounding spheres; static boxes use world transforms.

    A sphere encloses every vertex of each link's collision mesh. These primitives
    deliberately overestimate the arm; self-collision is left to MuJoCo contacts.
    """
    robot, static = [], []
    for i in range(model.ngeom):
        g = model.geom(i)
        if g.name.startswith("wall_") or g.name == "floor":
            T = np.eye(4)
            T[:3, 3] = g.pos
            static.append((g.name, "box", 2 * g.size.copy(), T))
        elif g.name.endswith("_collision") and g.name != "fr3_link0_collision":
            mid = int(g.dataid[0])
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            vertices = model.mesh_vert[start:start + count]
            center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            radius = np.linalg.norm(vertices - center, axis=1).max()
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, g.quat)
            T = np.eye(4)
            T[:3, 3] = g.pos + rotation.reshape(3, 3) @ center
            body = model.body(int(g.bodyid[0])).name
            robot.append((body, "sphere", np.array([radius]), T))
    # Also protect the task point beyond the wrist mesh.
    T = np.eye(4)
    robot.append(("index_tip", "sphere", np.array([.02]), T))
    return robot, static


def encode_geometry(items, stamp):
    msg = col_info_t()
    msg.timestamp = stamp
    msg.num_col_geoms = len(items)
    msg.concat_names = ",".join(x[0] for x in items)
    msg.concat_geom_types = ",".join(x[1] for x in items)
    msg.list_size_dims = [len(x[2]) for x in items]
    msg.concat_sizes = np.concatenate([x[2] for x in items]).tolist()
    msg.len_concat_sizes = len(msg.concat_sizes)
    msg.list_offsets = [x[3].tolist() for x in items]
    return msg.encode()


def decode_geometry(payload):
    msg = col_info_t.decode(payload)
    sizes = np.split(np.array(msg.concat_sizes), np.cumsum(msg.list_size_dims)[:-1])
    return [(n, k, s, np.array(T)) for n, k, s, T in zip(
        msg.concat_names.split(","), msg.concat_geom_types.split(","), sizes, msg.list_offsets)]


def sphere_box(center, radius, size, T):
    """Signed clearance and outward world normal from an oriented box to a sphere."""
    R, origin = T[:3, :3], T[:3, 3]
    local = R.T @ (center - origin)
    half = size / 2
    delta = local - np.clip(local, -half, half)
    norm = np.linalg.norm(delta)
    if norm > 1e-10:
        return norm - radius, R @ (delta / norm)
    # Sphere center is inside the wall: escape through its closest face.
    gaps = half - np.abs(local)
    axis = np.argmin(gaps)
    normal = np.zeros(3)
    normal[axis] = 1 if local[axis] >= 0 else -1
    return -gaps[axis] - radius, R @ normal


class IK:
    def __init__(self, model, speed=SPEED, cart_speed=.12):
        self.model = model
        self.data = mujoco.MjData(model)
        self.robot = []
        self.walls = []
        self.status = "waiting for collision geometry"
        self.speed = speed
        self.cart_speed = cart_speed

    def set_state(self, q, dq):
        self.data.qpos[:] = q
        self.data.qvel[:] = dq
        mujoco.mj_forward(self.model, self.data)

    def tip(self):
        J = np.zeros((3, 7))
        mujoco.mj_jacSite(self.model, self.data, J, None, self.model.site("index_tip").id)
        return self.data.site("index_tip").xpos.copy(), J

    def collision_rows(self):
        for name, kind, size, offset in self.robot:
            if kind != "sphere":
                raise ValueError("Expected robot spheres")
            body = self.data.body(name)
            point = body.xpos + body.xmat.reshape(3, 3) @ offset[:3, 3]
            J = np.zeros((3, 7))
            mujoco.mj_jac(self.model, self.data, J, None, point, body.id)
            for _, wall_kind, wall_size, T in self.walls:
                if wall_kind != "box":
                    raise ValueError("Expected static boxes")
                distance, normal = sphere_box(point, size[0], wall_size, T)
                yield distance, normal @ J

    def solve(self, target):
        if not self.robot or not self.walls:
            self.status = "waiting for collision geometry"
            return np.zeros(7)
        tip, J = self.tip()
        velocity = 3 * (target - tip) - .1 * (J @ self.data.qvel)
        velocity *= min(1., self.cart_speed / max(np.linalg.norm(velocity), 1e-12))
        # q + horizon*dq must remain inside the joint range, with a small margin.
        limits = self.model.jnt_range
        lower = np.maximum(-self.speed, (limits[:, 0] + .02 - self.data.qpos) / .1)
        upper = np.minimum(self.speed, (limits[:, 1] - .02 - self.data.qpos) / .1)
        A, lo, hi = [*np.eye(7)], list(lower), list(upper)
        for distance, row in self.collision_rows():
            A.append(row)
            lo.append(min(.15, 4 * (MARGIN - distance)))
            hi.append(np.inf)
        A = np.array(A)
        problem = osqp.OSQP()
        problem.setup(P=sparse.csc_matrix(2 * J.T @ J + .02 * np.eye(7)),
                      q=-2 * J.T @ velocity, A=sparse.csc_matrix(A),
                      l=np.array(lo), u=np.array(hi), verbose=False,
                      eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, polishing=False)
        result = problem.solve(raise_error=False)
        self.status = result.info.status
        x = result.x
        # Never forward an infeasibility certificate or an unchecked solution.
        if result.info.status_val != 1 or x is None or not np.isfinite(x).all():
            return np.zeros(7)
        if np.any(A @ x < np.array(lo) - 2e-5) or np.any(A @ x > np.array(hi) + 2e-5):
            self.status = "constraint residual too large"
            return np.zeros(7)
        return x


def drain(bus):
    # Bound work per tick even if another publisher floods the multicast group.
    for _ in range(100):
        if not bus.handle_timeout(0):
            break


def command(q, dq, stamp):
    msg = joint_ctrl_t()
    msg.timestamp, msg.num_joints, msg.valid = stamp, 7, True
    msg.q_des, msg.qd_des = q.tolist(), dq.tolist()
    msg.tau_ff, msg.kp, msg.kd = [0.] * 7, [0.] * 7, [0.] * 7
    return msg.encode()
