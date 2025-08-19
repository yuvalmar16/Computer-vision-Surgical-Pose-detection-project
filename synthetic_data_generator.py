#!/usr/bin/env python3
import blenderproc as bproc
import argparse
import os
import json
import glob
from colorsys import hsv_to_rgb
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import bpy
import numpy as np
import cv2
import h5py
import shutil
import random
import argparse
from PIL import Image


# Configuration 
class Cfg:
    NUM_IMAGES = 1000

    # Resources for tools, background and camera
    RESOURCES_DIR = "/datashare/project/"
    MODELS_DIR = os.path.join(RESOURCES_DIR, "surgical_tools_models/")
    TWEEZERS_DIR = os.path.join(MODELS_DIR, "tweezers/")
    NEEDLE_HOLDER_DIR = os.path.join(MODELS_DIR, "needle_holder/")
    HDRI_DIR = os.path.join(RESOURCES_DIR, "haven/hdris/")
    CAMERA_JSON = os.path.join(RESOURCES_DIR, "camera.json")
    # Output
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_ROOT = os.path.join(CURRENT_DIR, "output_variant/")  



# Utilities 
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def list_objs(folder: str):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".obj")]

def choose_obj_file(folder: str) -> str:
    return random.choice(list_objs(folder))

def load_obj(path: str):
    return bproc.loader.load_obj(path)[0]



# Scene Planning 
class PlanDirector:
    """Produces high-level random decisions for a scene/fram."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def choose_tools(self):
        """Return a list of file paths to instruments to load (1–2 tools)."""
        count = np.random.choice(range(1, 3), p=[0.2, 0.8])
        initial = [choose_obj_file(self.cfg.NEEDLE_HOLDER_DIR), choose_obj_file(self.cfg.TWEEZERS_DIR)]
        paths = []
        for _ in range(count):
            if initial:
                paths.append(initial.pop(random.randrange(len(initial))))
            else:
                folder = random.choice([self.cfg.NEEDLE_HOLDER_DIR, self.cfg.TWEEZERS_DIR])
                paths.append(choose_obj_file(folder))
        return paths

    def spawn_props(self):
        """Return a list of primitive types and their random transforms/materials."""
        count = random.randint(4, 7)  # a touch more clutter
        kinds = ["SPHERE", "CONE", "CYLINDER", "CUBE", "PLANE", "MONKEY"]  # supported set
        items = []
        for _ in range(count):
            items.append({
                "type": random.choice(kinds),
                "loc": [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), random.uniform(0, 2.8)],
                "scale": [random.uniform(0.07, 0.95), random.uniform(0.07, 0.95), random.uniform(0.07, 0.95)],
                "base_color": [random.uniform(0.05, 0.95), random.uniform(0.05, 0.95), random.uniform(0.05, 0.95), 1.0],
                "roughness": random.uniform(0.25, 0.65)
            })
        return items

    def stage_lights(self, anchor_a, anchor_b=None):
        """Produce parameters for 1–3 lights around objects' center."""
        center = anchor_a if anchor_b is None else (anchor_a + anchor_b) / 2
        num_lights = random.randint(1, 3)
        light_types = ["POINT", "SUN", "SPOT", "AREA"]
        lights = []
        for i in range(num_lights):
            # darker colors overall
            color = np.random.uniform([0.35, 0.35, 0.35], [0.9, 0.9, 0.9])
    
            if num_lights == 1:
                lo, hi = 400, 850
            elif i == 0:
                lo, hi = 650, 900
            else:
                lo, hi = 120, 380
            lights.append({
                "type": random.choice(light_types),
                "color": color,
                "energy": random.uniform(lo, hi),
                "pose_center": center
            })
        return lights

    def pick_hdri(self):
        folder = os.path.join(self.cfg.HDRI_DIR, random.choice(os.listdir(self.cfg.HDRI_DIR)))
        name = random.choice(os.listdir(folder))
        return os.path.join(folder, name)



# Scene Builder
class SceneComposer:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def reset_scene(self):
        bproc.utility.reset_keyframes()
        bproc.clean_up()

    def load_tools(self, paths):
        objs = []
        for p in paths:
            obj = load_obj(p)
            obj.set_cp("category_id", 2 if "tweezers" in p else 1)
            self._randomize_tool_materials(obj)
            self._randomize_pose(obj)
            objs.append(obj)
        return objs

    def _randomize_tool_materials(self, obj):
        for mat in obj.get_materials():
            if not any(n.type == "BSDF_PRINCIPLED" for n in mat.nodes):
                continue
            if mat.get_name().lower() == "gold":
                try:
                    hsv = np.random.uniform([0.03, 0.95, 48], [0.25, 1.0, 48])
                    rgba = list(hsv_to_rgb(*hsv)) + [1.0]
                    mat.set_principled_shader_value("Base Color", rgba)
                except AttributeError:
                    pass
            try:
                mat.set_principled_shader_value("Specular IOR Level", random.uniform(0, 1))
                mat.set_principled_shader_value("Roughness", 0.4)
                mat.set_principled_shader_value("Metallic", 1)
            except AttributeError:
                pass

    def _randomize_pose(self, obj):
        obj.set_shading_mode(random.choice(["AUTO", "SMOOTH", "FLAT"]), angle_value=random.uniform(20, 45))
        obj.set_location(np.random.uniform([-2, -2, -2], [2, 2, 2]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi]))
        obj.set_scale(np.random.uniform([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]))

    def add_props(self, clutter_plan):
        for spec in clutter_plan:
            obj = bproc.object.create_primitive(spec["type"])
            obj.set_location(spec["loc"])
            obj.set_scale(spec["scale"])
            mat = bproc.material.create("random_material_variant")
            mat.set_principled_shader_value("Base Color", spec["base_color"])
            mat.set_principled_shader_value("Roughness", spec["roughness"])
            obj.replace_materials(mat)
            obj.set_cp("category_id", 0)  # background

    def apply_hdri(self, hdri_path):
        world = bpy.context.scene.world
        if world.node_tree is None:
            world.use_nodes = True
        bproc.world.set_world_background_hdr_img(hdri_path)
        
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.08, 1.2)

    def setup_camera_from_json(self, camera_json):
        with open(camera_json, "r") as f:
            params = json.load(f)
        fx, fy, cx, cy = params["fx"], params["fy"], params["cx"], params["cy"]
        width, height = params["width"], params["height"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        bproc.camera.set_intrinsics_from_K_matrix(K, width, height)

    def place_lights(self, lights_plan):
        for spec in lights_plan:
            light = bproc.types.Light()
            light.set_type(spec["type"])
            light.set_color(spec["color"])
            light.set_energy(spec["energy"])
            light.set_location(bproc.sampler.shell(
                center=spec["pose_center"],
                radius_min=40, radius_max=60,
                elevation_min=1, elevation_max=90
            ))

    def sample_cam(self, anchors):
       
        obj1_loc = anchors[0].get_location()
        center = obj1_loc
        rmin, rmax = 10, 30
        if len(anchors) >= 2:
            obj2_loc = anchors[1].get_location()
            center = (obj1_loc + obj2_loc) / 2
            dist = np.linalg.norm(obj1_loc - obj2_loc)
            rmax = max(dist * 3, 30)

        cam_pos = bproc.sampler.shell(center=center, radius_min=rmin, radius_max=rmax,
                                      elevation_min=-90, elevation_max=90)
        poi = center + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        rot = bproc.camera.rotation_from_forward_vec(poi - cam_pos,
                                                     inplane_rot=np.random.uniform(-0.7854, 0.7854))
        cam2world = bproc.math.build_transformation_mat(cam_pos, rot)

        visible = bproc.camera.visible_objects(cam2world)
        if (anchors[0] in visible) or (len(anchors) >= 2 and anchors[1] in visible):
            bproc.camera.add_camera_pose(cam2world)
            return True
        return False



# Rendering & Post 
class RenderOrchestrator:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.hdf5_dir = ensure_dir(os.path.join(cfg.OUTPUT_ROOT, "hdf5_format/"))
        self.jpg_dir = ensure_dir(os.path.join(cfg.OUTPUT_ROOT, "jpg_format/"))

    def configure(self):
        bproc.renderer.set_max_amount_of_samples(10)
        bproc.renderer.set_output_format(enable_transparency=False)
        bproc.renderer.enable_segmentation_output(map_by=["category_id"])  
        bproc.renderer.set_denoiser("OPTIX")
        bproc.renderer.set_noise_threshold(0.05)

    def render_and_write(self):
        data = bproc.renderer.render()
        bproc.writer.write_hdf5(self.hdf5_dir, data, append_to_existing_output=True)

    def export_jpgs(self):
        # same export loop
        for i in range(len(os.listdir(self.hdf5_dir))):
            fp = os.path.join(self.hdf5_dir, f"{i}.hdf5")
            colors_out = os.path.join(self.jpg_dir, f"{i}_color.jpg")
            with h5py.File(fp, "r") as f:
                colors = f["colors"][:]
                Image.fromarray(colors, "RGB").save(colors_out, "JPEG")



# Pose Extraction 
class KeypointAnnotator:
    """
    Creates 5-keypoint labels + visualization for each instance via contour geometry.
    Class mapping in the output labels:
        0 = tweezer       (inst['category_id'] == 2)
        1 = needle_holder (inst['category_id'] == 1)
    """

    def __init__(self, jpg_dir: str, hdf5_dir: str):
        self.jpg_dir = jpg_dir
        self.hdf5_dir = hdf5_dir

    @staticmethod
    def _draw_point_box_and_label(image, pt, labels, yolo_cls, img_h, img_w, size=5, color=(0, 0, 255)):
        x, y = pt
        pt1 = (int(x - size), int(y - size))
        pt2 = (int(x + size), int(y + size))
        cv2.rectangle(image, pt1, pt2, color, thickness=1)
        box_w = box_h = size * 2
        # IMPORTANT: class_id is the tool class (0/1), not the keypoint type
        labels.append(f"{yolo_cls} {x / img_w:.6f} {y / img_h:.6f} {box_w / img_w:.6f} {box_h / img_h:.6f}")

    def run(self, num_images: int):
        for i in range(num_images):
            hdf5_path = os.path.join(self.hdf5_dir, f"{i}.hdf5")
            color_path = os.path.join(self.jpg_dir, f"{i}_color.jpg")
            out_img = os.path.join(self.jpg_dir, f"{i}_pose_keypoints.jpg")
            out_lbl = os.path.join(self.jpg_dir, f"{i}_pose_keypoints.txt")

            if not (os.path.exists(hdf5_path) and os.path.exists(color_path)):
                print(f"Skipping {i} – missing files")
                continue

            img = cv2.imread(color_path)
            if img is None:
                print(f"Skipping {i} – cannot read {color_path}")
                continue
            img_h, img_w = img.shape[:2]
            labels = []

            with h5py.File(hdf5_path, "r") as f:
                inst_map = f["category_id_segmaps"][:]
                meta_raw = f["instance_attribute_maps"][()]
                meta = json.loads(meta_raw.decode("utf-8"))

            for inst in meta:
                idx = inst["idx"]
                cat = inst["category_id"]  # 1 = needle_holder, 2 = tweezers (as set in load_tools)

                # Map BlenderProc category -> YOLO class id
                # 0 = tweezer, 1 = needle holder
                if cat == 2:
                    yolo_cls = 0
                elif cat == 1:
                    yolo_cls = 1
                else:
                    continue  # skip background/others

                mask = (inst_map == idx).astype(np.uint8)
                if mask.sum() == 0:
                    continue

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)
                if len(cnt) < 2:
                    continue

                pts = cnt.squeeze()
                if pts.ndim != 2:
                    continue

                # ----- Derive the 5 keypoints -----
                # 1-2) Two farthest points (principal axis)
                from scipy.spatial.distance import cdist
                dist_mat = cdist(pts, pts)
                i1, j1 = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
                p1, p2 = pts[i1], pts[j1]

                # 3) Lateral extreme via perpendicular projection
                main_vec = p2 - p1
                norm = np.linalg.norm(main_vec)
                if norm < 1e-6:
                    continue
                perp = np.array([-main_vec[1], main_vec[0]]) / norm
                proj = pts @ perp
                i_min, i_max = np.argmin(proj), np.argmax(proj)
                p3, p4 = pts[i_min], pts[i_max]
                d3 = min(np.linalg.norm(p3 - p1), np.linalg.norm(p3 - p2))
                d4 = min(np.linalg.norm(p4 - p1), np.linalg.norm(p4 - p2))
                lateral = p3 if d3 >= d4 else p4

                # 4-5) Two internal cluster centers
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(pts)
                    centers = [tuple(np.round(c).astype(int)) for c in kmeans.cluster_centers_]
                except Exception:
                    centers = []

                # ----- Draw & record labels (ALL use the tool class yolo_cls) -----
                self._draw_point_box_and_label(img, p1, labels, yolo_cls, img_h, img_w, color=(0, 0, 255))
                self._draw_point_box_and_label(img, p2, labels, yolo_cls, img_h, img_w, color=(0, 0, 255))
                self._draw_point_box_and_label(img, lateral, labels, yolo_cls, img_h, img_w, color=(0, 0, 255))

                for c in centers[:2]:
                    self._draw_point_box_and_label(img, c, labels, yolo_cls, img_h, img_w, color=(0, 255, 255))

            cv2.imwrite(out_img, img)
            with open(out_lbl, "w") as f:
                f.write("\n".join(labels))
            print(f"Pose saved: {out_img}, {out_lbl}")


# YOLO Dataset Builder 
class YoloPosePackager:
    def __init__(self, jpg_dir: str, out_root: str):
        self.jpg_dir = jpg_dir
        self.out_root = out_root
        self.train_img = ensure_dir(os.path.join(out_root, "train/images"))
        self.train_lbl = ensure_dir(os.path.join(out_root, "train/labels"))
        self.val_img = ensure_dir(os.path.join(out_root, "val/images"))
        self.val_lbl = ensure_dir(os.path.join(out_root, "val/labels"))

    @staticmethod
    def _compose_yolo_pose_lines(label_path):
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        objs = []
        for i in range(0, len(lines), 5):
            group = lines[i:i + 5]
            if len(group) < 5:
                continue
            xs, ys, kplist = [], [], []
            class_id = int(group[0].split()[0])
            for kp_line in group:
                parts = list(map(float, kp_line.split()))
                x, y = parts[1], parts[2]
                v = 2
                xs.append(x); ys.append(y)
                kplist.extend([f"{x:.6f}", f"{y:.6f}", str(v)])
            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
            w, h = x_max - x_min, y_max - y_min
            objs.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} " + " ".join(kplist))
        return objs

    def build(self):
        pose_txts = [f for f in os.listdir(self.jpg_dir) if f.endswith("_pose_keypoints.txt")]
        random.shuffle(pose_txts)
        split = int(len(pose_txts) * 0.8)
        train, val = pose_txts[:split], pose_txts[split:]

        def process(files, img_dst, lbl_dst):
            for fname in files:
                stem = fname.replace("_pose_keypoints.txt", "")
                src_img = os.path.join(self.jpg_dir, f"{stem}_color.jpg")
                src_lbl = os.path.join(self.jpg_dir, fname)
                if not os.path.exists(src_img):
                    print(f"Image missing: {src_img}")
                    continue
                shutil.copy(src_img, os.path.join(img_dst, f"{stem}.jpg"))
                out_lbl = os.path.join(lbl_dst, f"{stem}.txt")
                lines = self._compose_yolo_pose_lines(src_lbl)
                with open(out_lbl, "w") as f:
                    f.write("\n".join(lines))
                print(f"YOLO sample: {stem}")

        def filter_invalid(images_dir, labels_dir, expected_cols=20):
            removed = kept = 0
            for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
                with open(label_file, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                ok = all(len(line.split()) == expected_cols for line in lines)
                stem = os.path.splitext(os.path.basename(label_file))[0]
                if not ok:
                    for ext in [".jpg", ".png"]:
                        p = os.path.join(images_dir, stem + ext)
                        if os.path.exists(p):
                            os.remove(p)
                    os.remove(label_file)
                    removed += 1
                    print(f"Removed corrupt sample: {stem}")
                else:
                    kept += 1
            print(f"Filtering done: kept {kept}, removed {removed}")

        process(train, self.train_img, self.train_lbl)
        process(val, self.val_img, self.val_lbl)
        filter_invalid(self.train_img, self.train_lbl)
        filter_invalid(self.val_img, self.val_lbl)
        print(f"Dataset split complete: Train={len(train)}  Val={len(val)}")


def get_next_index(hdf5_dir):
    """Return the next index after the largest existing hdf5 file in the folder."""
    files = [f for f in os.listdir(hdf5_dir) if f.endswith(".hdf5")]
    if not files:
        return 0
    nums = [int(os.path.splitext(f)[0]) for f in files if os.path.splitext(f)[0].isdigit()]
    return max(nums) + 1 if nums else 0


class RenderOrchestrator:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.hdf5_dir = ensure_dir(os.path.join(cfg.OUTPUT_ROOT, "hdf5_format/"))
        self.jpg_dir = ensure_dir(os.path.join(cfg.OUTPUT_ROOT, "jpg_format/"))

    def configure(self):
        bproc.renderer.set_max_amount_of_samples(10)
        bproc.renderer.set_output_format(enable_transparency=False)
        bproc.renderer.enable_segmentation_output(map_by=["category_id"])
        bproc.renderer.set_denoiser("OPTIX")
        bproc.renderer.set_noise_threshold(0.05)

    def render_and_write(self, start_index):
        data = bproc.renderer.render()
        bproc.writer.write_hdf5(self.hdf5_dir, data, append_to_existing_output=True)


    def export_jpgs(self):
        """Export JPGs for every HDF5 in the folder."""
        files = sorted([f for f in os.listdir(self.hdf5_dir) if f.endswith(".hdf5")])
        for fname in files:
            stem = os.path.splitext(fname)[0]
            fp = os.path.join(self.hdf5_dir, fname)
            colors_out = os.path.join(self.jpg_dir, f"{stem}_color.jpg")
            with h5py.File(fp, "r") as f:
                colors = f["colors"][:]
                Image.fromarray(colors, "RGB").save(colors_out, "JPEG")
            print(f"Exported {colors_out}")


def run_variant(args):
    # Init
    bproc.init()
    ensure_dir(Cfg.OUTPUT_ROOT)

    renderer = RenderOrchestrator(Cfg)
    hdf5_dir = renderer.hdf5_dir
    jpg_dir = renderer.jpg_dir


    start_index = get_next_index(hdf5_dir)
    print(f"Starting rendering from index {start_index}")

    planner = PlanDirector(Cfg)
    builder = SceneComposer(Cfg)

    trials, success = 0, 0
    while (trials < 10000) and (success < args.num_images):
        builder.reset_scene()

        # Scene planning
        instrument_paths = planner.choose_tools()
        instruments = builder.load_tools(instrument_paths)

        # Clutter & lights
        builder.add_props(planner.spawn_props())
        builder.setup_camera_from_json(Cfg.CAMERA_JSON)
        builder.apply_hdri(planner.pick_hdri())
        anchor_a = instruments[0].get_location()
        anchor_b = instruments[1].get_location() if len(instruments) > 1 else None
        lights_plan = planner.stage_lights(anchor_a, anchor_b)
        builder.place_lights(lights_plan)

        # Camera sampling
        accepted = False
        while not accepted:
            trials += 1
            accepted = builder.sample_cam(instruments)
        success += 1

        # Render
        renderer.configure()
        renderer.render_and_write(start_index + success - 1)

    # Export jpgs
    renderer.export_jpgs()

    # Pose extraction for all HDF5
    num_files = len([f for f in os.listdir(hdf5_dir) if f.endswith(".hdf5")])
    KeypointAnnotator(jpg_dir, hdf5_dir).run(num_files)

    # YOLO dataset
    yolo_root = ensure_dir("/home/student/SYNTH_DATA/train_val_split")
    YoloPosePackager(jpg_dir, yolo_root).build()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-b", "--blender", action="store_true", help="Run in Blender")
    p.add_argument("-d", "--debug", action="store_true", help="Enable debugging")
    p.add_argument("-n", "--num_images", type=int, default=Cfg.NUM_IMAGES, help="Number of images to generate")
    try:
        return p.parse_args()
    except SystemExit:
        return argparse.Namespace(blender=True, debug=False, num_images=Cfg.NUM_IMAGES)


if __name__ == "__main__":
    args = parse_args()
    run_variant(args)
