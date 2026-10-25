import carla, random, os, queue, math, sys
import numpy as np

W, H, FOV = 1280, 720, 90
N_SCENES = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NEG_RATIO = 0.06
OUT = "hazard_dataset"
CLASSES = ["blocked_road", "debris", "damaged_building"]

WEATHERS = {
    "clear": carla.WeatherParameters.ClearNoon,
    "rain": carla.WeatherParameters.HardRainNoon,
    "fog": carla.WeatherParameters(cloudiness=80, fog_density=75, fog_distance=8,
                                   sun_altitude_angle=40),
    "night": carla.WeatherParameters.ClearNight,
    "rainnight": carla.WeatherParameters.HardRainNight,
}
MODES = {
    "ground":   {"h": (1.2, 2.2),  "d": (12, 28)},
    "elevated": {"h": (4, 10),     "d": (15, 30)},
    "uav":      {"h": (15, 40),    "d": (10, 30)},
}
wlist, mlist = list(WEATHERS), list(MODES)

for s in ("train", "val", "test"):
    for w in wlist:
        os.makedirs(f"{OUT}/images/{s}/{w}", exist_ok=True)
        os.makedirs(f"{OUT}/labels/{s}/{w}", exist_ok=True)
names_yaml = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
root = os.path.abspath(OUT)
with open(f"{OUT}/data.yaml", "w") as f:
    f.write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n" + names_yaml)
for w in wlist:
    with open(f"{OUT}/data_test_{w}.yaml", "w") as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test/{w}\n" + names_yaml)

client = carla.Client("localhost", 2000)
client.set_timeout(30)
world = client.get_world()
amap = world.get_map()
bp = world.get_blueprint_library()
old_settings = world.get_settings()
world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=0.05))

car_bps = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]


def props(names):
    out = []
    for n in names:
        found = bp.filter(n)
        if len(found) > 0:
            out.append(found[0])
    return out


debris_bps = props(["static.prop.box01", "static.prop.box02", "static.prop.box03",
                    "static.prop.trashcan01", "static.prop.trashcan02", "static.prop.trashcan03",
                    "static.prop.ironplank", "static.prop.streetbarrier"])
cone_bps = props(["static.prop.constructioncone", "static.prop.trafficwarning",
                  "static.prop.warningconstruction"])
print("debris types:", len(debris_bps), "cone types:", len(cone_bps))

cam_bp = bp.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", str(W))
cam_bp.set_attribute("image_size_y", str(H))
cam_bp.set_attribute("fov", str(FOV))
focal = W / (2 * math.tan(math.radians(FOV) / 2))
K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])


def box2d(actor, cam):
    w2c = np.array(cam.get_transform().get_inverse_matrix())
    pts = []
    for v in actor.bounding_box.get_world_vertices(actor.get_transform()):
        p = w2c @ np.array([v.x, v.y, v.z, 1])
        p = np.array([p[1], -p[2], p[0]])
        if p[2] <= 0.5:
            return None
        uv = K @ p
        pts.append((uv[0] / uv[2], uv[1] / uv[2]))
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def yolo_line(cls, b):
    x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"


def place(wp, lateral, yaw_off, z=0.3):
    t = wp.transform
    loc = t.location + t.get_right_vector() * lateral + carla.Location(z=z)
    return carla.Transform(loc, carla.Rotation(yaw=t.rotation.yaw + yaw_off))


def try_spawn(blueprint, tr, actors):
    a = world.try_spawn_actor(blueprint, tr)
    if a:
        actors.append(a)
    return a


spawn_points = amap.get_spawn_points()
count = {"train": 0, "val": 0, "test": 0}
actors = []
try:
    for i in range(START, N_SCENES):
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
        actors.clear()

        wname = wlist[i % len(wlist)]
        mname = mlist[(i // len(wlist)) % len(mlist)]
        world.set_weather(WEATHERS[wname])

        hz = amap.get_waypoint(random.choice(spawn_points).location)
        lw = hz.lane_width
        t = hz.transform
        hgt = random.uniform(*MODES[mname]["h"])
        dist = random.uniform(*MODES[mname]["d"])
        cam_loc = (t.location - t.get_forward_vector() * dist
                   + t.get_right_vector() * random.uniform(-2.5, 2.5)
                   + carla.Location(z=hgt))
        pitch = -math.degrees(math.atan2(max(hgt - 0.5, 0.1), dist)) + random.uniform(-4, 4)
        cam_tf = carla.Transform(cam_loc, carla.Rotation(pitch=pitch,
                                 yaw=t.rotation.yaw + random.uniform(-12, 12)))
        cam = world.spawn_actor(cam_bp, cam_tf)
        actors.append(cam)
        world.get_spectator().set_transform(cam_tf)

        negative = random.random() < NEG_RATIO
        blockers, debris = [], []
        if not negative:
            for k in range(random.randint(1, 3)):
                a = try_spawn(random.choice(car_bps),
                              place(hz, random.uniform(-lw / 3, lw / 3) + k * 0.3,
                                    random.choice([70, 90, 110])), actors)
                if a:
                    blockers.append(a)
            if cone_bps:
                for _ in range(random.randint(0, 3)):
                    a = try_spawn(random.choice(cone_bps),
                                  place(hz, random.uniform(-lw / 2, lw / 2),
                                        random.uniform(0, 360)), actors)
                    if a:
                        blockers.append(a)
            for _ in range(random.randint(4, 10)):
                if random.random() < 0.5:
                    nb = hz.next(random.uniform(0.5, 6))
                else:
                    nb = hz.previous(random.uniform(0.5, 6))
                base = nb[0] if nb else hz
                a = try_spawn(random.choice(debris_bps),
                              place(base, random.uniform(-lw / 2, lw / 2),
                                    random.uniform(0, 360), z=0.15), actors)
                if a:
                    debris.append(a)

        for _ in range(6):
            world.tick()
        q = queue.Queue()
        cam.listen(q.put)
        world.tick()
        img = q.get(timeout=10)
        cam.stop()

        lines = []
        bb = [b for b in (box2d(a, cam) for a in blockers) if b]
        if bb:
            ln = yolo_line(0, union(bb))
            if ln:
                lines.append(ln)
        for a in debris:
            b = box2d(a, cam)
            if b:
                ln = yolo_line(1, b)
                if ln:
                    lines.append(ln)
        if not negative and not lines:
            continue

        r = random.random()
        split = "train" if r < 0.8 else ("val" if r < 0.9 else "test")
        name = f"{wname}_{mname}_{i:04d}"
        img.save_to_disk(f"{OUT}/images/{split}/{wname}/{name}.png")
        with open(f"{OUT}/labels/{split}/{wname}/{name}.txt", "w") as f:
            f.write("\n".join(lines))
        count[split] += 1
        if i % 25 == 0:
            print(i, wname, mname, count)
finally:
    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass
    world.apply_settings(old_settings)
    print("Done", count)
