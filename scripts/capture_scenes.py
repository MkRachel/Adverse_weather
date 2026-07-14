import carla
import time
import os

# --- CONNECT TO CARLA ---
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# --- WEATHER PRESETS TO CAPTURE (rain, fog, night) ---
weather_presets = {
    "clear_day": carla.WeatherParameters.ClearNoon,
    "rain": carla.WeatherParameters.HardRainNoon,
    "fog": carla.WeatherParameters(cloudiness=80.0, fog_density=80.0,
                                    fog_distance=10.0, sun_altitude_angle=45.0),
    "night": carla.WeatherParameters.ClearNight,
    "rain_night": carla.WeatherParameters.HardRainNight,
}

# --- SPAWN VEHICLE ---
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_points = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
vehicle.set_autopilot(True)   # car drives itself so scenes vary naturally
print("Vehicle spawned and autopilot enabled.")

# --- ATTACH RGB CAMERA ---
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '720')
camera_bp.set_attribute('fov', '100')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# --- OUTPUT FOLDER SETUP ---
base_dir = "synthetic_dataset"
os.makedirs(base_dir, exist_ok=True)

image_counter = {"count": 0}
current_weather_name = {"name": "clear_day"}

def save_image(image):
    folder = os.path.join(base_dir, current_weather_name["name"])
    os.makedirs(folder, exist_ok=True)
    image.save_to_disk(f'{folder}/{image.frame}.png')
    image_counter["count"] += 1

camera.listen(save_image)

# --- CYCLE THROUGH WEATHER CONDITIONS ---
try:
    for name, weather in weather_presets.items():
        current_weather_name["name"] = name
        world.set_weather(weather)
        print(f"Capturing under: {name}")
        time.sleep(10)   # let car drive and camera capture for 10s per condition
finally:
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    print(f"Done. Total images captured: {image_counter['count']}")
