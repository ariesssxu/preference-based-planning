import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils import convert
import os

class Video_Recoder():
    def __init__(self, robot=None, camera=None, map=None, save_path=".", name=None, fps=30):
        self.robot = robot
        self.camera= camera
        self.map = map
        self.video_path = save_path
        self.fps = fps
        self.his_pos = None

    def set(self, camera=None, robot=None, save_path=None, name=None, trav_map_img=None, trav_map_size=222):
        self.robot = robot
        self.camera = camera
        self.video_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.robot_video_writer = imageio.get_writer(f"{save_path}/{name}_robot.mp4", fps=30)
        self.camera_video_writer = imageio.get_writer(f"{save_path}/{name}_camera.mp4", fps=30)
        self.map_video_writer = imageio.get_writer(f"{save_path}/{name}_map.mp4", fps=30)
        self.text_writer = f"{save_path}/{name}_text.log"
        with open(self.text_writer, "w+") as f:
            f.write("<s>")
        self.trav_map_img = Image.fromarray(trav_map_img)
        self.trav_map_size = trav_map_size
        self.draw = ImageDraw.Draw(self.trav_map_img)

    def get_video(self, start=False, end=False, text=" ", movement=True, path=None):
        # if movement, call get_map_video, else call get_map_video_without_movement and pass path
        self.get_robot_video()
        self.get_camera_video()
        if movement:
            self.get_map_video(start, end)
        else:
            self.get_map_video_without_movement(path, start, end)
        self.get_text(text)

    def get_text(self, text):
        if len(text) < 3:
            assert "Text is too short"
            assert 0
        with open(self.text_writer, "a") as f:
            f.write(text)
            f.write("\n")

    def get_robot_video(self, robot=None):
        robot_view = self.robot.get_obs() if robot is None else robot.get_obs()
        # robot_obs = robot_view['robot:eyes_Camera_sensor_rgb']
        # camera name changed in 2023.1
        rgb_idx = [eye for eye in list(robot_view[0]) if "eyes" in eye][0]
        robot_obs = robot_view[0][rgb_idx]["rgb"]
        self.robot_video_writer.append_data(robot_obs[:, :, :-1])

    def get_camera_video(self, camera=None):
        camera = self.camera if camera is None else camera
        camera_view = camera.get_obs()
        camera_obs = camera_view[0][list(camera_view[0].keys())[0]]
        self.camera_video_writer.append_data(camera_obs[:, :, :-1])
    
    def get_map_video(self, start=False, end=False):
        pos = convert(self.robot.get_position(), self.trav_map_size)[:2]
        if self.his_pos is None:
            self.his_pos = pos
        points = [(pos[0], pos[1]), (self.his_pos[0], self.his_pos[1])]
        self.draw.line(points, fill="green", width=5)
        # plt.plot([pos[0], self.his_pos[0]], [
        #          pos[1], self.his_pos[1]], c="g", linewidth=5)
        if start:
            self.draw.ellipse([pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5], fill="red")
            # plt.scatter(pos[0], pos[1], c="r", s=100)
        if end:
            self.draw.ellipse([pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5], fill="blue")
            # plt.scatter(pos[0], pos[1], c="b", s=100)
        self.his_pos = pos
        # get plt image and write to video
        self.map_video_writer.append_data(np.asarray(self.trav_map_img))

    def get_map_video_without_movement(self, path, start=False, end=False):
        for pos in path:
            pos = convert(pos, self.trav_map_size)[:2]
            if self.his_pos is None:
                self.his_pos = pos
            points = [(pos[0], pos[1]), (self.his_pos[0], self.his_pos[1])]
            self.draw.line(points, fill="green", width=5)
            if start:
                self.draw.ellipse([pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5], fill="red")
            if end:
                self.draw.ellipse([pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5], fill="blue")
            self.his_pos = pos
        self.map_video_writer.append_data(np.asarray(self.trav_map_img))

    def release(self):
        self.robot_video_writer.close()
        self.camera_video_writer.close()
        self.map_video_writer.close()

video_recorder = Video_Recoder(camera=None, save_path=None, name="Default")
