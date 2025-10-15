from __future__ import annotations
import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Literal

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.event_service_pb2 import EventServiceConfigList
from farm_ng.core.event_service_pb2 import SubscribeRequest
from farm_ng.core.events_file_reader import payload_to_protobuf
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.uri_pb2 import Uri
from turbojpeg import TurboJPEG

import cv2
import torch

os.environ["KIVY_NO_ARGS"] = "1"

from kivy.config import Config  # noreorder # noqa: E402
Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

from kivy.app import App  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.graphics.texture import Texture  # noqa: E402
from kivy.clock import Clock  # voor webcam fallback

logger = logging.getLogger("amiga.apps.camera")


class CameraApp(App):

    STREAM_NAMES = ["rgb", "disparity", "left", "right"]

    def __init__(self, service_config: EventServiceConfig) -> None:
        super().__init__()
        self.service_config = service_config
        self.image_decoder = TurboJPEG()
        self.view_name = "rgb"
        self.async_tasks: list[asyncio.Task] = []

        # YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        self.model.conf = 0.25
        self.labels = self.model.names

        # Webcam fallback
        self.webcam: cv2.VideoCapture | None = None
        self.use_webcam_fallback = False

    def build(self):
        return Builder.load_file("res/main.kv")

    def on_start(self):
        # Start webcam loop als fallback
        if self.use_webcam_fallback:
            Clock.schedule_interval(self.update_webcam_frame, 1/30)  # 30 FPS

    def on_exit_btn(self) -> None:
        """Kills the running kivy application."""
        for task in self.async_tasks:
            task.cancel()
        if self.webcam is not None:
            self.webcam.release()
        App.get_running_app().stop()

    def update_view(self, view_name: str):
        self.view_name = view_name

    async def app_func(self):
        async def run_wrapper():
            await self.async_run(async_lib="asyncio")
            for task in self.async_tasks:
                task.cancel()

        # Load service configs
        config_list = proto_from_json_file(self.service_config, EventServiceConfigList())
        oak0_client: EventClient | None = None

        for config in config_list.configs:
            if config.name == "oak0":
                oak0_client = EventClient(config)

        # If no OAK, fallback to webcam
        if oak0_client is None:
            logger.warning("No OAK camera config found. Falling back to laptop webcam.")
            self.use_webcam_fallback = True
            self.webcam = cv2.VideoCapture(0)
            # Set resolution voor laptop webcam (kan je aanpassen)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Start streaming tasks voor OAK-camera streams
        if oak0_client is not None:
            self.async_tasks = [
                asyncio.create_task(self.stream_camera(oak0_client, view_name))
                for view_name in self.STREAM_NAMES
            ]
            return await asyncio.gather(run_wrapper(), *self.async_tasks)
        else:
            # Alleen Kivy + webcam fallback loop
            return await run_wrapper()

    def update_webcam_frame(self, dt):
        if self.webcam is None or not self.webcam.isOpened():
            return
        ret, img = self.webcam.read()
        if not ret:
            return

        # YOLO detectie alleen op RGB
        if self.view_name == "rgb":
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.model(frame_rgb)
            for *box, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, box)
                label = self.labels.get(int(cls), "onbekend")
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update Kivy texture
        texture = Texture.create(size=(img.shape[1], img.shape[0]), icolorfmt="bgr")
        texture.flip_vertical()
        texture.blit_buffer(bytes(img.data), colorfmt="bgr", bufferfmt="ubyte", mipmap_generation=False)
        self.root.ids["rgb"].texture = texture

    async def stream_camera(
        self,
        oak_client: EventClient,
        view_name: Literal["rgb", "disparity", "left", "right"] = "rgb",
    ) -> None:
        """OAK-camera streaming."""
        while self.root is None:
            await asyncio.sleep(0.01)

        rate = oak_client.config.subscriptions[0].every_n

        async for event, payload in oak_client.subscribe(
            SubscribeRequest(uri=Uri(path=f"/{view_name}"), every_n=rate), decode=False
        ):
            if view_name != self.view_name:
                continue

            message = payload_to_protobuf(event, payload)
            try:
                img = self.image_decoder.decode(message.image_data)
            except Exception as e:
                logger.exception(f"Error decoding image: {e}")
                continue

            # YOLO detectie alleen op RGB
            if view_name == "rgb":
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.model(frame_rgb)
                for *box, conf, cls in results.xyxy[0]:
                    x1, y1, x2, y2 = map(int, box)
                    label = self.labels.get(int(cls), "onbekend")
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update Kivy texture
            texture = Texture.create(size=(img.shape[1], img.shape[0]), icolorfmt="bgr")
            texture.flip_vertical()
            texture.blit_buffer(bytes(img.data), colorfmt="bgr", bufferfmt="ubyte", mipmap_generation=False)
            self.root.ids[view_name].texture = texture


def find_config_by_name(service_configs: EventServiceConfigList, name: str) -> EventServiceConfig | None:
    for config in service_configs.configs:
        if config.name == name:
            return config
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="camera-app")
    parser.add_argument("--service-config", type=Path, default="service_config.json")
    args = parser.parse_args()

    # Compatibel met Python 3.8 t/m 3.12
    import sys
    if sys.version_info >= (3, 11):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(CameraApp(args.service_config).app_func())
    except asyncio.CancelledError:
        pass
    loop.close()
