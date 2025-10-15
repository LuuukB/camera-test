# Copyright (c) farm-ng, inc.
#
# Licensed under the Amiga Development Kit License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/farm-ng/amiga-dev-kit/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


logger = logging.getLogger("amiga.apps.camera")


class CameraApp(App):

    STREAM_NAMES = ["rgb", "disparity", "left", "right"]

    def __init__(self, service_config: EventServiceConfig) -> None:
        super().__init__()

        self.service_config = service_config

        self.image_decoder = TurboJPEG()

        self.view_name = "rgb"

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        self.model.conf = 0.25
        self.labels = self.model.names

        self.webcam: cv2.VideoCapture | None = None
        self.use_webcam_fallback = False

        self.async_tasks: list[asyncio.Task] = []

    def build(self):
        return Builder.load_file("res/main.kv")

    def on_exit_btn(self) -> None:
        """Kills the running kivy application."""
        for task in self.tasks:
            task.cancel()
        App.get_running_app().stop()

    def update_view(self, view_name: str):
        self.view_name = view_name

    async def app_func(self):
        async def run_wrapper():
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.async_tasks:
                task.cancel()

        config_list = proto_from_json_file(
            self.service_config, EventServiceConfigList()
        )

        #oak0_client: EventClient | None = None

        for config in config_list.configs:
           if config.name == "oak0":
               oak0_client = EventClient(config)

        #if oak0_client is None:
        logger.warning("No OAK camera config found. Falling back to laptop webcam.")
        self.use_webcam_fallback = True
        self.webcam = cv2.VideoCapture("/mnt/c/Users/<luukb>/Videos/test_video.mp4")

        # stream camera frames
        if self.use_webcam_fallback:
            # Alleen RGB stream bij webcam fallback
            self.tasks: list[asyncio.Task] = [
                asyncio.create_task(self.stream_camera(None, "rgb"))
            ]
        else:
            # Normaal gedrag: alle streams bij OAK
            self.tasks: list[asyncio.Task] = [
                asyncio.create_task(self.stream_camera(oak0_client, view_name))
                for view_name in self.STREAM_NAMES
            ]

        return await asyncio.gather(run_wrapper(), *self.tasks)

    async def stream_camera(
        self,
        oak_client: EventClient,
        view_name: Literal["rgb", "disparity", "left", "right"] = "rgb",
    ) -> None:
        """Subscribes to the camera service and populates the tabbed panel with all 4 image streams."""
        while self.root is None:
            await asyncio.sleep(0.01)

        rate = 1  # default 1 if using webcam fallback
        #if oak_client is not None:
         #   rate = oak_client.config.subscriptions[0].every_n

        if self.use_webcam_fallback and view_name == "rgb":
            while True:
                ret, img = self.webcam.read()
                if not ret:
                    logger.warning("Failed to read from webcam")
                    await asyncio.sleep(0.01)
                    continue

                # YOLO detectie op RGB
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.model(frame_rgb)

                # Draw boxes
                for *box, conf, cls in results.xyxy[0]:
                    x1, y1, x2, y2 = map(int, box)
                    label = self.labels.get(int(cls), "onbekend")
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Create OpenGL texture
                texture = Texture.create(size=(img.shape[1], img.shape[0]), icolorfmt="bgr")
                texture.flip_vertical()
                texture.blit_buffer(bytes(img.data), colorfmt="bgr", bufferfmt="ubyte", mipmap_generation=False)
                self.root.ids[view_name].texture = texture
                await asyncio.sleep(0.01)
            return

        async for event, payload in oak_client.subscribe(
            SubscribeRequest(uri=Uri(path=f"/{view_name}"), every_n=rate),
            decode=False,
        ):
            if view_name == self.view_name:
                message = payload_to_protobuf(event, payload)
                try:
                    img = self.image_decoder.decode(message.image_data)
                except Exception as e:
                    logger.exception(f"Error decoding image: {e}")
                    continue

                #hier test opencv
                #cv2.circle(img, (img.shape[1] // 2, img.shape[0] // 2), 100, (0, 0, 255), -1)

                #zelfde code als hierboven moet ik nog methode voor aanmaken
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

                # create the opengl texture and set it to the image
                texture = Texture.create(
                    size=(img.shape[1], img.shape[0]), icolorfmt="bgr"
                )
                texture.flip_vertical()
                texture.blit_buffer(
                    bytes(img.data),
                    colorfmt="bgr",
                    bufferfmt="ubyte",
                    mipmap_generation=False,
                )
                self.root.ids[view_name].texture = texture


def find_config_by_name(
    service_configs: EventServiceConfigList, name: str
) -> EventServiceConfig | None:
    """Utility function to find a service config by name.

    Args:
        service_configs: List of service configs
        name: Name of the service to find
    """
    for config in service_configs.configs:
        if config.name == name:
            return config
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="template-app")

    # Add additional command line arguments here
    parser.add_argument("--service-config", type=Path, default="service_config.json")

    args = parser.parse_args()

    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(CameraApp(args.service_config).app_func())
    except asyncio.CancelledError:
        pass
    loop.close()
