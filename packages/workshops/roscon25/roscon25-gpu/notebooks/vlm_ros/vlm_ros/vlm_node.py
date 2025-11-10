#!/usr/bin/env python3
"""ROS2 service node that proxies image+prompt queries to a llama.cpp server."""

import rclpy
from rclpy.node import Node
from vlm_ros.srv import VLMQuery
import base64
import cv2
import requests
from cv_bridge import CvBridge
from requests import RequestException


class VLMNode(Node):
    def __init__(self):
        super().__init__('vlm_node')

        self.declare_parameter('llama_server_url', 'http://localhost:8080')
        self._llama_url = self.get_parameter('llama_server_url').value.rstrip('/')

        self._bridge = CvBridge()
        self._session = requests.Session()

        self.create_service(VLMQuery, 'vlm_query', self._handle_query)
        self.get_logger().info(
            f'vlm_ros service ready on /vlm_query (llama.cpp at {self._llama_url})'
        )

    def _handle_query(self, request, response):
        try:
            image_b64 = self._image_to_base64(request.image)
            response.response = self._query_llama(request.prompt, image_b64)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f'Failed to handle request: {exc}')
            response.response = f'Error: {exc}'
        return response

    def _image_to_base64(self, image_msg):
        cv_image = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        success, encoded = cv2.imencode('.jpg', cv_image)
        if not success:
            raise RuntimeError('Failed to encode image to JPEG')
        return base64.b64encode(encoded.tobytes()).decode('ascii')

    def _query_llama(self, prompt, image_base64):
        payload = {
            "model": "vlm",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 256
        }

        try:
            resp = self._session.post(
                f'{self._llama_url}/v1/chat/completions',
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except RequestException as exc:
            raise RuntimeError(f'HTTP request failed: {exc}') from exc

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f'unexpected response format: {data}') from exc


def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
