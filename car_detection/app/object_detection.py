"""Gradio app demo."""
from typing import Tuple

import numpy as np
import gradio as gr
from model.detectron2_model import Detectron2Model


def detect(image: np.ndarray, config_name: str) -> Tuple[np.ndarray, str]:
    """Detect objects on image.

    :param image: Image.
    :param config_name: Detectron2 model config from Model ZOO, e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    :return: Image with detections, string with information about number of found objects.
    """
    model = Detectron2Model(config_name)
    return model.detect(image, ["car"])


if __name__ == "__main__":
    demo = gr.Interface(
        fn=detect,
        inputs=[
            "image",
            gr.Dropdown(
                [
                    "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml",
                    "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
                    "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
                    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                    "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
                    "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
                    "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
                    "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml",
                    "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
                    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
                    "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
                    "COCO-Detection/retinanet_R_50_FPN_1x.yaml",
                    "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
                    "COCO-Detection/rpn_R_50_C4_1x.yaml",
                    "COCO-Detection/rpn_R_50_FPN_1x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_giou.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
                ]
            ),
        ],
        outputs=["image", "text"],
    )

    demo.launch(server_name="0.0.0.0", server_port=7860)
