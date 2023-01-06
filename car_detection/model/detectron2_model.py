"""Detectron2 model - functions to load and predict on images."""

from typing import Tuple

import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Instances


def create_model(
    model_config_file: str, threshold: float = 0.5
) -> Tuple[DefaultPredictor, CfgNode]:
    """Create Detectron2 model based on given config file string.

    :param model_config_file: Detectron2 model config, e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    :param threshold: Threshold to use when predicting.
    :return: predictor, config object
    """
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file(model_config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config_file)
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def detect_objects_of_one_class(
    predictor: DefaultPredictor,
    cfg: CfgNode,
    image: np.ndarray,
    object_name: str = "car",
) -> Instances:
    """Detect objects using given predictor&config on given image. Only objects of one given class are returned.

    :param predictor: DefaultPredictor object.
    :param cfg: CfgNode object.
    :param image: Loaded image.
    :param object_name: String representing object name.
    :return: Instances with found objects of given class.
    """
    outputs = predictor(image)
    instances = outputs["instances"]
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    car_id = class_names.index(object_name)
    instances = instances[instances.pred_classes == car_id]
    return instances


def draw_predictions_on_image(
    image: np.ndarray, instances: Instances, cfg: CfgNode
) -> np.ndarray:
    """Draw predictions on image.

    :param image: Image.
    :param instances: Instances with found objects.
    :param cfg: CfgNode object.
    :return: Image with obejcts drawn on top of it.
    """
    v = Visualizer(
        image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    )
    out = v.draw_instance_predictions(instances.to("cpu"))
    return out.get_image()[:, :, ::-1]


def detect(
    image: np.ndarray,
    model_config_file: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
) -> Tuple[np.ndarray, str]:
    """Detect objects on given image using given model.

    :param image: Image to process.
    :param model_config_file: Model config file string from Detectron Model ZOO.
    :return: Image with detections, string with information about number of cars.
    """
    predictor, cfg = create_model(model_config_file)
    instances = detect_objects_of_one_class(predictor, cfg, image)
    image_with_detections = draw_predictions_on_image(image, instances, cfg)
    return image_with_detections, f"Found {len(instances)} car(s)."
