"""Detectron2 model - functions to load and predict on images."""

from typing import Tuple

import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Instances


class Detectron2Model:
    def __init__(self, model_config_file, threshold=0.5):
        """Init class.

        :param model_config_file: Detectron2 model config from Model ZOO, e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        :param threshold: Threshold to use when predicting.
        """
        self.predictor = None
        self.cfg = None
        self.model_config_file = model_config_file
        self.threshold = threshold

    def detect(self, image: np.ndarray, object_names: list) -> Tuple[np.ndarray, str]:
        """Detect objects on given image using given model.

        :param image: Image to process.
        :param object_names: Names of objects to detect.
        :return: Image with detections, string with information about number of found objects.
        """
        self.create_model()
        instances = self.detect_objects_of_some_classes(image, object_names)
        image_with_detections = self.draw_predictions_on_image(image, instances)
        return image_with_detections, f"Found {len(instances)} object(s)."

    def create_model(self) -> Tuple[DefaultPredictor, CfgNode]:
        """Create Detectron2 model based on given config file string.

        :return: predictor, config object
        """
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.merge_from_file(model_zoo.get_config_file(self.model_config_file))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_config_file)
        self.predictor = DefaultPredictor(self.cfg)
        return self.predictor, self.cfg

    def detect_all_objects(self, image: np.ndarray) -> Instances:
        """Detect all objects.

        :param image: image
        :return: Instances with all found objects.
        """
        outputs = self.predictor(image)
        instances = outputs["instances"]
        return instances

    def detect_objects_of_some_classes(
        self,
        image: np.ndarray,
        object_names: list = None,
    ) -> Instances:
        """Detect objects using given predictor&config on given image. Only objects of one given class are returned.

        :param image: Loaded image.
        :param object_name: String representing object name.
        :return: Instances with found objects of given class.
        """
        if object_names is None:
            object_names = ["car"]
        instances = self.detect_all_objects(image)
        class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes
        object_ids = [class_names.index(object_name) for object_name in object_names]
        instances = Instances.cat(
            [instances[instances.pred_classes == object_id] for object_id in object_ids]
        )
        return instances

    def draw_predictions_on_image(
        self, image: np.ndarray, instances: Instances
    ) -> np.ndarray:
        """Draw predictions on image.

        :param image: Image.
        :param instances: Instances with found objects.
        :param cfg: CfgNode object.
        :return: Image with objects drawn on top of it.
        """
        v = Visualizer(
            image[:, :, ::-1],
            MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            scale=1.2,
        )
        out = v.draw_instance_predictions(instances.to("cpu"))
        return out.get_image()[:, :, ::-1]
