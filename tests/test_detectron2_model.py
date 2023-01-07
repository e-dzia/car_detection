import unittest
from parameterized import parameterized
from model.detectron2_model import Detectron2Model
from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode


class TestDetectron2Model(unittest.TestCase):
    @parameterized.expand(
        [
            ["COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"],
            ["COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"],
        ]
    )
    def test_create_model(self, model_config_file):
        mdl = Detectron2Model(model_config_file)
        model, cfg = mdl.create_model(model_config_file)
        self.assertIsInstance(model, DefaultPredictor)
        self.assertIsInstance(cfg, CfgNode)

    @parameterized.expand(
        [
            ["not-COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"],
            ["something_different"],
        ]
    )
    def test_create_model_error(self, model_config_file):
        mdl = Detectron2Model(model_config_file)
        with self.assertRaises(RuntimeError) as context:
            mdl.create_model(model_config_file)
        self.assertTrue("not available in Model Zoo!" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
