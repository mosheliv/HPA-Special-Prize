from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import cv2

"""Template for special prize model prediction evaluation."""
DEFAULT_MODEL_PATH = 'default_model_path'


class Model:
    """Represent a model."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        """Set up model.

        Args:
            model_path (str): Path to model.

        """
        self.path = model_path
        plugin = IEPlugin("CPU")
        net = IENetwork.from_ir(model="sn.xml", weights="sn.bin")
        assert len(net.inputs.keys()) == 1
        assert len(net.outputs) == 1 
        self.input_blob = next(iter(net.inputs))
#	out_blob = next(iter(net.outputs))
        self.exec_net = plugin.load(network=net)
        del net

    def predict(self, image_data):
        """Predict labels of image_data using model.

        Args:
            image_data (numpy array): Image data array.

        Returns:
            The predicted set of labels.

        """
        image = image_data[:,:,0, :3] #drop yellow
#        print(image.shape)
        image = cv2.resize(image, (512,512))
        image = np.rollaxis(image, 2, 0)
        image = np.expand_dims(image, axis=0)
#        print(image.shape)
        res = self.exec_net.infer(inputs={self.input_blob: image})
        res = res['dense_1/Sigmoid']
        print(res[0])
        res_ = np.arange(28)[res[0]>=0.23]
        labels = set(res_)
        return labels

