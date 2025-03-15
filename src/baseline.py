from data import create_data
import fiftyone as fo
import fiftyone.zoo as foz

def load_data(name):
    dataset = fo.load_dataset(name)
    return dataset

dataset = load_data("my-dataset")

session = fo.launch_app(dataset)
model = foz.load_zoo_model('faster-rcnn-resnet50-fpn-coco-torch')
predictions_view = dataset.take(100, seed=51)
predictions_view.apply_model(model, label_field="faster_rcnn")
session.view = predictions_view