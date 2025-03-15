import fiftyone as fo

def create_data(name):
    data_path = "../data/JPEGImages"
    labels_path = "../data/Annotations/"

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.VOCDetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
        name=name,
    )
    return dataset

