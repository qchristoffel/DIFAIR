# CODE TAKEN AND ADAPTED FROM : https://github.com/ksachdeva/tiny-imagenet-tfds
# using the code from the github didn't work

"""tiny_imagenet dataset."""

import os
import tensorflow as tf
import tensorflow_datasets as tfds
from pprint import pprint

# Markdown description  that will appear on the catalog page.
_DESCRIPTION = """ Tiny ImageNet Challenge is a similar challenge as ImageNet 
with a smaller dataset but less image classes. It contains 200 image classes, 
a training dataset of 100, 000 images, a validation dataset of 10, 000 images, 
and a test dataset of 10, 000 images. All images are of size 64×64.
"""

# BibTeX citation
_CITATION = """@article{tiny-imagenet,
 author = {Li,Fei-Fei}, {Karpathy,Andrej} and {Johnson,Justin}"}
"""

_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
_EXTRACTED_FOLDER_NAME = "tiny-imagenet-200"
SUPPORTED_IMAGE_FORMAT = (".jpg", ".jpeg", ".png")

def _list_folders(root_dir):
    return [
        f for f in tf.io.gfile.listdir(root_dir)
        if tf.io.gfile.isdir(os.path.join(root_dir, f))
    ]


def _list_imgs(root_dir):
    return [
        os.path.join(root_dir, f)
        for f in tf.io.gfile.listdir(root_dir)
        if any(f.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMAT)
    ]

class TinyImagenet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for tiny_imagenet dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(64, 64, 3)),
                'id' : tfds.features.Text(),
                'label': tfds.features.ClassLabel(num_classes=200),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage="https://www.image-net.org/download.php",
            citation=_CITATION,
        )
    
    def _process_train_ds(self, ds_folder, identities):
        path_to_ds = os.path.join(ds_folder, 'train')
        names = _list_folders(path_to_ds)

        label_images = {}
        for n in names:
            images_dir = os.path.join(path_to_ds, n, 'images')
            total_images = _list_imgs(images_dir)
            label_images[n] = {
                'images': total_images,
                'id': identities.index(n)
            }

        return label_images

    def _process_val_ds(self, ds_folder, identities):
        path_to_ds = os.path.join(ds_folder, 'val')

        # read the val_annotations.txt file
        with tf.io.gfile.GFile(os.path.join(path_to_ds, 'val_annotations.txt')) as f:
            data_raw = f.read()

        lines = data_raw.split("\n")

        label_images = {}
        for line in lines:
            if line == '':
                continue
            row_values = line.strip().split()
            label_name = row_values[1]
            if not label_name in label_images.keys():
                label_images[label_name] = {
                    'images': [],
                    'id': identities.index(label_name)
                }

            label_images[label_name]['images'].append(
                os.path.join(path_to_ds, 'images', row_values[0]))

        return label_images
    
    # use grep to find association between wnid and class name
    def _grep(self, ds_folder):
        id_file = os.path.join(ds_folder, 'wnids.txt')
        dictionnary_file = os.path.join(ds_folder, 'words.txt')
        lines = os.popen(f'grep -F -f {id_file} {dictionnary_file}').readlines()
        classes = { line.split('\t')[0]: line.split('\t')[1].strip() for line in lines }
        return classes
        # pprint(classes)
        # exit()
        # grep -F -f wnids.txt words.txt > wnids_names.txt

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.extract(dl_manager.download(_URL))
        ds_folder = os.path.join(extracted_path, _EXTRACTED_FOLDER_NAME)

        # Load the label names
        # inspired from : 
        # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/cifar.py
        
        classes = self._grep(ds_folder) # return dict of {wnid: class_name}
        
        # wnids, label_names = zip(*classes)
        
        # grep looses the order of classes so we need to read wnids.txt and reorder
        with tf.io.gfile.GFile(os.path.join(ds_folder, 'wnids.txt')) as f:
            data_raw = f.read()
        wnids = data_raw.split("\n")
        wnids = wnids[:-1] # remove last empty line

        label_names = [classes[wnid] for wnid in wnids]
        self.info.features["label"].names = label_names
        
        train_label_images = self._process_train_ds(ds_folder, wnids)
        validation_label_images = self._process_val_ds(ds_folder, wnids)

        return {
            tfds.Split.TRAIN: self._generate_examples(train_label_images),
            tfds.Split.TEST: self._generate_examples(validation_label_images)
        }

    def _generate_examples(self, label_images):
        for label, image_info in label_images.items():
            for image_path in image_info['images']:
                key = "%s/%s" % (label, os.path.basename(image_path))
                # print(image_info['id'])
                yield key, {
                    "image": image_path,
                    "id": label,
                    "label": image_info['id'],
                }