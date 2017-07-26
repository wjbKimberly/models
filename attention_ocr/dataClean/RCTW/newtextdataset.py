import fsns

DEFAULT_DATASET_DIR = 'path/to/the/dataset'

DEFAULT_CONFIG = {
    'name':
        'MYDATASET',
    'splits': {
        'train': {
            'size': 123,
            'pattern': 'tfexample_train*'
        },
        'test': {
            'size': 123,
            'pattern': 'tfexample_test*'
        }
    },
    'charset_filename':
        'charset_size.txt',
    'image_shape': (150, 600, 3),
    'num_of_views':
        1,
    'max_sequence_length':
        110,
    'null_code':
        -1,
    'items_to_descriptions': {
        'label':
            'Characters codes.',
        'text':
            'A unicode string.',
        'length':
            'A length of the encoded text.',
        'num_of_views':
            'A number of different views stored within the image.'
    }
}


def get_split(split_name, dataset_dir=None, config=None):
  if not dataset_dir:
    dataset_dir = DEFAULT_DATASET_DIR
  if not config:
    config = DEFAULT_CONFIG

  return fsns.get_split(split_name, dataset_dir, config)