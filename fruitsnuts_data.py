from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
register_coco_instances("fruits_nuts", {'thing_classes':['date', 'fig', 'hazelnut'],
                                        'thing_dataset_id_to_contiguous_id':{1: 0, 2: 1, 3: 2}}, "./data/trainval.json", "./data/images")
fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
# Metadata(evaluator_type='coco', image_root='./data/images',
# json_file='./data/trainval.json',
# name='fruits_nuts', thing_classes=['date', 'fig', 'hazelnut'],
# thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2})
