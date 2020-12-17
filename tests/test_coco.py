from PIL import Image
import os
import os.path
from pycocotools.coco import COCO
import unittest


annFile = "C:\\zhulei\\data\\instance_segmentation\\rock\\shapes\\train\\instances_leaf_train2017.json"
root = "C:\\zhulei\\data\\instance_segmentation\\rock\\shapes\\train\\shapes_train2017"


class TestCocoApi(unittest.TestCase):
    def test_coco_api(self):
        coco = COCO(annFile)
        ids = list(sorted(coco.imgs.keys()))
        print(ids)
        img_id = ids[1]

        print(img_id)

        ann_ids = coco.getAnnIds(imgIds=img_id)

        print(ann_ids)

        target = coco.loadAnns(ann_ids)

        # print(target)

        path = coco.loadImgs(img_id)[0]['file_name']

        print(path)

        imgs_path = os.path.join(root, path)
        for img_path in os.listdir(imgs_path):
            img = Image.open(os.path.join(imgs_path, img_path)).convert('RGB')
        # img = Image.open(os.path.join(root, path)).convert('RGB')


if __name__ == "__main__":
    unittest.main()
