import os
from pycocotools.coco import COCO
import numpy as np

def download(url, dest):
    import urllib
    print(f'Downloading {url.split("/")[-1]} to {dest}')
    urllib.request.urlretrieve(url , filename = dest)

def extract_zip(file, extract_path):
    import zipfile
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    try:
        os.remove(file)
        print('Zip file extracted and removed')
    except:
        None

def download_coco(coco_api_dir, year):
    annotations_trainval_url = f'http://images.cocodataset.org/annotations/annotations_trainval{year}.zip'
    image_info_test_url = f'http://images.cocodataset.org/annotations/image_info_test{year}.zip'
    train_url = f'http://images.cocodataset.org/zips/train{year}.zip'
    test_url = f'http://images.cocodataset.org/zips/test{year}.zip'
    # val_url = f'http://images.cocodataset.org/zips/val{year}.zip'
    annotations_trainval_dest = os.path.join(coco_api_dir, annotations_trainval_url.split('/')[-1])
    image_info_test_dest = os.path.join(coco_api_dir, image_info_test_url.split('/')[-1])
    train_imgs_dest = os.path.join(coco_api_dir, train_url.split('/')[-1])
    test_imgs_dest = os.path.join(coco_api_dir, test_url.split('/')[-1])
    # val_imgs_dest = os.path.join(coco_api_dir, f'val{year}')
    annotations_dir = os.path.join(coco_api_dir, 'annotations')
    images_dir = os.path.join(coco_api_dir, 'images')

    # Download annotations for indexing
    download(annotations_trainval_url, annotations_trainval_dest)
    extract_zip(annotations_trainval_dest, coco_api_dir)
    download(image_info_test_url, image_info_test_dest)
    extract_zip(image_info_test_dest, coco_api_dir)
    download(train_url, train_imgs_dest)
    extract_zip(train_imgs_dest, images_dir)
    download(test_url, test_imgs_dest)
    extract_zip(test_imgs_dest, images_dir)

    # Verify the datasets have been downloaded and extracted correctly
    # Initialize COCO API for instance annotations
    dataType = f'val{year}'
    instances_annFile = os.path.join(annotations_dir, f'instances_{dataType}.json')
    coco = COCO(instances_annFile)

    # Initialize COCO API for caption annotations
    captions_annFile = os.path.join(annotations_dir, f'captions_{dataType}.json')
    coco_caps = COCO(captions_annFile)

    print('Verifying annotation downloads')
    # Get image ids 
    ids = list(coco.anns.keys())
    ann_id = np.random.choice(ids)
    img_id = coco.anns[ann_id]['image_id']
    img = coco.loadImgs( img_id )[0]
    url = img['coco_url']
    print(f'Image url {url}')
    ann_ids = coco_caps.getAnnIds(img_id)
    print(f'Indices for annotations/captions for the image: {ann_ids}')

    # print('Verifying image downloads')

def main():
    coco_api_dir = os.path.join('D:', 'dev', 'data', 'cocoapi')
    year = '2014'

    download_coco(coco_api_dir, year)

if __name__ == '__main__':
    main()