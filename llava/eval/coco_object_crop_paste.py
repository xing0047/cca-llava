import os
import argparse
import json
from llava.mm_utils import expand2square
from PIL import Image
from tqdm import tqdm

IMAGENET_PIXEL_MEAN = (122, 116, 104)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-folder', type=str, default=""
    )
    parser.add_argument(
        '--annotation-file', type=str, default=""
    )
    parser.add_argument(
        '--output-image-folder', type=str, default=""
    )
    parser.add_argument(
        '--output-file', type=str, default=""
    )
    parser.add_argument(
        '--max-object-num', type=int, default=3000
    )
    return parser.parse_args()
    
def main():

    args = parse_args()
    data = json.load(open(args.annotation_file))

    image_infos = {image['id']: image for image in data['images']}
    annotations = data['annotations']
    categories = {category['id']: category for category in data['categories']}

    object_num = 0
    question_id = 1
    with open(args.output_file, 'w') as f:
        for annotation_info in tqdm(annotations):
            image_id = annotation_info['image_id']
            instance_id = annotation_info['id']
            bbox = annotation_info['bbox']
            image_info = image_infos[image_id]
            # skip small / large objects.
            if bbox[2] * bbox[3] < 96 * 96 or bbox[2] * bbox[3] > 128 * 128:
                continue
            # skip abnormal aspect ratio objects.
            if bbox[2] / bbox[3] > 1.5 or bbox[3] / bbox[2] > 1.5:
                continue
            category_id = annotation_info['category_id']
            category_name = categories[category_id]['name']
            # load image.
            coco_image = Image.open(f"{args.image_folder}/{image_info['file_name']}").convert('RGB')
            # crop region of interest.
            crop_region = coco_image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            # crop region sexpand to square.
            crop_region = expand2square(crop_region, IMAGENET_PIXEL_MEAN).resize((28, 28))
            # mkdir.
            if not os.path.exists(f'{args.output_image_folder}/{str(image_id).zfill(12)}_{str(category_id).zfill(3)}_{str(instance_id).zfill(12)}'):
                os.makedirs(f'{args.output_image_folder}/{str(image_id).zfill(12)}_{str(category_id).zfill(3)}_{str(instance_id).zfill(12)}')
            for x1 in range(0, 336, 28):
                for y1 in range(0, 336, 28):
                    # new image.
                    image = Image.new("RGB", (336, 336), IMAGENET_PIXEL_MEAN)
                    # paste region over blank image.
                    image.paste(crop_region, [x1, y1])
                    # save image.
                    image.save(
                        f'{args.output_image_folder}/{str(image_id).zfill(12)}_{str(category_id).zfill(3)}_{str(instance_id).zfill(12)}/W{str(x1).zfill(3)}-H{str(y1).zfill(3)}.jpg'
                    )
                    # save pope question.
                    f.write(
                        json.dumps(
                            {
                                'question_id': question_id,
                                'image': f'{str(image_id).zfill(12)}_{str(category_id).zfill(3)}_{str(instance_id).zfill(12)}/W{str(x1).zfill(3)}-H{str(y1).zfill(3)}.jpg',
                                'text': f'Is there a {category_name} in the image?',
                                'label': 'yes'
                            }
                        ) + '\n'
                    )
                    question_id += 1
                    image.close()
            coco_image.close()
            object_num += 1
            if object_num == args.max_object_num:
                break
        f.close()
            
    

if __name__ == '__main__':
    main()