import os
import numpy as np
import json
import cv2


DATA_PATH = '/media/a2mind/Backup/OC_SORT_datasets/camel_dataset'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
# SPLITS = ['train', 'val', 'test']
SPLITS = ['train', "val", "test"]

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        data_path = os.path.join(DATA_PATH, split)
        out_ccd_path = os.path.join(OUT_PATH, '{}_ccd.json'.format(split))
        out_ir_path = os.path.join(OUT_PATH, '{}_ir.json'.format(split))

        out_ccd = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'person'}]}
        out_ir = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'person'}]}
        
        seqs = os.listdir(data_path)
        
        image_cnt = 0
        ann_cnt = 0
        ann_ir_cnt = 0
        image_ir_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            if '.DS_Store' in seq or '.ipy' in seq:
                continue

            video_cnt += 1  # video sequence number.

            out_ccd['videos'].append({'id': video_cnt, 'file_name': seq})
            out_ir['videos'].append({'id': video_cnt, 'file_name': seq})
            
            seq_path = os.path.join(data_path, seq)
            ccd_img_path = os.path.join(seq_path, 'ccd')
            ir_img_path = os.path.join(seq_path, 'ir') 

            ann_ccd_path = os.path.join(seq_path, '{}-Vis.txt'.format(seq))
            ann_ir_path = os.path.join(seq_path, '{}-IR.txt'.format(seq))
            
            images = os.listdir(ccd_img_path)
            num_images = len([image for image in images if 'jpg' in image])  # half and half

            for i in range(num_images):
                img = cv2.imread(os.path.join(data_path, '{}/ccd/{:06d}.jpg'.format(seq, i + 1)))
                img_ir = cv2.imread(os.path.join(data_path, '{}/ir/{:06d}.jpg'.format(seq, i + 1)))
                height, width = img.shape[:2]
                image_info = {'file_name': '{}/ccd/{:06d}.jpg'.format(seq, i + 1),  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1,  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height,
                              'width': width}
                ir_info = {'file_name': '{}/ir/{:06d}.jpg'.format(seq, i + 1),  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1,  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height,
                              'width': width}
                out_ccd['images'].append(image_info)
                out_ir['images'].append(ir_info)
            print('{}: {} images'.format(seq, num_images))

            if split != 'test':
                anns = np.loadtxt(ann_ccd_path, dtype=np.float32, delimiter='\t')
                anns_ir = np.loadtxt(ann_ir_path, dtype=np.float32, delimiter='\t')
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][2])
                    ann_cnt += 1
                    category_id = 1
                    ann = {'id': ann_cnt,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': track_id,
                           'bbox': anns[i][3:7].tolist(),
                           'conf': 1.0,
                           'iscrowd': 0,
                           'area': float(anns[i][4] * anns[i][5])}
                    out_ccd['annotations'].append(ann)
                for i in range(anns_ir.shape[0]):
                    frame_id = int(anns_ir[i][0])
                    track_id = int(anns_ir[i][1])
                    cat_id = int(anns_ir[i][2])
                    ann_cnt += 1
                    category_id = 1
                    ann = {'id': ann_cnt,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': track_id,
                           'bbox': anns_ir[i][3:7].tolist(),
                           'conf': 1.0,
                           'iscrowd': 0,
                           'area': float(anns_ir[i][4] * anns_ir[i][5])}
                    out_ir['annotations'].append(ann)

                print('{}: {} ann images'.format(seq, int(anns[:, 0].max())))
                print('{}: {} ann images'.format(seq, int(anns_ir[:, 0].max())))

            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(split, (len(out_ccd['images']), len(out_ir['images'])), (len(out_ccd['annotations']), len(out_ir['annotations']))))
        json.dump(out_ccd, open(out_ccd_path, 'w'))
        json.dump(out_ir, open(out_ir_path, 'w'))