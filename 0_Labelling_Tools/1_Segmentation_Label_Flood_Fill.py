from __future__ import print_function

from pathlib import Path
from PIL import Image
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'input_dir', help='input label directory produced by EISeg')
    parser.add_argument(
        'output_dir', help='output YOLOv5 bounding box label directory')
    return parser.parse_args()


def flood_fill(x, y, bx, ex, by, ey):
    nexts = [(1, 0, 0, 1, 0, 0), (-1, 0, 1, 0, 0, 0),
             (0, 1, 0, 0, 0, 1), (0, -1, 0, 0, 1, 0)]
    for next_x, next_y, off_bx, off_ex, off_by, off_ey in nexts:
        new_x = x + next_x
        new_y = y + next_y

        if new_x < 0 or new_x >= np_data.shape[0] or new_y < 0 or new_y >= np_data.shape[1]:
            continue
        if np_data[new_x][new_y] != 1:
            continue

        np_data[new_x][new_y] = 0

        pbx, pex, pby, pey = flood_fill(
            new_x, new_y, new_x + off_bx, new_x + off_ex, new_y + off_by, new_y + off_ey)

        bx = min(bx, pbx)
        ex = max(ex, pex)
        by = min(by, pby)
        ey = max(ey, pey)

    return bx, ex, by, ey


def segmentation_label_flood_fill(args):
    global np_data

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in list(input_dir.glob('*.png')):

        if '_cutout' in filename.stem or '_pseudo' in filename.stem:
            continue

        print(f'now processing {filename}.')

        data = Image.open(filename)
        data = data.convert('L')
        data.show()
        np_data = np.array(data)
        np_data[np_data != 0] = 1

        bbox = []

        itemindex = np.where(np_data == 1)
        while len(itemindex[0]) != 0:
            bx, ex, by, ey = flood_fill(
                itemindex[0][0], itemindex[1][0], itemindex[0][0], itemindex[0][0], itemindex[1][0], itemindex[1][0])
            bbox.append([bx, ex, by, ey])
            itemindex = np.where(np_data == 1)

        output_file_path = (output_dir / filename.name).with_suffix('.txt')

        classes_file_path = output_dir / 'classes.txt'
        classes_file_path.unlink(missing_ok=True)
        with open(classes_file_path, 'a') as the_file:
            the_file.write('precipitate\n')

        output_file_path.unlink(missing_ok=True)
        with open(output_file_path, 'a') as the_file:
            img_x, img_y = np_data.shape
            for bx, ex, by, ey in bbox:
                x_center = ((bx + ex) / 2) / img_x
                y_center = ((by + ey) / 2) / img_y
                width = (ex - bx) / img_x
                height = (ey - by) / img_y

                the_file.write(
                    f'0 {round(y_center, 6)} {round(x_center, 6)} {round(height, 6)} {round(width, 6)}\n')


if __name__ == '__main__':
    args = parse_args()
    segmentation_label_flood_fill(args)
