#!/usr/bin/python3

import os
import argparse
import json
import glob
from tqdm import tqdm
import cv2
import torch

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--images-dir", type=str, default="/share/private/27th/hirotaka_saito/Images/")
    parser.add_argument("-o", "--output-dir", type=str, default="/share/private/27th/hirotaka_saito/dataset/Images/")
    parser.add_argument("-c", "--num", type=int, default="1")
    args = parser.parse_args()

    torch_imgs = []

    i = 0
    j = args.num

    for img_path in tqdm(glob.glob(os.path.join(args.images_dir,"*"))):
        img = cv2.imread(img_path)
        torch_img = torch.tensor(img)
        h,w,c = torch_img.shape
        if h != 256 or w != 256 or c != 3:
            print("not size")
        torch_imgs.append(torch_img)

        if i % args.num == 0:
            path = os.path.join(args.output_dir, str(j) + ".pt")

            with open(path,"wb") as f:
                torch.save(torch_imgs,f)
            torch_imgs = []
            j += args.num

        i += 1

if __name__ == "__main__":
    main()
