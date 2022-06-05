#!/usr/bin/env python3
import os
import argparse
import json
from tqdm import tqdm
import cv2

import torch

from util.rosbaghandler import RosbagHandler
from util.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.json")
    args = parser.parse_args()


    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("cannot find config file")

    for bagfile_name in config["bagfile_name"]:
        bagfile = os.path.join(config["bagfile_dir"], bagfile_name)
        if not os.path.exists(bagfile):
            raise ValueError('set bagfile')
        file_name = os.path.splitext(os.path.basename(bagfile))[0]
        rosbag_handler = RosbagHandler(bagfile)
        t0 = rosbag_handler.start_time
        t1 = rosbag_handler.end_time

        sample_data = rosbag_handler.read_messages(topics=config["topics"], start_time=t0, end_time=t1, hz=config["hz"])
        for topic in sample_data.keys():
            topic_type = rosbag_handler.get_topic_type(topic)
            if topic_type == "sensor_msgs/CompressedImage":
                print("==== convert compressed image ====")
                images = convert_CompressedImage(sample_data[topic], config["height"], config["width"])
        print("==== converted rosbag to jpeg ====")

        for idx,img in enumerate(images):
            img_name = file_name + str(idx) + '.jpg'
            save_name = os.path.join(config["output_dir"],img_name)
            cv2.imwrite(save_name, img)




