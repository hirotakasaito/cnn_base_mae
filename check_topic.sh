#!/bin/bash

topic_list=("/usb_cam/")

topic_list_size=${#topic_list[*]}
echo "=====Not enough data===="

for file in $(find /share/bagfiles/2019/cats/tsukuba/0914/ -maxdepth 1 -type f -name "*.bag"); do
    count=0

    for i in "${topic_list[@]}";do
        result=$(rosbag info $file | grep $i)

        if [ -n "$result" ];then
            count=$(expr $count + 1)
        fi
    done
    if [ $count -lt $topic_list_size ];then
         mv $file ./trash/
         echo $file
    fi
done

e
