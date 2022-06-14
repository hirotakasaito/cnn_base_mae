import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import animation

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2,im3,im4):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height + im3.height + im4.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    dst.paste(im3, (0, im1.height + im2.height))
    dst.paste(im4, (0, im1.height + im2.height + im3.height))
    return dst

def get_concat_h_multi(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_h(_im, im)
    return _im

def save_video_as_gif(frames, interval=150, file_name="video_prediction.gif", title=""):
    """
    make video with given frames and save as "video_prediction.gif"
    """
    plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        plt.title(title + ' \n Step %d' % (i))

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval)
    anim.save(file_name, writer='imagemagick')
