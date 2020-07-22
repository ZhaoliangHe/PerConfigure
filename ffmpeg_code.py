import os
import argparse
from subprocess import call

# you can use
# python ffmpeg_code.py -h #查看可传入参数
# python ffmpeg_code.py -r 15 -s 1280x720

ffmpeg_path = "/home/hezhaoliang/ffmpeg/ffmpeg-4.2.3/ffmpeg" # instead of yours

def videotoimages(fps,resolution,ffmpeg_path):
    path = "./dataset/youtube/"
    images_file = '11images' + '_' + fps + '_' + resolution
    image_path = path + images_file + '/image-%4d.png'
    call(["mkdir", path + images_file])
    call([ffmpeg_path, "-i", os.path.join(path, "11SecondVideo.mp4"), "-r", fps, "-s", resolution, "-f", "image2",
          image_path])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ffmpeg')
    parser.add_argument("-r", help="fps", default="15", type=str)
    parser.add_argument("-s", help="resolution", default="1280x720", type=str)
    args = parser.parse_args()
    fps = args.r
    resolution = args.s
    videotoimages(fps,resolution,ffmpeg_path)

