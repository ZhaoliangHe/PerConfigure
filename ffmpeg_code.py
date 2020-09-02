import os
<<<<<<< HEAD
import argparse
from subprocess import call

# you can use
# python ffmpeg_code.py -h #查看可传入参数
# python ffmpeg_code.py -r 15 -s 1024x1024 640x640 320x320
# //剪切视频
# ffmpeg -ss 0:0:2 -t 0:0:10 -i demo1.mp4 -vcodec copy -acodec copy demo1_10s.mp4
# // -ss 开始时间; -t 持续时间
ffmpeg_path = "/home/hezhaoliang/ffmpeg/ffmpeg-4.2.3/ffmpeg" # instead of yours

def videotoimages(fps,resolution,ffmpeg_path):
    path = "./dataset/youtube/"
    # images_file = '11images' + '_' + fps + '_' + resolution
    images_file = 'demo1_10s' + '_' + fps + '_' + resolution
    image_path = path + images_file + '/image-%d.jpg'
    # input_video = "11SecondVideo.mp4"
    input_video = "demo1_10s.mp4"
    call(["mkdir", path + images_file])
    call([ffmpeg_path, "-i", os.path.join(path, input_video), "-r", fps, "-s", resolution, "-f", "image2",
          image_path])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ffmpeg')
    parser.add_argument("-r", help="fps", default="15", type=str)
    parser.add_argument("-s", help="resolution", default="640x640", type=str)
    args = parser.parse_args()
    fps = args.r
    resolution = args.s
    videotoimages(fps,resolution,ffmpeg_path)

=======
import config as conf
import argparse
ffmpeg = conf.path['ffmpeg_path']
ffprobe = conf.path['ffprobe_path']


def nr_file(dir):
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        for file in filenames:
            file_count = file_count + 1
    return file_count


def str2fps(r):
    return int(round((int(r[0]) / int(r[1])), 0))


def do_command(c):
    ret = os.popen(c)
    result = ret.read()
    ret.close()
    return result


def delta():
    # the original function in ffmpeg is aimed to calculate the number of frame
    # here we define it to ...
    pass


def delta0(n, last_opts, o_fps, t_fps):
    return n * t_fps / o_fps - last_opts


def get_file_size(path):
    try:
        size = os.path.getsize(path)
        return size
    except Exception as err:
        print(err)


def cmp_file(a, b):
    a_size = get_file_size(a)
    b_size = get_file_size(b)
    return a_size == b_size


def cal_r_frames(t_fps, o_fps, nr_frames):
    lop = 0
    last_opts = 1
    frames = []

    for i in range(1, nr_frames + 1):
        k = round(delta0(i, last_opts, o_fps, t_fps), 1)
        # print(k)
        if k <= -0.6:  # ffmpeg think this frame should be abandoned
            lop += 1
            frames.append(i)
        else:
            last_opts += 1
    return frames


def re_encode(tmp_file, t_fps, r, model_name, video_info):
    file = tmp_file + ".mp4"
    path = tmp_file + "_" + model_name + "_" + str(t_fps) + "_" + r
    do_command("mkdir " + path)

    if not os.path.exists(path + "/opt.mp4"):
        if t_fps != video_info['fps']:
            do_command(ffmpeg + " -i " + file +
                       " -vcodec libx264 -vf fps=" + str(t_fps) + " -bf 0 -g " + str(
                t_fps) + " -pix_fmt yuv420p " + path + "/opt.mp4")
        else:
            do_command("cp " + file + " " + path + "/opt.mp4")

        do_command(ffmpeg + " -i " + path + "/opt.mp4" +
                   " -s " + r + " " + path + "/image-%3d.jpg")
    # o_fps = video_info['fps']
    # nr_frames = video_info['frames']
    # lop = 0
    # last_opts = 1
    # frames = []
    # for i in range(1, nr_frames + 1):
    #     k = round(delta0(i, last_opts, o_fps, t_fps), 1)
    #     # print(k)
    #     if k <= -0.6:  # ffmpeg think this frame should be abandoned
    #         lop += 1
    #         frames.append(i)
    #     else:
    #         last_opts += 1
    frames = cal_r_frames(t_fps, video_info['fps'], video_info['frames'])
    return frames, path


def non_re_encode(tmp_path, t_fps, video_info):
    file = tmp_path + ".mp4"
    do_command(ffmpeg + " -i " + file + " -r " + t_fps +
               " -f image2 " + t_fps + "/image-%3d.png")
    do_command(ffmpeg + " -i " + file + " -r " +
               str(video_info['fps']) + " -f image2 " + str(video_info['fps']) + "/image-%3d.png")
    frames = []
    k = 1
    i = 1
    error = 0
    nr_frames = nr_file(t_fps)
    while i < nr_frames + 1:
        if not cmp_file(t_fps + "/image-" + str(k).zfill(3) + ".png",
                        str(video_info['fps']) + "/image-" + str(k + error).zfill(3) + ".png"):
            frames.append(k + error)
            k = i
            error += 1
        else:
            k += 1
        i += 1
    return frames


def get_video_info(path):
    video_info = {}
    file = path + ".mp4"
    c = ffprobe + " -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 " + file
    video_info['frames'] = int(do_command(c))
    r = (do_command(
        ffprobe + " -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate " + file)).split(
        "/")
    video_info['fps'] = str2fps(r)
    return video_info


def ffmpeg_entry(model_name, s, r):
    t_fps = s
    global ffmpeg, ffprobe
    video_file = conf.path['user_path'] + conf.path['project_path'] \
                 + conf.path['dataset_path']
    video_info = get_video_info(video_file)
    r_frames, path = re_encode(video_file, t_fps, r, model_name, video_info)
    # n_frames = non_re_encode(video_file, str(t_fps))

    # print("re encode:")
    # for i in range(len(r_frames)):
    #     print(r_frames[i], end=',')
    #
    # print()
    #
    # print("none encode:")
    # for i in range(len(n_frames)):
    #     print(n_frames[i], end=',')
    return r_frames, path
>>>>>>> chentang
