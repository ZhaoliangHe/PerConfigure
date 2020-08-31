import object_inference as oi
import ffmpeg_code as fc
import config as conf
from multiprocessing import Pool, Manager


def parse_model_name(key):
    model_name = oi.MODELS[key]
    t = model_name.split("_")
    k = key.split("_")
    for i in range(len(t)):
        if "x" in t[i]:
            return t[i], k[0] + "_" + k[1]
    return None, None


def build_ground_truths():
    # here we use FasterRCNN InceptionResNetV2 1024p as the truth set
    gt_model_key = "rcnn_in_1024"
    gpu_id = 0
    fps = 30
    img_dir = conf.path['user_path'] + conf.path['project_path'] \
                 + conf.path['dataset_path'] + "_ground_truths"
    d_path = conf.path['user_path'] + conf.path['project_path'] + conf.path['dataset_path'] + ".mp4"
    fc.do_command(conf.path['ffmpeg_path'] + " -i " + d_path + " -s 1024x1024 " + img_dir + "/image-%3d.jpg")
    oi.main(gt_model_key, fps, "", gpu_id, img_dir, type=1)


if __name__ == "__main__":
    build_ground_truths()
    fps_list = conf.fps
    for j in range(len(fps_list)):
        t_fps = fps_list[j]
        nr_gpu = gpu_id = conf.nr_gpu
        models_keys = list(oi.MODELS.keys())
        p = Pool(nr_gpu)
        m = Manager()
        m_list = m.list()
        n = len(models_keys) - 1 # the last model cannot be used now
        for i in range(n):
            r, model_info = parse_model_name(models_keys[i])
            if r is not None:
                gpu_id -= 1
                r_frames, path = fc.ffmpeg_entry(model_info, t_fps, r)
                p.apply_async(oi.main, args=(models_keys[i], t_fps, r, gpu_id, path, m_list,))
            if gpu_id == 0 or i + 1 == n:
                p.close()
                p.join()
                gpu_id = nr_gpu
                p = Pool(nr_gpu)
    # if len(m_list) == n:
    #     ca.main()
    #     print("all work done!")