import os
from multiprocessing.pool import Pool
from tqdm import tqdm
import requests
import pandas as pd
def download(pid, sp, ep):
    colors = ['red', 'green', 'blue', 'yellow']
    DIR = "/mnt/bigdisk/hpa_site_data/"
    v18_url = 'http://v18.proteinatlas.org/images/'
    imgList = pd.read_csv("/mnt/bigdisk/hpa_site_data/HPAv18RBGY_wodpl.csv")
    for i in tqdm(imgList['Id'][sp:ep], postfix=pid):  # [:5] means downloard only first 5 samples, if it works, please remove it
        img = i.split('_')
        for color in colors:
            img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
            img_name = i + "_" + color + ".jpg"
            img_url = v18_url + img_path
            r = requests.get(img_url, allow_redirects=True)
            open(DIR + img_name, 'wb').write(r.content)

def run_proc(name, sp, ep):
    print('Run child process %s (%s) sp:%d ep: %d' % (name, os.getpid(), sp, ep))
    download(name, sp, ep)
    print('Run child process %s done' % (name))

if __name__ == "__main__":
    print('Parent process %s.' % os.getpid())
    img_list = pd.read_csv("/mnt/bigdisk/hpa_site_data/HPAv18RBGY_wodpl.csv")['Id']
    list_len = len(img_list)
    process_num = 10
    p = Pool(process_num)
    for i in range(process_num):
        p.apply_async(run_proc, args=(str(i), int(i * list_len / process_num), int((i + 1) * list_len / process_num)))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
