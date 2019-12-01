from html_parser import *
from config import *
from data_storage import *

def spider_man_mdidi(reuse=False, file_path_name="", save_path="./data"):
    """
    DFS traverse webpages with cutoff and download midi files to specified path
    :param reuse: use a persistanced url-list file to down load midi files
    :param file_path_name: path and file name of url-list file
    :save_path: the path of saving midi files
    """
    if not reuse:
        # initialize
        s_url_list = []
        new_url_list = []
        visited_url_list = []
        new_url_list.append(ROOT_URL)
        # dfs
        iter = 1
        while len(new_url_list) > 0:
            print("iter:" + str(iter) + "  remain url number:" + str(len(new_url_list)))
            iter += 1
            url = new_url_list[0]
            new_url_list.remove(url)

            if url not in visited_url_list:
                visited_url_list.append(url)
                if (url.count("/") - 3) <= CUTOFF:
                    s_url_list += get_new_data(url, ".mid")
                    if (url.count("/") - 3) < CUTOFF:
                        new_url_list += get_new_urls(url)
        # save
        save_list(s_url_list, save_path, "s_url_list.txt")
        save_midi(s_url_list, save_path)
    else:
        s_url_list = load_list(file_path_name)
        save_midi(s_url_list, save_path)

spider_man_mdidi(True, file_path_name="./data/s_url_list.txt")