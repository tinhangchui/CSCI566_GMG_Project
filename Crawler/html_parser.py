"""
Prase HLML and return new url or midi
"""

import re
import requests

def _parse(page_url):
    """
    parse page specified by page_url
    :param page_url:
    :return: parsed data
    """
    r = requests.get(page_url)
    data = r.text
    return data


def get_new_urls(page_url, next_layer=True):
    """
    find a list of urls from page and do screening
    :param page_url: get urls from this page
    :param re: regular statement for screening urls
    :return: url list from page_data
    """
    data = _parse(page_url)
    page_url += "/"
    line_list = re.findall(r"https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", data)
    url_list = []
    if next_layer:
        for line in line_list:
            if line.startswith(page_url) and re.match(r'^((?!/).)*$', line.split(page_url)[-1]):
                url_list.append(line)
    else:
        url_list = line_list
    return url_list


def get_new_data(page_url, source_regular):
    """
    find a list of source urls from page
    :param page_url: find source url on this page
    :param source_regular: regular statement for screening source url
    :return: list of source url
    """
    data = _parse(page_url)
    line_list = re.findall(r"https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", data)
    url_list = []
    for line in line_list:
        if line.endswith(source_regular):
            url_list.append(line)
    return url_list