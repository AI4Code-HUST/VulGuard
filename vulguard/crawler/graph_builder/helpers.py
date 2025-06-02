import datetime
import logging
import re
import subprocess
from threading import Timer

import hashlib
import sys
import time
from itertools import chain, combinations
import urllib.parse

import Levenshtein

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')


def get_logger(name):
    return logging.getLogger(name)


logger = get_logger(__name__)


def sleep(secs):
    logger.info("Sleeping in {} secs".format(secs))
    time.sleep(secs)


def hash_md5(input_str):
    return hashlib.md5(input_str.encode()).hexdigest()


def hash_to_int(input_str):
    return int(hashlib.sha1(input_str.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def get_current_timestamp():
    return round(time.time())


def get_version_by_time():
    return datetime.datetime.today().strftime('%Y%m%d_%H%M%S')


def find_plugin_name(command):
    matched = re.search("[^ /]+\.(jar|sh)", command)
    if matched:
        return matched.group()


def subprocess_cmd(command: str):
    """Run command line process.

    Example:
    subprocess_cmd('echo a; echo b', verbose=1)
    """
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    timer = Timer(60*10, process.kill)
    try:
        timer.start()
        stout, stderr = process.communicate()
    except:
        logger.warning(f"Error when run command: {command[:100]}")
        timer.cancel()
        raise RuntimeError("Error when run command: timeout!!")
    finally:
        timer.cancel()

    return stout.decode(), stderr.decode()


def natural_sort(l):
    # in case: sort by file name without considering full path
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key.rsplit("/", 1)[1])]
    return sorted(l, key=alphanum_key)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def remove_whitespace(line_str):
    return ''.join(line_str.strip().split())


def compute_line_ratio(line_str1, line_str2):
    l1 = remove_whitespace(line_str1)
    l2 = remove_whitespace(line_str2)
    return Levenshtein.ratio(l1, l2)


def encode_special_characters_with_html_rules(input_str):
    return urllib.parse.quote(input_str)
