import os
import errno


def mkdir_if_missing(dir_path):
    """
    新建文件夹
    """
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
