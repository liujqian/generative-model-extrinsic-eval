import datetime


def log_progress(cur_idx: int, total: int, additional_info: str):
    s = f"{str(datetime.datetime.now())}: Processing the {cur_idx}th item, there are {total} items in total. {additional_info}"
    print(s)
