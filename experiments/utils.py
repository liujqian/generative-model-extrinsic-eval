import datetime


def log_progress(cur_idx: int, total: int, log_every: int = 1, additional_info: str = ""):
    if cur_idx % log_every == 0:
        s = f"{str(datetime.datetime.now())}: Processing the {cur_idx}th item, there are {total} items in total. {additional_info}"
        print(s)
