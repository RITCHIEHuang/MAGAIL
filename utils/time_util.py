#!/usr/bin/env python
# Created at 2020/2/15
import time
from functools import wraps

__all__ = ['timer']


def timer(message=None, show_result=False):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time_start = time.time()
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_start))}, Start {func.__name__}()".center(80,
                                                                                                                    "*"))

            res = func(*args, **kwargs)

            if show_result:
                print("Result: ")
                print(res)

            if message:
                print(message)
            print(f"After {(time.time() - time_start)} s".center(80, " "))

            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}, End {func.__name__}()".center(80,
                                                                                                                   "*"))
            return res

        return wrapper

    return decorate
