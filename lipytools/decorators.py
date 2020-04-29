"""

 2020 (c) piteren

    some basic / useful decorators

"""

import time

# decorator printing time report
def timing(f):
    def new_f(*args, **kwargs):
        stime = time.time()
        f(*args, **kwargs)
        print(f'(@timing) {f.__name__} finished, taken {time.time() - stime:.1f}sec')
    new_f.__name__ = f'{f.__name__}:@timing'
    return new_f