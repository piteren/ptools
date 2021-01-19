"""

 2020 (c) piteren

    multiprocessing decorators >> run decorated functions in subprocess

    @proc
        - does not hold till finish
        - does not return any value

    @proc_wait
        - holds main process till join

    @proc_return
        - holds main process till join
        - returns function result (using que)

    @proc_que(que)
        - does not hold till finish
        - will put function result on given que

"""

from multiprocessing import Process, Queue
from functools import partial
import time


def proc(f):

    def new_f(*args, **kwargs):
        Process(target=partial(f, *args, **kwargs)).start()

    new_f.__name__ = f'@proc:{f.__name__}'
    return new_f


def proc_wait(f):

    def new_f(*args, **kwargs):
        p = Process(target=partial(f, *args, **kwargs))
        p.start()
        p.join()

    new_f.__name__ = f'@proc_wait:{f.__name__}'
    return new_f

# helper class (process with que - puts target result on que)
class MProc(Process):

    def __init__(
            self,
            que :Queue,
            f,
            args,
            kwargs):

        super().__init__(target=self.proc_m)
        self.que = que
        self.f = f
        self.ag = args
        self.kw = kwargs

    def proc_m(self):
        res = self.f(*self.ag, **self.kw)
        self.que.put(res)


def proc_return(f):
    def new_f(*args, **kwargs):
        que = Queue()
        p = MProc(que=que, f=f, args=args, kwargs=kwargs)
        p.start()
        ret = que.get()
        return ret
    new_f.__name__ = f'@proc_return:{f.__name__}'
    return new_f


def proc_que(que):
    def wrap(f):

        def new_f(*args, **kwargs):
            p = MProc(que=que, f=f, args=args, kwargs=kwargs)
            p.start()

        new_f.__name__ = f'@qproc:{f.__name__}'
        return new_f

    return wrap


# *********************************************************************** examples

def example_proc():

    @proc
    def task(to=5):
        for i in range(to):
            print(i, 'done')
            time.sleep(0.1)

    print(task.__name__)
    task(to=10)
    task(10)
    task(10)

    @proc
    def calc(n):
        val = 2 * n
        print(f'will return {val}')
        return val

    print(f'returned {calc(3)}')
    print(f'returned {calc(4)}')


def example_proc_wait():

    @proc_wait
    def task(to=5):
        for i in range(to):
            print(i, 'done')
            time.sleep(0.1)

    task(to=10)
    task(10)
    task(10)


def example_proc_return():
    @proc_return
    def calc(n):
        val = 2 * n
        print(f'will return {val}')
        return val

    print(f'returned {calc(3)}')
    print(f'returned {calc(4)}')


def example_proc_que():

    import random

    que = Queue()
    @proc_que(que)
    def task(name='def', to=5):
        sum = 0
        for i in range(to):
            print(f'calculating sum ({to})...')
            time.sleep(0.5)
            sum += i
        return name, sum

    print(task.__name__)

    n = 50
    for i in range(n): task(name=f'task_{i}', to=random.randrange(5,20))
    for _ in range(n): print(f'calculated result: {que.get()}')


if __name__ == '__main__':

    example_proc()
    #example_proc_wait()
    #example_proc_return()
    #example_proc_que()