"""

 2020 (c) piteren

    multiprocessing decorators

"""

from multiprocessing import Process, Queue
from functools import partial
import time


# subprocess decorator, runs decorated function in subprocess
def proc(f):

    def new_f(*args, **kwargs):
        Process(target=partial(f, *args, **kwargs)).start()

    new_f.__name__ = f'{f.__name__}:@proc'
    return new_f

# subprocess decorator, runs decorated function in subprocess, but holds main process till join
def proc_wait(f):

    def new_f(*args, **kwargs):
        p = Process(target=partial(f, *args, **kwargs))
        p.start()
        p.join()

    new_f.__name__ = f'{f.__name__}:@proc_wait'
    return new_f

# helper class (process with que) for qproc
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

# subproces decorator, runs decorated function in subprocess, result will be put on que
def qproc(que):
    def wrap(f):

        def new_f(*args, **kwargs):
            p = MProc(que=que, f=f, args=args, kwargs=kwargs)
            p.start()

        new_f.__name__ = f'{f.__name__}:@qproc'
        return new_f

    return wrap


# ****************************************************************************************************** examples

def example_proc():

    @proc
    def task(to=5):
        for i in range(to):
            print(i, 'done')
            time.sleep(0.1)

    task(to=10)
    task(10)
    task(10)


def example_proc_wait():

    @proc_wait
    def task(to=5):
        for i in range(to):
            print(i, 'done')
            time.sleep(0.1)

    task(to=10)
    task(10)
    task(10)


def example_qproc():

    import random

    que = Queue()
    @qproc(que)
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

    #example_proc()
    #example_proc_wait()
    example_qproc()