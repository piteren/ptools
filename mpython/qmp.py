"""

 2020 (c) piteren

"""

from inspect import getfullargspec
from multiprocessing import Queue, Process, cpu_count
import numpy as np
import random
import time
from typing import Callable
from queue import Empty

from ptools.mpython.mpdecor import qproc
from ptools.lipytools.little_methods import short_scin


# internal processor for DeQueMP
class InternalProcessor(Process):

    def __init__(
            self,
            func :Callable,     # function to run
            tq :Queue,
            rq :Queue,
            devices :list,
            sorted=     False):

        Process.__init__(self, target=self.__run)

        self.func = func
        ins = getfullargspec(self.func)
        self.device_for_func = False
        if 'device' in ins.args: self.device_for_func = 'device'
        if 'devices' in ins.args: self.device_for_func = 'devices'

        self.tq = tq
        self.rq = rq
        self.devices = devices
        self.n_devices = len(self.devices)
        self.inq = Queue() # internal que for subprocesses

        self.sorted = sorted

    # wraps func
    def __proc_task(self, task, taskIX, device):

        @qproc(self.inq)
        def wrap(task, taskIX, device):
            if self.device_for_func: task[self.device_for_func] = device
            res = self.func(**task)
            return {
                'res':      res,
                'taskIX':   taskIX,
                'device':   device}

        wrap(task, taskIX, device)

    # process loop
    def __run(self):

        task_counter = 0
        last_processed_task_IX = -1
        res_cache = {}
        while True:

            #print(f'D:{len(self.devices)} T:{self.tq.qsize()} R:{self.inq.qsize()}')

            if self.devices:

                # try get task
                task = None
                try: task = self.tq.get_nowait()
                except Empty: pass

                # poison case / exit loop
                if task == 'poison':
                    while len(self.devices) < self.n_devices: self.devices.append(self.inq.get()['device'])
                    self.rq.put('finished')
                    break

                # process task
                if task is not None:
                    device = self.devices.pop(0)
                    taskIX = task_counter
                    self.__proc_task(task, taskIX, device)

                    task_counter += 1

            # look for ready result
            result = None
            try: result = self.inq.get_nowait()
            except Empty: pass

            if not self.devices and result is None: result = self.inq.get() # have to wait for result now

            # process result
            if result is not None:

                res =    result['res']
                taskIX = result['taskIX']
                self.devices.append(result['device'])

                if self.sorted:
                    # got res in order
                    if last_processed_task_IX + 1 == taskIX:
                        last_processed_task_IX += 1
                        self.rq.put(res)

                        # take from cache
                        while last_processed_task_IX + 1 in res_cache:
                            last_processed_task_IX += 1
                            self.rq.put(res_cache.pop(last_processed_task_IX))

                    else: res_cache[taskIX] = res # put to cache

                else: self.rq.put(res)

# decorated que multi processor
class DeQueMP:

    def __init__(
            self,
            func :Callable,                 # function to run
            devices=                None,   # devices to use
            use_cores :int or True= True,   # True: when devices is None >> uses all cores, otherwise as set by int or devices
            user_tasks=             False,  # user_tasks mode, in user_tasks mode user sends kwargs dicts (via put_task)
            rq_trg=                 500,    # r_que target size (if not user_tasks mode), won't grow above
            name=                   'dqmp',
            sorted=                 False,  # returns results sorted by task_IX
            verb=                   0):

        self.verb = verb
        self.name = name

        # manage special case of devices
        if devices is None:
            if use_cores is True: use_cores = cpu_count()
            devices = [None] * use_cores
        self.n_devices = len(devices)

        if verb > 0: print(f'\n{self.name} (DeQueMP) starts for {func.__name__} on {self.n_devices} devices')

        self.tq = Queue()
        self.rq = Queue()

        self.ip = InternalProcessor(
            func=       func,
            tq=         self.tq,
            rq=         self.rq,
            devices=    devices,
            sorted=     sorted)

        self.user_tasks = user_tasks

        self.ip.start()

        if not self.user_tasks:
            for _ in range(rq_trg):
                self.__put_self_task()


    # puts task in not user_tasks mode
    def __put_self_task(self): self.tq.put({})

    def put_task(self, task :dict):
        assert self.user_tasks, 'ERR: user cannot put_task not in user_tasks mode!'
        self.tq.put(task)

    def get_result(self):
        result = self.rq.get()
        if not self.user_tasks: self.__put_self_task()
        return result

    def close(self):
        self.tq.put('poison')
        while True:
            result = self.rq.get()
            if result == 'finished': break

# multiprocessing accessible data, np.array wont explode RAM usage by processes in contrast to list or dict, but is very sensitive to sentence length (long sentences will explode mem usage @np.array)
class MPData:

    def __init__(
            self,
            fileFP=     None,   # full path to file with text lines
            data=       None,   # data in form of iterable (list, dict, etc)
            name=       'MPData',
            verb=       0):

        self.verb =verb
        self.name = name

        if self.verb > 0: print('\n%s loading file data...' % self.name)
        self.len, self.data = 0, None

        new_data = data
        if fileFP:
            with open(fileFP, 'r') as file:
                new_data = [line[:-1] for line in file]

        self.len = len(new_data)
        self.data = np.asarray(new_data)

        print(self.data.shape)
        print(self.data.dtype)

        if self.verb > 0: print('\n%s got %d (%s) lines of data' % (self.name, self.len, short_scin(self.len)))

    # returns line indexed with num from data
    def get_data(
            self,
            num=    None):      # None returns random

        if num is None: num = np.random.randint(self.len)
        if num < self.len: return self.data[num]
        return None


def example_dqmp_self():

    def func(device=None):
        time.sleep(random.random()*1)
        print(f'processed with {device}')
        return device

    dqmp = DeQueMP(
        func=       func,
        devices=    [1,2,3,4],
        rq_trg=     10)

    for _ in range(100):
        res = dqmp.get_result()
        print(res)
    dqmp.close()


def example_dqmp_self_nodev():

    def func():
        time.sleep(random.random()*1)
        print(f'processed')
        return True

    dqmp = DeQueMP(
        func=       func,
        devices=    [1,2,3,4],
        sorted=     True,
        rq_trg=     10)

    for _ in range(100):
        res = dqmp.get_result()
        print(res)
    dqmp.close()


def example_dqmp_self_all():

    def func():
        time.sleep(random.random()*1)
        print(f'processed')
        return True

    dqmp = DeQueMP(
        func=       func,
        devices=    None,
        sorted=     True,
        rq_trg=     100,
        verb=       1)

    for _ in range(1000):
        res = dqmp.get_result()
        print(res)
    dqmp.close()


def example_dqmp_user():

    def func(
            pa,
            pb,
            device=None):
        time.sleep(random.random()*1)
        print(f'processed {pa}/{pb} with {device}')
        return pa, pb, device

    dqmp = DeQueMP(
        func=       func,
        devices=    [1,2,3,4],
        user_tasks= True)

    for _ in range(100):
        dqmp.put_task({'pa': random.randrange(100), 'pb': random.randrange(100)})

    for _ in range(100):
        res = dqmp.get_result()
        print(res)
    dqmp.close()


if __name__ == '__main__':

    #example_dqmp_self()
    #example_dqmp_self_nodev()
    example_dqmp_self_all()
    #example_dqmp_user()