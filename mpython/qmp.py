"""

 2018 (c) piteren

 QueMultiProcessor processes tasks with (internal) subprocesses.

 Task taken from tQue will be passed as an argument to iProcFunction.
 Result of processing will be put on rQue.

 ### arguments:

 iProcFuction
    - must accept task as an argument (may not use...)
    - returns result
    - if result is None then all next results should be None

 user_tasks (mode)
    - assumes tasks (objects) are placed with putTask() method by user
    - if not userTasks QMP takes tasks from TaskGenerator object

 rQueTSize
    - QMP will try to maintain the size of rQue to this amount of results (in selfTask mode)
    - QMP in selfTask mode puts selfTaskIndex on tQue (user cannot then)

"""

from multiprocessing import Queue, Process, cpu_count
import numpy as np
import os
import random
from typing import Callable, List

from ptools.lipytools.little_methods import short_scin

# decorated que multi processor
class DeQueMP:

    def __init__(
            self,
            func :Callable,             # function to run
            devices=        None,       # devices to use
            use_all_cores=  True,       # True: when devices is None >> uses all cores, otherwise as set by devices
            user_tasks=     False,      # user_tasks mode, in user_tasks mode user sends kwargs dicts (via put_task)
            rq_trgsize=     500,        # r_que target size (if not user_tasks mode), won't grow above
            name=           'qmp',
            sorted_res=     False,      # returns results sorted by task_IX
            verb=           0):

        self.verb = verb
        if devices is None: devices = [None] * (cpu_count() if use_all_cores else 1)  # manage case of None for devices
        self.n_devices = len(devices)
        if verb > 0: print(f'\nDeQueMP starting for {func.__name__} on {self.n_devices} devices')

        self.task_IX = 0

    def put_task(self):
        pass

    def get_result(self):
        pass

    def close(self):
        pass

# default task generator
class TaskGenerator:

    def __init__(self):
        self.IX = -1

    # returns task
    def get_task(self):
        self.IX += 1
        return self.IX

# TODO: refactor and test
# QMP
class QueMultiProcessor:

    poison =  'poison'          # kill task

    # QMP states
    class QMPStates:
        init =      'init'      # constructor state
        running =   'running'   # constructor finished, QMP is ready and running
        killing =   'killing'   # self.__halt() called, received None as result.
        closed =    'closed'    # closed

    def __init__(
            self,
            proc_func,                          # function (method) to perform by iProc - must take task (task=None)
            task_object=            None,       # single object to be given 4 iProcFunction
            n_proc=                 0,          # number of subprocesses to launch, 0 launches one proces/core
            reload=                 100,        # reloads random subprocess every N tasks
            user_tasks=             False,      # userTasks mode
            tgen: TaskGenerator=    None,       # generates tasks for QMP
            sorted_res=             False,      # returns results sorted by taskID
            tq_maxsize=             None,       # tQue max size (for userTask mode)
            rq_trgsize=             500,        # rQue target size (if not userTasks mode), won't grow above
            name=                   'qmp',
            verb=                   0):

        self.verb = verb
        self.name = name
        self.proc_function = proc_func
        self.task_object = task_object
        self.n_proc = cpu_count() if n_proc == 0 else n_proc
        self.reload = reload
        self.user_tasks = user_tasks
        self.tgen = tgen if tgen else TaskGenerator()
        self.sorted_res = sorted_res

        self.state = self.QMPStates.init  # QMP init state
        if self.verb > 0: print('\n *** QMP *** %s (nProc %d) is @%s state...' % (self.name, self.n_proc, self.state))
        if self.verb > 1:
            if self.reload:    print(' > reloadEvery %d tasks' % self.reload)
            else:                   print(' > will not reload iProc')
            print(' > userTask %s' % self.user_tasks)
            print(' > using default taskGen %s' % (tgen is None))
            print(' > results will be sorted %s' % self.sorted_res)

        if tq_maxsize is None: tq_maxsize = 0
        self.tQue = Queue(maxsize=tq_maxsize)                              # tasks que
        self.rQue = Queue()                                                 # results que

        self.triggerQue = Queue()                                           # trigger stopping iProcManager
        self.reloadQue = Queue()
        if self.reload is not None: self.reloadQue.put('reloadActive')
        self.pQue = Queue()                                                 # killed process que

        self.taskIX = 0                                                     # task index
        self.resIX = 0                                                      # result index (expected)
        self.resBuff = {}                                                   # results buffer

        # create and start iProcesses
        self.kTasks = []

        self.iProcManager = Process(target=self.__iProcManagerFunc)
        self.iProcManager.start()

        #self.__startProc(nProc)
        self.state = self.QMPStates.running
        if self.verb > 0: print('%s is @%s state...' % (self.name, self.state))

        # put initial selfTasks
        if not self.user_tasks:
            if self.verb > 1: print(' > putting %d selfTasks into tQue' % rq_trgsize)
            for ix in range(rq_trgsize):
                self.tQue.put((self.taskIX, self.tgen.get_task()))
                self.taskIX += 1

    # iProc Manager function
    def __iProcManagerFunc(self):

        if self.verb > 1: print('%s iProcManager starts for %d iProcesses...' % (self.name, self.n_proc))

        # start iProcesses
        iProcs = []
        iProcsToJoin = []
        for _ in range(self.n_proc):
            iProc = Process(target=self.__ipRun)
            iProc.start()
            iProcs.append(iProc)

        # reload loop
        while not self.reloadQue.empty():

            # try get finished iProc
            try: pid = self.pQue.get(timeout=1)
            except Exception:
                if self.verb > 1: print('--- (re)')
                pass
            else:
                if self.verb > 1: print('+++ (re)')
                ix = 0
                while iProcs[ix].pid != pid: ix += 1
                iProcsToJoin.append(iProcs.pop(ix))
                if self.verb > 1: print(' >> iProc %d moved to join list' % pid)

                iProc = Process(target=self.__ipRun)
                iProc.start()
                iProcs.append(iProc)
                if self.verb > 1: print(' >> iProc %d started (reload)' % iProc.pid)

            # try join sth
            joined = []
            for ix in range(len(iProcsToJoin)):
                iProcsToJoin[ix].join(timeout=0.1)
                if iProcsToJoin[ix].exitcode == 0:
                    joined.append(iProcsToJoin[ix])
            for iProc in joined:
                iProcsToJoin.remove(iProc)
                if self.verb > 1: print(' >> iProc %d joined' % iProc.pid)

        # triggered stop
        self.triggerQue.get()
        if self.verb > 0: print('%s iProcManager stops...' % self.name)

        for _ in range(self.n_proc): self.tQue.put((-1, self.poison))
        if self.verb > 0: print(' > all poison send to tQue')

        for _ in range(self.n_proc): self.pQue.get()
        if self.verb > 0: print(' > all iProc finished (%s)' % self.name)

        n = 0
        while not self.rQue.empty():
            self.rQue.get()
            n += 1
        if self.verb > 0: print(' > rQue flushed %d results (%s)' % (n, self.name))

        iProcs += iProcsToJoin
        if self.verb > 0: print(' > got %d iProc to join (%s)' % (len(iProcs), self.name))
        while len(iProcs) > 0:
            iProc = iProcs.pop()
            iProc.join()
        if self.verb > 0: print(' > all iProc joined (%s)' % self.name)

    # iProc target method (loop)
    def __ipRun(self):

        while True:

            tix, task = self.tQue.get()

            if task == self.poison:
                self.pQue.put(os.getpid())
                break

            if self.task_object is not None: task = self.task_object

            result = self.proc_function(task)
            self.rQue.put((tix, result))

    # user puts his task for iProcesses
    def putTask(
            self,
            task):

        assert self.user_tasks, 'Err: User cannot put task in non userTasks mode'
        self.tQue.put((self.taskIX, task))
        self.taskIX += 1

        if self.reload is not None:
            if self.taskIX % self.reload == 0:
                self.tQue.put((-1, self.poison))

    # gets one result from que
    def getResult(self):

        if self.state is not self.QMPStates.running: return None

        if self.sorted_res:

            # take result from buffer
            if self.resIX in self.resBuff:
                result = self.resBuff.pop(self.resIX)

            # look 4 result from que
            else:

                while True:
                    resIX, result = self.rQue.get()
                    if resIX == self.resIX: break

                    self.resBuff[resIX] = result

            self.resIX += 1

        else: resIX, result = self.rQue.get()

        if not self.user_tasks and self.state is self.QMPStates.running:

            self.tQue.put((self.taskIX, self.tgen.get_task()))
            self.taskIX += 1

            if self.reload is not None:
                if self.taskIX % self.reload == 0:
                    self.tQue.put((-1, self.poison))

        return result

    # closes QMP
    def close(self):

        if self.state is not self.QMPStates.closed:

            self.state = self.QMPStates.killing
            if self.verb > 0: print('%s closes, now @%s state...' % (self.name, self.state))

            if not self.reloadQue.empty(): self.reloadQue.get() # to exit reload loop
            self.triggerQue.put('stopTrigger')                  # to enter triggered stop

            self.iProcManager.join()

            self.state = self.QMPStates.closed
            if self.verb > 0: print('%s closed successfully!' % self.name)

        else: print('%s already closed!' % self.name)

# multiprocessing accessible data
# np.array wont explode RAM usage by processes in contrast to list or dict
# but is very sensitive to sentence length (long sentences will explode mem usage @np.array)
class MPData:

    def __init__(
            self,
            fileFP=     None,   # full path to file with text lines
            data=       None,   # data in form of iterable (list, dict, etc)
            name=       'MPData',
            verbLev=    0):

        self.verbLev =verbLev
        self.name = name

        if self.verbLev > 0: print('\n%s loading file data...' % self.name)
        self.len, self.data = 0, None

        newData = data
        if fileFP:
            with open(fileFP, 'r') as file:
                newData = [line[:-1] for line in file]

        self.len = len(newData)
        self.data = np.asarray(newData)

        print(self.data.shape)
        print(self.data.dtype)

        if self.verbLev > 0: print('\n%s got %d (%s) lines of data' % (self.name, self.len, short_scin(self.len)))

    # returns line indexed with num from data
    def getData(
            self,
            num=    None):      # None returns random

        if num is None: num = np.random.randint(self.len)
        if num < self.len: return self.data[num]
        return None

"""
from putils.textool.textMethods import levDist
def test_Just2MPD(fa, fb):

    mpDataA = MPData(
        fileFP= fa,
        name=   'MPtest_A',
        verbLev=1)

    mpDataB = MPData(
        fileFP= fb,
        name=   'MPtest_B',
        verbLev=1)

    for ix in range(mpDataA.len):

        la = mpDataA.getData(ix)
        lb = mpDataB.getData(ix)
        print(levDist(la, lb))


def test_MPDwithQMP(fa, fb):

    mpDataA = MPData(
        fileFP= fa,
        name=   'MPtest_A',
        verbLev=1)

    mpDataB = MPData(
        fileFP= fb,
        name=   'MPtest_B',
        verbLev=1)

    def levDF(task):

        la = mpDataA.getData(task)
        lb = mpDataB.getData(task)

        levD = levDist(la, lb)
        return levD

    qMPl = QueMultiProcessor(
        iProcFunction=  levDF,
        userTasks=      True,
        name=           'QMP_levD',
        verbLev=        2)

    for ix in range(mpDataA.len):
        print(' >>> put task', ix)
        qMPl.putTask(ix)

    for ix in range(mpDataA.len):
        print(' <<< got result %d' % ix)
        print(qMPl.getResult())

    qMPl.close()
"""

if __name__ == '__main__':

    fna = '../../_Gdata/charMLM/wortschatz/wortschatzCorp_VLD_noised.txt'
    fnb = '../../_Gdata/charMLM/wortschatz/wortschatzCorp_VLD_source.txt'

    #test_Just2MPD(fna, fnb)

    #test_MPDwithQMP(fna, fnb)

    #fn0 = '../../_Gdata/charMLM/wortschatzMIX/wortschatzCorp.txt'
    #fn0 = '../../_Gdata/charMLM/wortschatz/wortschatzCorp.txt'
    #fn0 = '../../_Gdata/charMLM/wortschatz/wortschatzCorp_TRN_source.txt'
    #fn0 = '../../_Gdata/charMLM/wortschatz/wortschatzCorp_VLD_source.txt'

    import time

    def procF(task=None):

        pid = os.getpid()
        sleep = 0.1 + random.random()
        time.sleep(sleep)
        return 'task %d (%.1f) done by %d' % (task, sleep, pid)

    queMP = QueMultiProcessor(
        proc_func=      procF,
        n_proc=         10,
        reload=         100,
        sorted_res=     True,
        rq_trgsize=     100,
        verb=           1)

    time.sleep(5)

    startTime = time.time()
    for _ in range(500):
        #time.sleep(0.05)
        print(queMP.getResult())

    print(' ### time taken %.1f sec' % (time.time() - startTime))

    queMP.close()
