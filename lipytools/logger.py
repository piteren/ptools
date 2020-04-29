"""

 2018 (c) piteren

"""

import os
import sys
import time

# logger duplicating print() output to given file
class Logger:

    def __init__(
            self,
            fileName):
        self.terminal = sys.stdout
        self.log = open(fileName, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# method setting logger to folder
def set_logger(
        logFD,
        custom_name= None,
        verb=       1):

    if not os.path.isdir(logFD): os.mkdir(logFD)

    cDate = time.strftime("%Y%m%d.%H%M%S")
    fileName = custom_name if custom_name else 'run'
    fileName += '_' + cDate + '.log'

    sys.stdout = Logger(logFD + '/' +  fileName)

    path = os.path.join(os.path.dirname(sys.argv[0]), os.path.basename(sys.argv[0]))
    if verb>0: print('\nLogger started %s for %s' %(cDate, path))