"""

 2019 (c) piteren

"""

import GPUtil
import platform
import os
import tensorflow as tf
from typing import List
import warnings

from ptools.lipytools.logger import set_logger

# masks GPUs from given list of ids or single one
def mask_cuda(ids :list or int= None):
    if ids is None: ids = []
    if type(ids) is int: ids = [ids]
    mask = ''
    for id in ids: mask += '%d,'%id
    if len(mask) > 1: mask = mask[:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = mask

# returns cuda memory size (system first device)
def get_cuda_mem(): return GPUtil.getGPUs()[0].memoryTotal

# returns list of available GPUs ids
def get_available_cuda_id(max_mem=None): # None sets automatic, otherwise (0,1.1] (above 1 for all)
    if not max_mem:
        tot_mem = get_cuda_mem()
        if tot_mem < 5000:  max_mem=0.35 # small GPU case, probably system single GPU
        else:               max_mem=0.2
    return GPUtil.getAvailable(limit=20, maxMemory=max_mem)

# prints report of system cuda devices
def report_cuda():
    print('\nSystem CUDA devices:')
    for device in GPUtil.getGPUs():
        print(' > id: %d name: %s MEM(U/T): %d/%d' % (device.id, device.name, device.memoryUsed, device.memoryTotal))

# resolves given devices, returns list of str in TF naming convention
def tf_devices(
        devices :list or int or None=   -1,
        verb=                           1) -> List[str]:

    """
    devices may be given as:
        1.  int         one (system) cuda id
        2.  -1          last available cuda id
        3.  None        CPU device
        4.  []          all available cudas
        5.  [int]       list of cuda ids, may contain None and repetitions
        6.  str         device in TF format (e.g. '/device:GPU:0')
        7.  [str]       list of devices in TF format

    for 1-5 masks cuda devices
    for OSX returns CPU only
    """

    if platform.system() == 'Darwin':
        print('device_TF: OSX does not support GPUs >> using only CPU')
        num = len(devices) if type(devices) is list and devices else 1
        devices = ['/device:CPU:0']*num
    # TODO: handle: no-GPU system >> devices=None or ^ ['/device:CPU:0']*num
    else:
        # got nicely TF formated devices (and probably earlier masked) >> no action needed (convert to list only...)
        if type(devices) is str or (type(devices) is list and devices and type(devices[0]) is str):
            if type(devices) is not list: devices = [devices]
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            if verb>0: report_cuda()

            # resolve given devices
            av_dev = get_available_cuda_id()
            if type(devices) is list and not devices:  devices = av_dev        # take all available GPUs (empty list)
            if devices==-1:                            devices = [av_dev[-1]]  # take last available GPU
            if type(devices) is not list:              devices = [devices]     # change to list

            if verb>0: print('Going to use devices (id): %s' % devices)
            # mask them
            new_ix = 0
            dev_id_tomask = []
            dev_id_aftermask = []
            if verb>0: print('Masking GPUs to new TF ids:')
            for dev in devices:
                if dev is not None:
                    dev_id_tomask.append(dev)
                    dev_id_aftermask.append(new_ix)
                    new_ix += 1
                else:
                    dev_id_aftermask.append(None)
                if verb>0: print(' > GPU id: %s (sys) >> %s (TF)' % (dev, dev_id_aftermask[-1]))
            mask_cuda(dev_id_tomask)

            devices = []
            if not dev_id_aftermask: dev_id_aftermask = [None]  # at least one CPU
            for dev in dev_id_aftermask:
                if dev is not None: devices.append('/device:GPU:%d'%dev)
                else:               devices.append('/device:CPU:0')
    if verb > 0: print(' >> returning %d devices: %s'%(len(devices),devices))
    return devices

# init function for every TF.py script:
# - sets low verbosity of TF
# - starts logger
# - manages TF devices
def nestarter(
        log_folder: str or None=    '_log', # for None doesn't logs
        custom_name: str or None=   None,   # set custom logger name, for None uses default
        devices=                    -1,     # False for not managing TF devices
        verb=                       1):

    # tf verbosity
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    #tf.logging.set_verbosity(tf.logging.ERROR)
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if log_folder: set_logger(logFD=log_folder, custom_name=custom_name, verb=verb) # set logger
    if devices is not False: return tf_devices(devices=devices, verb=verb)


if __name__ == '__main__':

    print(tf_devices([]))