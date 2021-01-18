"""

 2018 (c) piteren

    some little methods for Python

"""

import inspect
import os
import pickle
import random
import shutil
import string
import time


# returns default args (with their values)
def get_defaults(function):
    arg_dict = {}
    if function:
        specs = inspect.getfullargspec(function)
        args = specs.args
        vals = specs.defaults
        if vals:
            vals = list(vals)
            args.reverse()
            vals.reverse()
            for ix in range(len(vals)):
                arg_dict[args[ix]] = vals[ix]
    return arg_dict

# short scientific notation of floats
def short_scin(
        fl,
        precise=    False):

    sh = '%.1E' % float(fl) if not precise else '%.3E' % float(fl)
    sh = sh.replace('+0','')
    sh = sh.replace('+','')
    sh = sh.replace('-0','-')
    sh = sh.replace('E','e')
    return sh

# returns object from given pickleFileName
# returns None if file not exists
def r_pickle(
        pickleFP,       # pickle full path
        oType=None):    # if given checks for compatibility with given type
    obj = None
    if os.path.isfile(pickleFP):
        obj = pickle.load(open(pickleFP, 'rb'))
        if oType: assert type(obj) is oType, 'ERROR: obj from file is not %s type !!!' % str(oType)
    return obj

# writes obj to pickle
def w_pickle(
        obj,
        pickleFP):      # pickle full path
    pickle.dump(obj, open(pickleFP, 'wb'))

# returns timestamp string
def stamp(
        year=       False,
        date=       True,
        letters=    3):
    random.seed(time.time())
    if year:        stp = time.strftime('%y%m%d.%H%M')
    else:           stp = time.strftime('%m%d.%H%M')
    if not date:    stp = ''
    if letters:
        if date:    stp += '.'
        if True:    stp += ''.join([random.choice(string.ascii_letters) for _ in range(letters)])
    return stp

# prints nested dict
def print_nested_dict(dc: dict, ind_scale=2):

    tpD = {
        dict:   'D',
        list:   'L',
        tuple:  'T'}

    def __prn_root(root: dict, ind, ind_scale=2):
        spacer = ' ' * ind * ind_scale
        for k in sorted(list(root.keys())):
            tp = tpD.get(type(root[k]),'O')
            ln = len(root[k]) if tp in tpD.values() else ''
            lst = f' : {str(root[k])}' if tp=='L' else ''
            print(f'{spacer}{k} [{tp}{ln}]{lst}')
            if type(root[k]) is dict: __prn_root(root[k],ind+1,ind_scale)

    __prn_root(dc,ind=0,ind_scale=ind_scale)


# prepares folder, creates or flushes
def prep_folder(
        folder_path :str,
        flush_non_empty=    False):

    if not os.path.isdir(folder_path): os.mkdir(folder_path)
    elif flush_non_empty:
        shutil.rmtree(folder_path)
        prep_folder(folder_path)



if __name__ == '__main__':
    print(stamp())

    dc = {
        'a0': {
            'a1': {
                'a2': ['el1','el2']
            }
        },
        'b0': ['el1','el2','el3']
    }
    print_nested_dict(dc)