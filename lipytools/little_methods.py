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

# print nested dict
def print_nested_dict(dc, ind=0):

    # print obj with n indentation
    def prnIND(
            nInd,       # number of indentations
            obj,        # object to print
            indL=3):    # length of indentation

        tab = ''
        tabInc = ''
        for _ in range(indL): tabInc += ' '
        for _ in range(nInd): tab += tabInc
        print('%s%s' % (tab, obj))

    if type(dc) is dict:
        for key in sorted(list(dc.keys())):
            prnIND(ind, '%s'%key)
            print_nested_dict(dc[key], ind + 1)
    else:
        if type(dc) is list:    prnIND(ind, '%s...(len %d)'%(dc[0], len(dc)))
        else:                   prnIND(ind, '%s'%dc)

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