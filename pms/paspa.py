"""

 2018 (c) piteren

    PaSpa - parameters space
        > samples points from that space (volume)
            - point in space is a dict {key:value} key - axis of space, value - point on axis
            - may be forced to sample points from ax_rrad (relative distance on axis, on both sides)
        > calculates euclidean distance between points in that space
            - every axis is normalized to 1, so max distance for n-dim space is sqrt(n*1)
            - non-numeric axes do not have width >> distance on that axis 0

        PaSpa is build from psd (dict):

            psd - {param_axis: list or tuple}
                > list of ints or floats defines continuous range
                > tuple may contain elements of any type (but non-numeric will affect axis_length >> 0)

                example:
                {   'a':    [0.0, 1],
                    'b':    (-1,-7,10,15.5,90,30),
                    'c':    ('tat','mam','kot'),
                    'd':    [0,10],
                    'e':    (-2.0,2,None),
                    'f':    [-3,3.0]    }
"""

import math
import time
import random


# parameters space
class PaSpa:

    def __init__(
            self,
            psd :dict,                  # params space dictionary (lists & tuples)
            seed :int or None=  12321,  # seed for random sampling
            verb=               0):

        self.psd = psd
        self.seed_counter = seed if seed is not None else time.time()
        if verb > 0:  print('\n*** PaSam *** inits with seed %d ...' % self.seed_counter)

        # some safety check and resolve Type and Width
        self.psd_T = {} # param space type
        self.psd_W = {} # param space width
        for a in self.psd.keys():
            param_def = self.psd[a]
            assert type(param_def) in [list,tuple] # only list or tuples are supported

            tp = 'list'
            if type(param_def) is tuple: tp = 'tuple'

            tpn = '_int' # default
            are_flt = False
            are_dif = False
            for el in param_def:
                if type(el) is float: are_flt = True
                if type(el) is not int and type(el) is not float: are_dif = True
            if are_flt: tpn = '_float'  # downgrade
            if are_dif: tpn = '_diff'   # downgrade

            assert not (tp=='list' and tpn=='_diff')

            # sort numeric
            if tpn!='_diff':
                param_def = sorted(list(param_def))
                self.psd[a] = param_def if tp=='list' else tuple(param_def) # update type of sorted

            self.psd_T[a] = tp+tpn # string like 'list_int'
            self.psd_W[a] = param_def[-1]-param_def[0] if tpn != '_diff' else None # axis width, None for non-numeric

    # checks if given value is on axis of space
    def __is_in_axis(
            self,
            axis,       # axis key
            value):     # value

        if axis not in self.psd_T:                                  return False # axis not in space
        if 'list' in self.psd_T[axis]:
            if type(value) is float and 'int' in self.psd_T[axis]:  return False # type mismatch
            if not self.psd[axis][0] <= value <= self.psd[axis][1]: return False # value not in range
        elif value not in self.psd[axis]:                           return False # value not in tuple
        return True

    # checks if point is in space
    def __is_in_space(self, p: dict):
        for a in p:
            if not self.__is_in_axis(a,p[a]): return False
        return True

    # get (single) parameter (random) value
    # ...algorithm is a bit complicated, but ensures same probability(0.5) for both sides of ref_val (!edges of space)
    def __get_pval(
            self,
            axis: str,
            ref_val=    None,   # if given >> new value will be sampled from +-rad around ref_val
            ax_rrad=    None):

        # increase counter
        random.seed(self.seed_counter)
        self.seed_counter += 1

        if 'list' in self.psd_T[axis]:
            # min_val, rng, add, fval are floats, ...so int_list will have to
            min_val =   self.psd[axis][0]   if not ref_val else ref_val - ax_rrad * self.psd_W[axis]    # left (min) value
            rng =       self.psd_W[axis]    if not ref_val else 2*ax_rrad*self.psd_W[axis]              # whole axis range or twice ax_rrad
            add =       random.random()*rng                                                             # random from range                                 # if 'float' in self.psd_T[axis] else random.randint(0, int(rng))
            fval =      min_val + add                                                                   # add to left

            if 'int' in self.psd_T[axis]:   val = int(round(fval))
            else:                           val = fval
            while not self.__is_in_axis(axis,val):
                fval = (fval+ref_val)/2 # get closer by half to ref_val
                if 'int' in self.psd_T[axis]:   val = int(round(fval))
                else:                           val = fval

        else:
            # numeric tuple
            if self.psd_W[axis] is not None and ref_val is not None:

                # select vals from tuple in range
                sub_vals = []
                for e in self.psd[axis]:
                    if ref_val - ax_rrad*self.psd_W[axis] <= e <= ref_val + ax_rrad*self.psd_W[axis]:
                        sub_vals.append(e)
                assert sub_vals
                val = random.choice(sub_vals)

            else: val = random.choice(self.psd[axis]) # non-numeric tuple or sample from whole axis >> always choice
        return val

    # euclidean distance (normalized by axis width) between two points in space
    def dist(
            self,
            pa: dict,           # point a
            pb: dict) -> float: # point b

        dst = [0 if self.psd_W[param] is None else ((pa[param] - pb[param]) / self.psd_W[param])**2 for param in pa]
        return math.sqrt(sum(dst))

    # samples point in space
    def sample_point(
            self,
            ref_point :dict=    None,           # if reference point is given point will be sampled in given radius(rad)
            ax_rrad=            None) -> dict:  # relative distance on axis (both sides), where to sample

        if ref_point is None: ref_point = {key: None for key in self.psd.keys()}
        return {key: self.__get_pval(key, ref_point[key], ax_rrad) for key in self.psd.keys()}

    # point -> nicely formatted string
    @staticmethod
    def point_2str(p :dict) -> str:
        s = '{'
        for key in sorted(list(p.keys())):
            s += '%s:'%key
            if type(p[key]) is float:
                if p[key]<0.001: s += '%.7f ' % p[key]
                else: s += '%.3f ' % p[key]
            else: s += '%s ' % p[key]
        s = s[:-1] + '}'
        return s

    # returns info(string) about self
    def __str__(self):
        info = '*** PaSpa *** parameters space:\n'
        for param in sorted(list(self.psd.keys())):
            info += f' > {param}: {self.psd[param]}\n      {self.psd_T[param]}, width: {self.psd_W[param]}\n'
        return info[:-1]


def example_paspa():

    mdict = {
        'a':    0.5,
        'b':    7,
        'c':    'tata',
        'rgs':{
            'a':    [0.0, 1],
            'b':    (-1,-7,10,15.5,90,30),
            'c':    ('tat','mam','kot'),
            'd':    [0,10],
            'e':    (-2.0,2,None),
            'f':    [-3,3.0]}}

    spsa = PaSpa(mdict['rgs'])

    print()
    print(spsa)

    print()
    points = []
    for _ in range(10):
        point = spsa.sample_point()
        points.append(point)
        print(PaSpa.point_2str(point))

    print()
    for ix in range(10):
        ax_rrad = random.random()
        point_a = points[ix]
        point_b = spsa.sample_point(point_a, ax_rrad)
        print(f'distance {spsa.dist(point_a, point_b):.3f} (ax_rrad {ax_rrad}) between:')
        print(f' {PaSpa.point_2str(point_a)}')
        print(f' {PaSpa.point_2str(point_b)}')

    print()
    for _ in range(5):
        point_a = spsa.sample_point()
        point_b = spsa.sample_point()
        print(f'distance {spsa.dist(point_a, point_b):.3f} between:\n {PaSpa.point_2str(point_a)}\n {PaSpa.point_2str(point_b)}')


if __name__ == '__main__':

    example_paspa()

