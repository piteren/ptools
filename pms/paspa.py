"""

 2018 (c) piteren

    PaSpa - parameters space
        - each parameter (key) defines an axis of space
        - each axis has:
            - type (list - continuous, tuple - discrete)
            - range (width)
        - non numeric parameters are handled (PaSpa handles distance for those)

        PaSpa is a metric space (https://en.wikipedia.org/wiki/Metric_space)
        - PaSpa uses L1N distance (Manhattan Distance with axis width normalized to 1, finally normalized by dimm (num of axes))


        PaSpa defines a point in space (self)
        - point in the space is a dict {key:value} key - axis of space, value - point on axis
        PaSpa samples (random) points from space and may be forced to sample points from within a given distance

        PaSpa is build from psd (dict):
            psd - {axis(parameter name): list or tuple}
                > list of ints or floats defines continuous range
                > tuple may contain elements of any type (even non-numeric)

                example:
                {   'a':    [0.0, 1],
                    'b':    (-1,-7,10,15.5,90,30),
                    'c':    ('tat','mam','kot'),
                    'd':    [0,10],
                    'e':    (-2.0,2,None),
                    'f':    [-3,3.0]    }
"""

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
        if verb > 0:  print(f'\n*** PaSpa *** ({len(self.psd)} dim) inits with seed {self.seed_counter} ...')

        # some safety check and resolve Type and Width
        self.psd_T = {} # axis space type
        self.psd_W = {} # axis space width
        for axis in self.psd:
            param_def = self.psd[axis]
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
                self.psd[axis] = param_def if tp=='list' else tuple(param_def) # update type of sorted

            self.psd_T[axis] = tp+tpn # string like 'list_int'
            self.psd_W[axis] = param_def[-1]-param_def[0] if tpn != '_diff' else len(param_def)-1

    # checks if given value belongs to an axis of space
    def __is_in_axis(
            self,
            axis,       # axis key
            value):     # value

        if axis not in self.psd_T:                                  return False # axis not in a space
        if 'list' in self.psd_T[axis]:
            if type(value) is float and 'int' in self.psd_T[axis]:  return False # type mismatch
            if not self.psd[axis][0] <= value <= self.psd[axis][1]: return False # value not in a range
        elif value not in self.psd[axis]:                           return False # value not in a tuple
        return True

    # checks if point belongs to a space
    def __is_in_space(self, p: dict):
        for a in p:
            if not self.__is_in_axis(a,p[a]): return False
        return True

    # get (single) parameter (random) value
    # ...algorithm is a bit complicated, but ensures same probability(0.5) for both sides of ref_val (!edges of space)
    def __get_pval(
            self,
            axis: str,
            ref_val=    None,   # if given >> new value will be sampled from +-ax_dst around ref_val
            ax_dst=     None):

        # increase counter
        random.seed(self.seed_counter)
        self.seed_counter += 1

        if 'list' in self.psd_T[axis]:
            # min_val, rng, add, fval are floats, ...so int_list will have to
            min_val =   self.psd[axis][0]   if not ref_val else ref_val - ax_dst * self.psd_W[axis] # left (min) value
            rng =       self.psd_W[axis]    if not ref_val else 2*ax_dst*self.psd_W[axis]           # whole axis range or twice ax_rrad
            add =       random.random()*rng                                                         # random from range                                 # if 'float' in self.psd_T[axis] else random.randint(0, int(rng))
            fval =      min_val + add                                                               # add to left

            if 'int' in self.psd_T[axis]:   val = int(round(fval))                                  # round to int
            else:                           val = fval                                              # just float
            # only when ref_val:
            while not self.__is_in_axis(axis,val):
                fval = (fval+ref_val)/2 # get closer by half to ref_val
                if 'int' in self.psd_T[axis]:   val = int(round(fval))                                  # round to int
                else:                           val = fval                                              # just float

        else:
            if ref_val is not None:

                # select vals from tuple in range
                sub_vals_L = [] # left
                sub_vals_R = [] # right
                for e in self.psd[axis]:
                    un_ax_dst = ax_dst*self.psd_W[axis]
                    if 'diff' not in self.psd_T[axis]:
                        if ref_val - un_ax_dst <= e <= ref_val + un_ax_dst: # in range
                            if e < ref_val: sub_vals_L.append(e)
                            else:           sub_vals_R.append(e)
                    else:
                        psdL = list(self.psd[axis])
                        eIX = psdL.index(e)
                        rIX = psdL.index(ref_val)
                        if rIX - un_ax_dst <= eIX <= rIX + un_ax_dst:
                            if eIX < rIX: sub_vals_L.append(e)
                            else:         sub_vals_R.append(e)

                # same quantity for both sides (...probability(0.5))
                sh_vals = sub_vals_L
                lg_vals = sub_vals_R
                # swap
                if len(sub_vals_L) > len(sub_vals_R):
                    sh_vals = sub_vals_R
                    lg_vals = sub_vals_L
                if sh_vals:
                    while len(sh_vals) < len(lg_vals):
                        sh_vals.append(random.choice(sh_vals))
                sub_vals = sh_vals + lg_vals
                assert sub_vals

                val = random.choice(sub_vals)

            else: val = random.choice(self.psd[axis]) # sample from whole axis
        return val

    # L1N distance between two points in space
    def dist(
            self,
            pa: dict,           # point a
            pb: dict) -> float: # point b

        dst = [] # list of distances on axes
        for axis in pa:
            if 'diff' not in self.psd_T[axis]:
                d = pa[axis] - pb[axis]
            else:
                psdL = list(self.psd[axis])
                #print(axis, psdL)
                #print(pa, pb)
                #print(pa[axis], pb[axis])
                d = psdL.index(pa[axis]) - psdL.index(pb[axis])
            dst.append( abs(d) / self.psd_W[axis] )
        return sum(dst) / len(dst)

    # samples point in space
    def sample_point(
            self,
            ref_point :dict=    None,           # if reference point is given point will be sampled in given radius(rad)
            ax_dst=             None) -> dict:  # relative distance on axis (both sides), where to sample

        if ref_point is None: ref_point = {key: None for key in self.psd.keys()}
        return {key: self.__get_pval(key, ref_point[key], ax_dst) for key in self.psd.keys()}

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
        info = f'*** PaSpa *** {len(self.psd)} parameters space:\n'
        max_ax_l = 0
        max_ps_l = 0
        for axis in sorted(list(self.psd.keys())):
            if len(axis)                > max_ax_l: max_ax_l = len(axis)
            if len(str(self.psd[axis])) > max_ps_l: max_ps_l = len(str(self.psd[axis]))
        if max_ax_l > 40: max_ax_l = 40
        if max_ps_l > 40: max_ps_l = 40

        for axis in sorted(list(self.psd.keys())):
            info += f' > {axis:{max_ax_l}s}  {str(self.psd[axis]):{max_ps_l}s}  {self.psd_T[axis]:11s}  width: {self.psd_W[axis]}\n'
        return info[:-1]


def example_paspa(dc):

    spsa = PaSpa(dc)

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
        ax_dst = random.random()
        point_a = points[ix]
        point_b = spsa.sample_point(point_a, ax_dst)
        print(f'ax_dst {ax_dst:.3f} >> distance {spsa.dist(point_a, point_b):.3f}')
        print(f' {PaSpa.point_2str(point_a)}')
        print(f' {PaSpa.point_2str(point_b)}')

    print()
    for _ in range(10):
        point_a = spsa.sample_point()
        point_b = spsa.sample_point()
        print(f'distance {spsa.dist(point_a, point_b):.3f} between:\n {PaSpa.point_2str(point_a)}\n {PaSpa.point_2str(point_b)}')


if __name__ == '__main__':

    rgs = {
        'a':    (True, False, None),
        'b':    (-1, -7, 10, 15.5, 90, 30),
        'c':    ('tat', 'mam', 'kot'),
        'd':    (0, 1, 2, 3, None),
        'e':    (-2.0, 2, None),
        'f':    [-3, 3.0],
        'g':    (-3, 3),
        'h':    (-3, None, 3)
    }

    example_paspa(rgs)
