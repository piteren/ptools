"""

 2020 (c) piteren

 GXRng is a dict type of GX keys where each key keeps range for its GX

 range may be defined as:
    - list of two ints
    - list of two floats
    - tuple of any values

"""

from copy import deepcopy
import random

from ptools.pms.foldered_dna import FMDna


# GX Ranges
class GXRng(dict):

    def __init__(
            self,
            **kwargs):

        super().__init__()

        for k in kwargs: assert type(kwargs[k]) in [list, tuple]
        for k in kwargs:
            if type(kwargs[k]) is list:
                ls = kwargs[k]
                assert len(ls)==2
                av, bv = (ls[0],ls[1]) if ls[0]<ls[1] else (ls[1],ls[0])
                assert av in [int, float]
                assert bv in [int, float]
                if type(av) is float or type(bv) is float:
                    av = float(av)
                    bv = float(bv)
                self[k] = [av,bv]
            else: self[k] = deepcopy(kwargs[k])

    # moves ints and floats into defined range
    def __move_val_into_range(
            self,
            key,
            val):
        if type(self[key][0]) is int: val = int(val)
        if val < self[key][0]: val = self[key][0]
        if val > self[key][1]: val = self[key][1]
        return val

    # returns child value
    def get_child_val(
            self,
            key,
            pa_val,
            pb_val,
            mix_prob=   0.5,    # mix: for floats - avg, for int - takes int(avg), for tuples - random; no mix takes one parent value
            noise_prob= 0.05,   # adds noise to float or int value
            noise_rng=  0.05):  # noise range

        # mix or random_select
        if random.random() < mix_prob:
            if type(self[key]) is list:
                val = pa_val + pb_val
                val /= 2
            else: val = random.choice(self[key])
        else: val = pa_val if random.random()<0.5 else pb_val

        # add noise
        if type(self[key]) is list and random.random()<noise_prob:
            noise_rng = noise_rng * (self[key][1] - self[key][0])
            noise_val = random.random() * noise_rng
            if random.random() < 0.5: noise_val *= -1
            val += noise_val

        if type(self[key]) is list: val = self.__move_val_into_range(key,val)

        return val

    # returns dict of randomly sampled  value for each key
    def sample(self):
        sm = {}
        for k in self:
            if type(self[k]) is tuple: val = random.choice(self[k])
            else:
                val = self[k][0] + random.random()*(self[k][1]-self[k][0])
                val = self.__move_val_into_range(k,val)
            sm[k] = val
        return sm


# genetic xrossing for Folder_Managed_DNA objects (saved already in folders)
def gx(
        name_A: str,    # name of parent A
        name_B: str,    # name of parent B
        name_child: str,# name of child

        top_FD: str,    # top folder
        fn_pfx: str,    # dna filename prefix

        gxr: GXRng,     # object with ranges of gx arguments
        mix_prob=       0.5,    # mix: for floats - avg, for int - takes int(avg), for tuples - random; no mix takes one parent value
        noise_prob=     0.05,   # adds noise to float or int value
        noise_rng=      0.05):  # noise range

    pa_fdna = FMDna(
        top_FD= top_FD,
        name=   name_A,
        fn_pfx= fn_pfx)
    pa_dna = pa_fdna.get_updated_dna()
    pa_dna = {k: pa_dna[k] for k in gxr}

    pb_fdna = FMDna(
        top_FD= top_FD,
        name=   name_B,
        fn_pfx= fn_pfx)
    pb_dna = pb_fdna.get_updated_dna()
    pb_dna = {k: pb_dna[k] for k in gxr}

    # mix parents dna
    pc_dna = {key: gxr.get_child_val(
        key=        key,
        pa_val=     pa_dna[key],
        pb_val=     pb_dna[key],
        mix_prob=   mix_prob,
        noise_prob= noise_prob,  # adds noise to float or int value
        noise_rng=  noise_rng) for key in gxr}

    cfmd = FMDna(
        top_FD= top_FD,
        name=   name_child,
        fn_pfx= fn_pfx)
    cfmd.save_dna(pc_dna)