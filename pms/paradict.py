"""

 2020 (c) piteren

    ParaDict - Parameters Dictionary with CONTROLLED update (of keys and values)

"""

import copy
import itertools

from ptools.textool.text_metrics import lev_dist


# params dictionary
class ParaDict(dict):

    def __init__(
            self,
            dct :dict,
            verb=               0):

        super(ParaDict, self).__init__()
        self.verb = verb

        self.update(dct)

        if verb > 0: print('\n*** ParaDict *** inits...')

    # updates values of self keys with subdct but all keys from subdct MUST BE PRESENT in self
    def refresh(
            self,
            subdct :dict):

        for key in subdct:
            assert key in self, 'ERR: key \'%s\' from refresher not present in params dict'%key
        self.update(subdct)

    # updates values of self keys with subdct but only for those keys ALREADY PRESENT in self
    def update_values(
            self,
            subdct :dict):

        for key in subdct:
            if key in self: self[key] = subdct[key]

    # extends self keys with new subdct keys
    def add(
            self,
            subdct :dict,
            check_pmss=   True):

        if check_pmss: self.check_params_sim(subdct)
        for key in subdct:
            if key not in self:
                self[key] = subdct[key]

    # checks for params similarity
    def check_params_sim(
            self,
            params :dict or list,
            lev_dist_diff: int=     1):

        found_any = False

        # look for params not in self.keys
        paramsL = params if type(params) is list else list(params.keys())
        self_paramsL = list(self.keys())
        diff_paramsL = [par for par in paramsL if par not in self_paramsL]

        # prepare dictionary of lists of lowercased underscore splits of params not in self.keys
        diff_paramsD = {}
        for par in diff_paramsL:
            split = par.split('_')
            split_lower = [el.lower() for el in split]
            perm = list(itertools.permutations(split_lower))
            diff_paramsD[par] = [''.join(el) for el in perm]

        self_paramsD = {par: par.replace('_','').lower() for par in self_paramsL} # self params lowercased with removed underscores

        for p_key in diff_paramsD:
            for s_key in self_paramsD:
                sim_keys = False
                s_par = self_paramsD[s_key]
                for p_par in diff_paramsD[p_key]:
                    levD = lev_dist(p_par,s_par)
                    if levD <= lev_dist_diff: sim_keys = True
                    if s_par in p_par or p_par in s_par: sim_keys = True
                if sim_keys:
                    if not found_any:
                        print('\nParaDict was asked to check for params similarity and found:')
                        found_any = True
                    print(f' @@@ ### >>> ACHTUNG: keys \'{p_key}\' and \'{s_key}\' are CLOSE !!!')

    # returns deepcopy of self dict
    def get_dict_copy(self): return copy.deepcopy(self)

    # converts dict (...of params)
    @staticmethod
    def dict_2str(d: dict) -> str:
        info = ''
        for k, v in sorted(d.items()):
            sk = '%s'%k
            sv = '%s'%v
            if len(sk)>25: sk = sk[:25] + ' ...'
            if len(sv)>35: sv = sv[:35] + ' ...'
            info += '%30s : %s\n'%(sk,sv)
        return info[:-1]

    def __str__(self):
        return self.dict_2str(self)

    # writes self(dict) to txt file
    def write_txtfile(self, file_name):
        file = open(file_name, 'w')
        file.write(str(self))
        file.close()
