"""

 2018 (c) piteren

 MultiSaver for NN models
 - saves and loads dictionaries of variables lists (with subfolders)
 - supports list of savers (names for different savers for same variable list)

"""

import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from ptools.neuralmess.get_tf import tf


class MultiSaver:

    def __init__(
            self,
            model_name :str,
            vars: list or dict,             # variables may be given as list (ALL) or as dict of {key: list} where key is name for the list
            root_FD :str,                   # root (save) folder
            savers: tuple=      (None,),    # list of names of savers
            max_keep: list=      None,       # None keeps one
            session=            None,
            verb=               0):

        if not os.path.isdir(root_FD): os.mkdir(root_FD)
        self.save_FD = f'{root_FD}/{model_name}'
        if not os.path.isdir(self.save_FD): os.mkdir(self.save_FD)

        self.verb = verb
        self.model_name = model_name
        self.vars = vars
        self.session = session

        if type(self.vars) is list: self.vars = {'ALL': self.vars}
        if not max_keep: max_keep = [1 for _ in savers]

        self.savers = {}
        for ix in range(len(savers)):
            self.savers[savers[ix]] = {}
            for var in self.vars:
                self.savers[savers[ix]][var] = tf.train.Saver(
                    var_list=               self.vars[var],
                    pad_step_number=        True,
                    save_relative_paths=    True,
                    max_to_keep=            max_keep[ix])

        self.s_step = {sv: {var: 0 for var in self.vars} for sv in savers} # self step per saver per vars
        if self.verb > 0:
            print('\n*** MultiSaver *** for %s model' % self.model_name)
            print(' > got %d lists of variables' % len(self.vars))
            for var in self.vars: print(' >> list %s got %d variables' % (var, len(self.vars[var])))
            print(' > for every list of var got %d savers: %s' % (len(savers), savers))
            print(' > savers will save to %s' % self.save_FD)

    # saves checkpoint of given saver
    def save(
            self,
            saver=      None,   # saver name
            step :int=  None,   # for None uses self step
            session=    None):

        assert saver in self.savers, 'ERR: unknown saver'

        sv_name = ' ' + saver if saver else ''
        if self.verb > 0: print('MultiSaver%s saves variables...' % sv_name)

        for var in self.vars:
            ckpt_path = self.save_FD + '/' + var + '/' + self.model_name
            if saver: ckpt_path += '_' + saver
            ckpt_path += '.ckpt'

            if not session: session = self.session

            latest_filename = 'checkpoint'
            if saver: latest_filename += '_' + saver

            self.savers[saver][var].save(
                sess=               session,
                save_path=          ckpt_path,
                global_step=        step if step else self.s_step[saver][var],
                latest_filename=    latest_filename,
                write_meta_graph=   False,
                write_state=        True)
            self.s_step[saver][var] += 1
            if self.verb > 1: print(' > saved variables %s' % var)

    # loads last checkpoint of given saver
    def load(
            self,
            saver=                          None,
            session :tf.compat.v1.Session=  None,
            allow_init=                     True):

        if not session: session = self.session
        if self.verb > 0: print()

        for var in self.vars:
            # look for checkpoint
            latest_filename = 'checkpoint'
            if saver: latest_filename += '_' + saver
            ckpt = tf.train.latest_checkpoint(
                checkpoint_dir=self.save_FD + '/' + var,
                latest_filename=    latest_filename)

            if ckpt:
                if self.verb > 1:
                    print('\n >>> tensors @ckpt %s' % ckpt)
                    print_tensors_in_checkpoint_file(
                        file_name=      ckpt,
                        tensor_name=    '',
                        all_tensors=    False)
                self.savers[saver][var].restore(session, ckpt)
                if self.verb > 0: print('Variables %s restored from checkpoint %s' % (var, saver if saver else ''))

            else:
                assert allow_init, 'Err: saver load failed: checkpoint not found and not allowInit'
                session.run(tf.initializers.variables(self.vars[var]))
                if self.verb > 0: print('No checkpoint found, variables %s initialized with default initializer' % var)