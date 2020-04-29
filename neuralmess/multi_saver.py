"""

 2018 (c) piteren

 MultiSaver for NN models
 - saves and loads dictionaries of variables lists (with subfolders)
 - supports list of savers (names for different savers for same variable list)

"""

import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


class MultiSaver:

    def __init__(
            self,
            modelName,                      # root folder of saves, every model will be put in subfolder of model name
            variables: list or dict,        # variables may be given as list (ALL) or as dict of {key: list} where key is name for the list
            savePath,                       # remember to create folder before
            savers: tuple=      (None,),    # list of names of savers
            maxKeep: list=      None,       # None keeps one
            session=            None,
            verbLev=            0):

        assert os.path.isdir(savePath), 'ERR: save path of model does not exists, please create first'

        # create subfolder if needed
        savePath += '/' + modelName
        if not os.path.isdir(savePath): os.mkdir(savePath)

        self.verb = verbLev
        self.modelName = modelName
        self.variables = variables
        self.savePath = savePath
        self.session = session

        if type(self.variables) is list: self.variables = {'ALL': self.variables}
        if not maxKeep: maxKeep = [1 for _ in savers]

        self.savers = {}
        for ix in range(len(savers)):
            self.savers[savers[ix]] = {}
            for var in self.variables:
                self.savers[savers[ix]][var] = tf.train.Saver(
                    var_list=               self.variables[var],
                    pad_step_number=        True,
                    save_relative_paths=    True,
                    max_to_keep=            maxKeep[ix])

        self.sStep = {sv: {var: 0 for var in self.variables} for sv in savers} # self step per saver per vars
        if self.verb > 0:
            print('\n*** MultiSaver *** for %s model' % self.modelName)
            print(' > got %d lists of variables' % len(self.variables))
            for var in self.variables: print(' >> list %s got %d variables'%(var,len(self.variables[var])))
            print(' > for every list of var got %d savers: %s' % (len(savers), savers))
            print(' > savers will save to %s' % self.savePath)

    # saves checkpoint of given saver
    def save(
            self,
            saver=      None,   # saver name
            step :int=  None,   # for None uses self step
            session=    None):

        assert saver in self.savers, 'ERR: unknown saver'

        svName = ' ' + saver if saver else ''
        if self.verb > 0: print('MultiSaver%s saves variables...' % svName)

        for var in self.variables:
            ckptPath = self.savePath + '/' + var + '/' + self.modelName
            if saver: ckptPath += '_' + saver
            ckptPath += '.ckpt'

            if not session: session = self.session

            latest_filename = 'checkpoint'
            if saver: latest_filename += '_' + saver

            self.savers[saver][var].save(
                sess=               session,
                save_path=          ckptPath,
                global_step=        step if step else self.sStep[saver][var],
                latest_filename=    latest_filename,
                write_meta_graph=   False,
                write_state=        True)
            self.sStep[saver][var] += 1
            if self.verb > 1: print(' > saved variables %s' % var)

    # loads last checkpoint of given saver
    def load(
            self,
            saver=                  None,
            session :tf.Session=    None,
            allow_init=             True):

        if not session: session = self.session
        if self.verb > 0: print()

        for var in self.variables:
            # look for checkpoint
            latest_filename = 'checkpoint'
            if saver: latest_filename += '_' + saver
            ckpt = tf.train.latest_checkpoint(
                checkpoint_dir=     self.savePath + '/' + var,
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
                session.run(tf.initializers.variables(self.variables[var]))
                if self.verb > 0: print('No checkpoint found, variables %s initialized with default initializer' % var)