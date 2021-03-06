"""

 2018 (c) piteren

 DVC Starter
 - builds DVC model for sentence training/test

"""

import numpy as np
import shutil
from sklearn.metrics import f1_score
import tensorflow as tf
import time

from putils.neuralmess.dev_manager import tf_devices
from putils.neuralmess.dvc.presets import dvc_presets
from putils.pms.paradict import ParaDict
from putils.neuralmess.nemodel import NEModel
from putils.neuralmess.dvc.data import UDD, DVCData
from putils.neuralmess.dvc.model import dvc_model
from putils.neuralmess.dvc.batcher import DVCBatcher

# prepares report from all_results (train)
def report_all_results(ar, if_lables=None, lbl_dict=None):
    print()
    if_pred = None
    if ar['ts_acc'] is not None:
        n = ar['ts_acc'].shape[0]
        msi = '' if n==1 else '(m_seed %d)'%n
        print('Results%s: ts_acc %.3f (std %.3f), ts_loss %.3f'%(msi,float(np.mean(ar['ts_acc'])),float(np.std(ar['ts_acc'])),float(np.mean(ar['ts_loss']))))
    if ar['if_pred'] is not None:
        n = ar['if_pred'].shape[0]
        if_pred = np.sum(ar['if_pred'], axis=0)
        if_pred = np.argmax(if_pred, axis=-1)

        if if_lables and lbl_dict:
            tns_labels = [lbl_dict[lbl] for lbl in if_lables]
            tns_labels = np.asarray(tns_labels)
            equal = np.equal(if_pred, tns_labels, dtype=np.int)
            print('Inference accuracy (%2d) %.3f'%(n,float(np.mean(equal))))
            for ix in range(n):
                if_pred_s = np.argmax(ar['if_pred'][ix], axis=-1)
                equal = np.equal(if_pred_s, tns_labels, dtype=np.int)
                print('                  > %2d_ %.3f' % (ix, float(np.mean(equal))))

        # translate labels
        inv_lbl_dict = {lbl_dict[lbl]: lbl for lbl in lbl_dict}
        if_pred = [inv_lbl_dict[v] for v in if_pred]
    return if_pred


class DVCStarter:

    def __init__(
            self,
            dvc_dict :dict or ParaDict,         # model dict
            dvc_data :DVCData=      None,
            dvc_dd :dict=           None,       # dict to unpack for DVCData constructor
            seed :int=              12321,      # overrides seed of model, data, batcher
            custom_name: str=       None,       # set to override model name
            name_timestamp=         True,       # adds timestamp to model name
            devices=                -1,         # single device supported
            rand_batcher=           True,       # for None sets automatic, True is recommended
            save_TFD: str=          '.',        # save top folder (of starter)
            do_mlog=                True,       # model does own logging (saves txt to its folder)
            do_TX=                  True,       # prints txt reports while training
            do_TB=                  True,       # run TB
            ckpt_load: str=         'TM',       # checkpoint type to load while init
            init_DBM=               False,      # builds Data, Batcher, Model
            verb=                   1):

        self.verb = verb
        if self.verb > 0: print('\n*** DVC_starter *** initializes...')

        self.mdict = dvc_dict
        if type(self.mdict) is not ParaDict:
            self.mdict = ParaDict(dvc_presets['dvc_base'], verb=self.verb-1)
            self.mdict.refresh(dvc_dict)

        self.seed = seed

        self.device = tf_devices(devices, verb=self.verb)[0]

        self.rand_batcher = rand_batcher
        self.dvc_data = dvc_data if not dvc_dd else DVCData(**dvc_dd)
        self.save_TFD = save_TFD
        self.do_mlog = do_mlog
        self.do_TX = do_TX
        self.do_TB = do_TB
        self.ckpt_load = ckpt_load

        # model base name of starter, since m_seed training needs to be kept separately (timestamp will be added while building: @__build_DBM)
        self.name_base = self.mdict['name']
        if not self.name_base: self.name_base = 'dvc' # default name
        if custom_name: self.name_base = custom_name # custom name
        self.name_timestamp = name_timestamp

        self.model = None
        self.batcher = None
        self.batch_ix = 0
        if init_DBM:
            build_results = self.__build_DBM(seed=self.seed, verb=self.verb)
            self.model =    build_results['model']
            self.batcher =  build_results['batcher']
            self.batch_ix = build_results['batch_ix']

    # builds batcher & model with given seed, sets new distribution of data
    def __build_DBM(
            self,
            seed :int,
            seed_data=  True, # for false uses fixed data seed
            seed_model= True, # for false uses fixed model seed
            verb=       0):

        self.mdict['name'] = self.name_base

        # since self.seed does not change, conditions below may ensure constant seed
        self.mdict['seed'] = seed if seed_model else self.seed  # model seed
        if self.dvc_data: self.dvc_data.new_data_distribution(seed if seed_data else self.seed, force_silent=verb<=0) # data distribution seed

        if verb > 1: print('\nDVC_starter builds: data, batcher, model (seed %d)'%seed)

        do_opt = True if self.dvc_data and self.dvc_data.uDD_size['TR'] > 0 else False # turn off for no training data
        model = NEModel(
            fwd_func=       dvc_model,
            devices=        self.device,
            do_opt=         do_opt,
            mdict=          self.mdict,
            name_timestamp= self.name_timestamp,
            save_TFD=       self.save_TFD + '/_models',
            savers_names=   ('VL','TM'),
            load_saver=     self.ckpt_load,
            do_log=         self.do_mlog,
            verb=           verb-1)

        batcher = DVCBatcher(
            dvc_data=       self.dvc_data,
            seed=           seed if seed_data else self.seed,
            batch_size=     self.mdict['batch_size'],
            random_TR=      self.rand_batcher,
            verb=           verb) if self.dvc_data else None

        return {
            'model':    model,
            'batcher':  batcher,
            'batch_ix': 0}

    # trains model
    def train(
            self,
            n_batches=      2000,
            m_seed: int=    1,          # for 2 or more runs M multi_seed training
            ms_data=        True,       # multi_seed for data distribution and batcher
            ms_model=       True,       # multi_seed for model architecture (TF: init, drop)
            # validation
            fq_VL=          50,         # frequency of VL
            save_max_VL=    True,       # save model for max VL
            validation_TS=  True,       # perform TS after validation
            # reporting
            fq_AVG=         10,         # (base) frequency of averaging (acc,loss) >> txt reports
            fqM_TB=         1,          # frequency *M (fq_AVG*M) of TB reports (scalars)
            fqM_HTB=        0,          # frequency *M (fq_AVG*M) of HTB reports (histograms)
            zeros_it=       (50,500),   # intervals for zeros reporting
            # finally
            close=          True,       # close model after test
            delete_MFD=     True):      # delete model folders after training

        do_TX = self.do_TX
        rep_fq_TB = fq_AVG * fqM_TB
        rep_fq_HTB = fq_AVG * fqM_HTB
        if not self.do_TB: rep_fq_TB, rep_fq_HTB = 0, 0

        # set envy for m_seed
        if m_seed > 1:
            self.name_timestamp =   True
            self.mdict['verb'] =    0
            self.do_mlog =          False
            do_TX =                 False
            rep_fq_TB =             0
            rep_fq_HTB =            0

        if self.verb > 0: print('\nDVC_starter starts training, m_seed %d, %d batches'%(m_seed,n_batches))
        avg_acc = 0
        avg_loss = 0
        m_seed_results = []
        for msix in range(m_seed):

            loop_stime = time.time() # loop start time
            seed = self.seed + msix # counted seed

            if m_seed>1 or not self.model:
                build_results = self.__build_DBM(
                    seed=       seed,
                    seed_data=  ms_data,
                    seed_model= ms_model,
                    verb=       self.verb-1 if m_seed==1 else 0)
                self.model =    build_results['model']
                self.batcher =  build_results['batcher']
                self.batch_ix = build_results['batch_ix']

            if self.verb > 0: print('\nTraining loop (%d/%d), model %s (seed %d)' % (msix+1,m_seed,self.model['name'],seed))
            # one train loop results (lists (per batch): [[val,step],])
            t_res = {
                'train.a':      [],
                'train.l':      [],
                'valid.a':      [],
                'valid.l':      [],
                'test.a':       [],
                'test.l':       [],
                'max_vl_acc':   None,
                'ts_acc':       None,
                'ts_loss':      None,
                'ts_type':      None,
                'if_pred':      None}
            indZeros = [[] for _ in zeros_it]
            report_stime = time.time() # report start time
            for _ in range(n_batches):

                batch = self.batcher.get_batch()
                self.batch_ix += 1

                feed = {self.model['train_flag_PH']:  True}

                for cix in range(len(batch['lbl'])): # labels per classifier
                    feed[self.model['lab_PHL'][cix]] = batch['lbl'][cix]

                dTypes = ['vec','tks','seq']
                for tp in dTypes:
                    if self.model[tp+'_PHL'] is not None:
                        for nS in range(len(batch[tp])):
                            feed[self.model[tp+'_PHL'][nS]] = batch[tp][nS]

                fetches = [
                    self.model['accuracy'],
                    self.model['loss'],
                    self.model['gg_norm'],
                    self.model['avt_gg_norm'],
                    self.model['scaled_LR'],
                    self.model['optimizer'],
                    self.model['nn_zeros']]

                if rep_fq_HTB and self.batch_ix%rep_fq_HTB==0 and m_seed==1:
                    fetches.append(self.model['hist_summ'])

                runVal = self.model.session.run(fetches, feed)
                while len(runVal) < 8: runVal.append(None)

                acc, loss, gg_norm, avt_gg_norm, lr, _, nn_zeros, hist_summ = runVal
                avg_acc += acc*100
                avg_loss += loss

                if not nn_zeros.size: nn_zeros = [0]
                if rep_fq_TB:
                    for ls in indZeros: ls.append(nn_zeros)

                if fq_AVG and self.batch_ix%fq_AVG==0:
                    avg_acc /= fq_AVG
                    avg_loss /= fq_AVG
                    t_res['train.a'].append([avg_acc, self.batch_ix])
                    t_res['train.l'].append([avg_loss, self.batch_ix])

                    if do_TX:
                        print('%7d aAcc: %.1f, aLoss: %.3f ' % (self.batch_ix, avg_acc, avg_loss), end='')
                        if self.batch_ix%(fq_AVG * 10)==0:
                            print('(%d s/sec)' % ((self.batcher.target_batch_size['TR'] * fq_AVG * 10) / (time.time() - report_stime)))
                            report_stime = time.time()
                        else: print()

                    if rep_fq_TB and self.batch_ix%rep_fq_TB==0:
                        acc_summ = tf.Summary(value=[tf.Summary.Value(tag='a.train/1.acc', simple_value=avg_acc)])
                        lss_summ = tf.Summary(value=[tf.Summary.Value(tag='a.train/2.loss', simple_value=avg_loss)])
                        lr_summ = tf.Summary(value=[tf.Summary.Value(tag='a.train/3.lR', simple_value=lr)])
                        ggn_summ = tf.Summary(value=[tf.Summary.Value(tag='d.grad/1.gg_norm', simple_value=gg_norm)])
                        aggn_summ = tf.Summary(value=[tf.Summary.Value(tag='d.grad/2.avt_gg_norm', simple_value=avt_gg_norm)])
                        self.model.summ_writer.add_summary(acc_summ, self.batch_ix)
                        self.model.summ_writer.add_summary(lss_summ, self.batch_ix)
                        self.model.summ_writer.add_summary(lr_summ, self.batch_ix)
                        self.model.summ_writer.add_summary(ggn_summ, self.batch_ix)
                        self.model.summ_writer.add_summary(aggn_summ, self.batch_ix)

                        zeros_T0 = np.mean(nn_zeros)
                        zeros_T0_summ = tf.Summary(value=[tf.Summary.Value(tag='e.zeros/nT0', simple_value=zeros_T0)])
                        self.model.summ_writer.add_summary(zeros_T0_summ, self.batch_ix)
                        for ix in range(len(zeros_it)):
                            cizT = zeros_it[ix]
                            if len(indZeros[ix]) > cizT - 1:
                                indZeros[ix] = indZeros[ix][-cizT:]
                                zerosT = np.mean(np.where(np.mean(np.stack(indZeros[ix], axis=0), axis=0) == 1, 1, 0))
                                indZeros[ix] = []
                                zerosTSumm = tf.Summary(value=[tf.Summary.Value(tag='e.zeros/nT%d'%cizT, simple_value=zerosT)])
                                self.model.summ_writer.add_summary(zerosTSumm, self.batch_ix)

                    if hist_summ: self.model.summ_writer.add_summary(hist_summ, self.batch_ix)

                    avg_acc, avg_loss = 0, 0

                # validate
                if fq_VL and self.batch_ix%fq_VL==0:

                    # validation data
                    if self.batcher.data_size['VL']>0:
                        vr = self.test(
                            data_part=  'VL',
                            do_TB=      rep_fq_TB>0)
                        v_acc = vr[0]
                        v_lss = vr[1]
                        if do_TX: print(' >>> %s  acc: %.1f,  loss: %.3f,  f1: %.1f' % ('VL',v_acc,v_lss,vr[2]))

                        t_res['valid.a'].append([v_acc, self.batch_ix])
                        t_res['valid.l'].append([vr[1], self.batch_ix])

                        # store max_VL_acc and save if needed
                        if t_res['max_vl_acc'] is None or v_acc > t_res['max_vl_acc']:
                            t_res['max_vl_acc'] = v_acc
                            if save_max_VL: self.model.saver.save('VL', self.batch_ix)

                    # test data
                    if validation_TS and self.batcher.data_size['TS'] > 0:
                        tr = self.test(
                            data_part=  'TS',
                            do_TB=      rep_fq_TB>0)
                        t_acc = tr[0]
                        t_lss = tr[1]
                        if do_TX: print(' >>> %s  acc: %.1f,  loss: %.3f,  f1: %.1f' % ('TS',t_acc,t_lss,tr[2]))
                        t_res['test.a'].append([t_acc, self.batch_ix])
                        t_res['test.l'].append([t_lss, self.batch_ix])

            self.model.saver.save('TM', self.batch_ix)

            # finally test
            if self.batcher.data_size['TS'] > 0:
                if do_TX: print('finally testing:')
                tr = self.test(
                    data_part=  'TS',
                    do_TB=      rep_fq_TB>0)
                t_acc = tr[0]
                t_lss = tr[1]
                if do_TX: print(' >>>(TM) acc: %.2f, loss: %.3f,  f1: %.2f' % (t_acc, t_lss, tr[2]))
                max_ts_acc = t_acc
                max_ts_lss = t_lss
                max_ts_type = 'TM'
                if self.batcher.data_size['VL']>0 and save_max_VL: # do test for VL checkpoint if VL was saved
                    self.model.saver.load('VL')
                    tr = self.test(
                        data_part=  'TS',
                        do_TB=      rep_fq_TB>0)
                    t_acc = tr[0]
                    t_lss = tr[1]
                    if do_TX: print(' >>>(VL) acc: %.2f, loss: %.3f,  f1: %.2f'%(t_acc, t_lss, tr[2]))
                    if t_acc > max_ts_acc:
                        max_ts_acc = t_acc
                        max_ts_lss = t_lss
                        max_ts_type = 'VL'

                t_res['ts_acc'] =   max_ts_acc
                t_res['ts_loss'] =  max_ts_lss
                t_res['ts_type'] =  max_ts_type

            if self.batcher.data_size['IF'] > 0: t_res['if_pred'] = self.infer() # finally infer

            m_seed_results.append(t_res)

            if self.verb > 0:
                print(' > training loop finished (%.2fmin)'%((time.time()-loop_stime)/60),end='')
                if t_res['ts_type']:
                    all_accs = [tr['ts_acc'] for tr in m_seed_results]
                    print(', ts_acc %.3f (%s) >> %.3f'%(t_res['ts_acc'],t_res['ts_type'],sum(all_accs)/len(all_accs)))
                else: print()

            # delete model folder
            if delete_MFD:
                delete_this_run = True
                if self.do_TB:
                    if m_seed==1: delete_this_run = False
                    elif msix+1==m_seed: delete_this_run = False
                if delete_this_run:
                    m_folder = self.save_TFD + '/_models'
                    shutil.rmtree(m_folder + '/' + self.model['name'])

        all_results = {
            'max_vl_acc':   [],
            'ts_acc':       [],
            'ts_loss':      [],
            'ts_type':      [],
            'if_pred':      []}
        for rix in range(len(m_seed_results)):
            for key in all_results:
                all_results[key].append(m_seed_results[rix].pop(key))
        for key in all_results:
            if all_results[key][0] is None: all_results[key] = None
            else: all_results[key] = np.asarray(all_results[key])

        if m_seed>1:
            # prep TB for m_seed
            if self.do_TB:
                # summarize stats
                for rix in range(1,m_seed):
                    for key in m_seed_results[0].keys():
                        for vix in range(len(m_seed_results[0][key])):
                            m_seed_results[0][key][vix][0] += m_seed_results[rix][key][vix][0]
                # put them into TB
                for key in m_seed_results[0].keys():
                    tag = 'a.train/'
                    if 'valid' in key: tag = 'b.valid/'
                    if 'test' in key: tag = 'c.test/'
                    tag += '2.loss' if '.l' in key else '1.acc'
                    for vix in range(len(m_seed_results[0][key])):
                        summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=m_seed_results[0][key][vix][0]/m_seed)])
                        self.model.summ_writer.add_summary(summ, m_seed_results[0][key][vix][1])

        if close: self.model.session.close()

        return all_results

    # tests model with VL or TS part
    def test(
            self,
            data_part=  'VL',
            do_TB=      None):

        do_TB = do_TB if do_TB is not None else self.do_TB

        n_sampl = 0
        sum_acc = 0
        sum_loss = 0
        labels_Y = []
        labels_O = []
        while True:

            batch = self.batcher.get_batch(part=data_part)
            if not batch: break # empty batch breaks test

            feed = {self.model['train_flag_PH']: False}

            for cix in range(len(batch['lbl'])):
                feed[self.model['lab_PHL'][cix]] = batch['lbl'][cix]

            all_batch_size = 0
            dTypes = ['vec','tks','seq']
            for tp in dTypes:
                if self.model[tp+'_PHL'] is not None:
                    for nS in range(len(batch[tp])):
                        feed[self.model[tp+'_PHL'][nS]] = batch[tp][nS]
                        all_batch_size = len(batch[tp][nS])

            fetches = [
                self.model['predictions'],
                self.model['accuracy'],
                self.model['loss']]

            runVal = self.model.session.run(fetches, feed)

            pred, acc, loss = runVal

            labels_Y += batch['lbl'][-1]
            labels_O += pred.tolist()

            sum_acc += acc*100 * all_batch_size
            sum_loss += loss * all_batch_size
            n_sampl += all_batch_size

        acc = sum_acc / n_sampl
        loss = sum_loss / n_sampl
        f1 = f1_score(labels_Y, labels_O, average='macro')*100

        if do_TB:
            tag = 'b.valid' if data_part == 'VL' else 'c.test'
            acc_summ = tf.Summary(value=[tf.Summary.Value(tag=tag+'/1.acc', simple_value=acc)])
            lss_summ = tf.Summary(value=[tf.Summary.Value(tag=tag+'/2.loss', simple_value=loss)])
            f1_summ = tf.Summary(value=[tf.Summary.Value(tag=tag+'/3.F1', simple_value=f1)])
            self.model.summ_writer.add_summary(acc_summ, self.batch_ix)
            self.model.summ_writer.add_summary(lss_summ, self.batch_ix)
            self.model.summ_writer.add_summary(f1_summ, self.batch_ix)

        return acc, loss, f1

    # runs model IF
    def infer(
            self,
            batcher=    None): # batcher may be given (for direct IF cases)

        probabilities = []
        if not batcher: batcher = self.batcher
        while True:

            batch = batcher.get_batch(part='IF')
            if not batch: break # empty batch breaks test

            feed = {self.model['train_flag_PH']: False}

            dTypes = ['vec','tks','seq']
            for tp in dTypes:
                if self.model[tp+'_PHL'] is not None:
                    for nS in range(len(batch[tp])):
                        feed[self.model[tp+'_PHL'][nS]] = batch[tp][nS]

            probs = self.model.session.run(self.model['probs'], feed)
            probabilities += probs.tolist()

        return probabilities

    # runs model IF with given data (directly)
    def infer_direct(
            self,
            vec :list or tuple= None,
            tks :list or tuple= None,
            seq :list or tuple= None,
            batch_size=         128):

        udd = UDD(IFvec=vec, IFtks=tks, IFseq=seq)
        dvc_data = DVCData(uDD=udd)
        batcher = DVCBatcher(dvc_data=dvc_data, batch_size=batch_size, bsm_IF=1)
        return self.infer(batcher)

    # attention retrieval
    def attend(self):

        batch = None
        while not batch:
            batch = self.batcher.get_batch(part='TS')

        feed = {
            self.model['tksAPH']:          batch['tksA'],
            self.model['labPH']:           batch['lbl'],
            self.model['trainFlagPH']:     False}
        if batch['tksB']: feed[self.model['tksBPH']] = batch['tksB']

        fetches = [
            self.model['predictions'],
            self.model['accuracy'],
            self.model['loss'],
            self.model['attVals']]

        runVal = self.model.session.run(fetches, feed)

        pred, acc, loss, attVals = runVal

        return batch['tksA'], batch['tksB'], batch['lbl'], pred, attVals

    """
    # sample code for attention retrieval
    
    import pandas as pd
    tksA, tksB, lbl, prd, att = dvcStarter.attend()
    print(len(att), att[0].shape)
    pdSamples = []
    columns = ['six', 'dix', 'str', 'a1', 'a2', 'a3', 'a4', 'a5']
    for nS in range(10):
        sample = {col: [] for col in columns}
        print('\n ################# sample %d' % nS)
        print('label %d, pred %d' % (lbl[nS], prd[nS]))
        nWords = len(tks[nS])
        zerosSum = [0, 0, 0, 0, 0]
        nZeros = 0
        for ix in range(nWords):
            tID = tks[nS][ix]
            word = ''
            if tID < ftVEC.vecARR.shape[0]: word = ftVEC.vecITS[tID]
            if tID == ftVEC.vecARR.shape[0]: word = '< oov >'
            if tID > ftVEC.vecARR.shape[0]:
                nZeros += 1
                for lay in range(5): zerosSum[lay] += att[lay][nS, 0, 0, ix]
            else:
                print('%3d %10d %15s ' % (ix, tID, word), end='')
                for lay in range(5): print('%4.1f ' % (nWords * (att[lay][nS, 0, 0, ix])), end='')
                print()
                sample['six'].append(ix)
                sample['dix'].append(tID)
                sample['str'].append(word)
                sample['a1'].append(nWords * (att[0][nS, 0, 0, ix]))
                sample['a2'].append(nWords * (att[1][nS, 0, 0, ix]))
                sample['a3'].append(nWords * (att[2][nS, 0, 0, ix]))
                sample['a4'].append(nWords * (att[3][nS, 0, 0, ix]))
                sample['a5'].append(nWords * (att[4][nS, 0, 0, ix]))
        df = pd.DataFrame(sample, columns=columns)
        df.to_excel('generatedTextClassification/attSamples/sample_%d%d_%d.xlsx' % (lbl[nS], prd[nS], nS))
        print(' ### %3d zeros                 ' % nZeros, end='')
        for lay in range(5): print('%4.1f ' % (nWords * zerosSum[lay] / nZeros), end='')
        print()
    """


# sample running code
if __name__ == '__main__':

    udd = UDD(
        TRsen=      (['This is sentence A.','This is sentence B']),
        TRlbl=      [0,1])
    dvc_data = DVCData(
        uDD=        udd,
        vl_split=   0,
        ts_split=   0,
        use=        True,
        verb=       1)

    from putils.neuralmess.dvc.presets import dvc_presets

    starter = DVCStarter(
        dvc_data=       dvc_data,
        dvc_dict=       dvc_presets['dvc_base'])
    tr_results = starter.train(n_batches=10)