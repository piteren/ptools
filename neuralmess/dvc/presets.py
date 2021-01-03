"""

 2019 (c) piteren

    presets for Deep Vector Classifier

"""

# DVC presets dict
dvc_presets = {
    # base preset with defaults (base/simple preset to run dvc_model, defines all params of dvc)
    'dvc_base': {
        'name':             '',     # (str) preset/model name
        'seed':             None,   # (int) seed
        'multi_sen':        1,      # (int) number of sentences (multi-sent mode) >> number of towers
        'classes':          2,      # (int) output shape (num of classes) or list for Multi-Classif, for None does not build classifier graph part
        'vec_width':        512,    # (int) input vector feats width (TOff: None)
        'tok_emb':          None,   # token embeddings variable or tuple (dictW,embW) or None(off)
        'tok_emb_train':    False,  # (bool) train tok_emb (when given tok_emb)
        'tok_emb_add':      None,   # additional token embeddings variable, always trainable (used to add additional embeddings, concatenated then with 'tok_emb')
        'seq_width':        None,   # (int) input vector sequence feats width
        'train_tower':      True,   # False trains only classifier
            # vectors processing
        'inV_drop':         0.0,    # input vector dropout
        'inV_proj':         None,   # input vector projection
        'drt_nLay':         None,   # num of DRT layers
        'drt_scale':        6,      # scale of DRT dense @layer ...shared with ALL DRT
        'drt_drop':         0.0,    # dropout of DRT layer
            # sequence params
        'inS_drop':         0.0,    # input sequence dropout
        'intime_drop':      0.0,    # input time dropout (constant across feat)
        'infeat_drop':      0.0,    # input feat dropout (constant across time)
        'inS_proj':         None,   # input sequence projection (TOff: None)
        'inS_actv':         False,  # inS_proj activation (gelu for True, None for False)
            # seq encoder
        'cnn_nLay':         None,   # num of layers @cnnSeqEncoder
        'rnn_nLay':         None,   # num of layers @lstmSeqEncoder
        'max_seq_len':      None,   # max seq length (for TNS & TAT)
        'tns_nBlocks':      None,   # num TNS blocks
        'enc_drop':         0.0,    # seq encoder (layer) dropout
        'tnsAT_drop':       0.0,    # TNS attention dropout
        'tns_scale':        4,      # scale (dense) for TNS & TAT
            # task attn enc
        'tat_nBlocks':      None,   # num of blocks @taskAttention
        'tat_drop':         0.0,    # dropout of TAT block
        'tatAT_drop':       0.0,    # attention dropout of TAT

        'vtc_drop':         0.0,    # vector to classifier dropout
        'vtc_proj':         None,   # (int) vector to classifier projection
            # DRT(vec) encoder @classifier
        'drtC_nLay':        None,   # num of DRTlayers
        'drtC_drop':        0.0,    # DRTlayer dropout

        'out_drop':         0.0,    # output (before logits) dropout
            # training params
        'iLR':              5e-4,   # lr_scaler param (initial LR)
        'warm_up':          500,    # lr_scaler param
        'ann_base':         1.0,    # lr_scaler param (base for LR*base**(gStep/step))
        'ann_step':         1.0,    # lr_scaler param (step for LR*base**(gStep/step))
        'n_wup_off':        1,      # lr_scaler param
        'do_clip':          True,   # gc_loss_reductor param
        'avt_SVal':         0.1,    # gc_loss_reductor param
        'avt_window':       100,    # gc_loss_reductor param
        'avt_max_upd':      1.5,    # gc_loss_reductor param
        'l2lc':             None,   # l2 loss coefficient
        'batch_size':       512,
        'rgs':              None,   # (dict) rgs pspace
        'verb':             0,      # verbosity
    }
}