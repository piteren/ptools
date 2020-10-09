"""

 2020 (c) piteren

"""

from ptools.pms.foldered_dna import FMDna


# genetic xrossing for Folder_Managed_DNA objects
def gx(
        name_A: str,    # name of parent A
        name_B: str,    # name of parent B
        name_child: str,# name of child
        top_FD: str,    # top folder
        fn_pfx: str,    # dna filename prefix
        gx_rng: dict,   # dict with ranges of gx arguments
        ratio: float=   0.5,
        noise: float=   0.03) -> None:

    pa_fdna = FMDna(
        top_FD=     top_FD,
        name=       name_A,
        fn_pfx=     fn_pfx)
    pa_dna = pa_fdna.get_updated_dna()
    pa_dna = {k: pa_dna[k] for k in gx_rng}

    pb_fdna = FMDna(
        top_FD=     top_FD,
        name=       name_B,
        fn_pfx=     fn_pfx)
    pb_dna = pb_fdna.get_updated_dna()
    pb_dna = {k: pb_dna[k] for k in gx_rng}

    # TODO:
    #  - mix dnas with ratio & noise
    #  - save to a child folder