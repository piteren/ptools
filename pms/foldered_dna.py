"""

 2020 (c) piteren

 Folder Managed DNA(ParaDict)
    - saves DNA to a folder
    - loads updated DNA from folder

"""

import os

from ptools.pms.paradict import ParaDict



class FMDna:

    def __init__(
            self,
            top_FD: str,            # top folder for dna folder
            name: str,              # name of dna/folder
            fn_pfx: str=  'dna'):   # dna filename prefix

        # full path to .dct file is f'{self._tf}/{self._nm}/{self._fpx}.dct'
        self._tf = top_FD
        if self._tf and not os.path.isdir(self._tf): os.mkdir(self._tf)
        self._nm = name
        self._fpx = fn_pfx

    def get_dna_FD(self):
        return f'{self._tf}/{self._nm}' if self._tf and self._nm else None

    def _load_dna(self):
        dna = None
        dna_FD = self.get_dna_FD()
        if os.path.isdir(dna_FD): dna = ParaDict.build(folder=dna_FD, fn_pfx=self._fpx)
        if not dna: dna = ParaDict()
        return dna

    # saves updated dna in a folder
    def save_dna(
            self,
            dna: dict) -> None:

        dna = self.get_updated_dna(dna)
        dna_FD = self.get_dna_FD()
        if dna_FD:
            if not os.path.isdir(dna_FD): os.mkdir(dna_FD)
            if type(dna) is not ParaDict: dna = ParaDict(dna)
            dna.save(folder=dna_FD, fn_pfx=self._fpx)
        else: assert False, 'ERR: cannot save to NO folder!'

    # returns updated dna
    def get_updated_dna(
            self,
            dna: dict=  None) -> dict:

        dna_mrg = ParaDict()
        dna_mrg.update(self._load_dna())    # update with dna from folder
        if dna: dna_mrg.update(dna)         # update with given dna
        return dna_mrg

    # copies dna to target folder
    def copy(
            self,
            name_T: str,
            folder_T: str) -> None:

        dna_FDT = f'{folder_T}/{name_T}'
        if not os.path.isdir(dna_FDT): os.mkdir(dna_FDT)
        dna = self._load_dna()
        if 'name' in dna: dna['name'] = name_T # update name
        dna.save(folder=dna_FDT, fn_pfx=self._fpx)

    # updates dna without building na object
    @staticmethod
    def static_update_dna(
            dna: dict,
            topfolder: str,
            fn_pfx: str = 'dna'):

        dna_FD = f'{topfolder}/{dna["name"]}'
        fdna = ParaDict.build(folder=dna_FD, fn_pfx=fn_pfx)
        fdna.update(dna)
        fdna.save(folder=dna_FD, fn_pfx=fn_pfx)

    # copies dna without building an object
    @staticmethod
    def static_copy_dna(
            name_S: str,
            name_T: str,
            folder_S: str,
            folder_T: str,
            fn_pfx: str = 'dna'):

        dna_SFD = f'{folder_S}/{name_S}'
        sdna = ParaDict.build(folder=dna_SFD, fn_pfx=fn_pfx)

        dmk_TFD = f'{folder_T}/{name_T}'
        if 'name' in sdna: sdna['name'] = name_T # update name
        if not os.path.isdir(dmk_TFD): os.mkdir(dmk_TFD)
        sdna.save(folder=dmk_TFD, fn_pfx=fn_pfx)


