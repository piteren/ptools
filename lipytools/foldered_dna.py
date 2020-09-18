"""

 2020 (c) piteren

 Folder Managed DNA - dna(dict or ParaDict) saved to a folder

"""
import os

from ptools.pms.paradict import ParaDict



class FMDna:

    def __init__(
            self,
            topfolder: str,         # top folder for dna folder
            name: str,              # name of dna/folder
            fn_pfx: str=  'dna'):   # dna filename prefix

        # full path to .dct file is f'{self._tf}/{self._nm}/{self._fn}.dct'
        self._tf = topfolder
        self._nm = name
        self._fpx = fn_pfx

    def _load_dna(self):
        dna = None
        dna_FD = f'{self._tf}/{self._nm}'
        if os.path.isdir(dna_FD): dna = ParaDict.build(folder=dna_FD, fn_pfx=self._fpx)
        if not dna: dna = ParaDict()
        return dna

    def save_dna(
            self,
            dna: dict) -> None:

        dfd = f'{self._tf}/{self._nm}'
        if not os.path.isdir(dfd): os.mkdir(dfd)

        if type(dna) is not ParaDict: dna = ParaDict(dna)
        dna.save(folder=dfd, fn_pfx=self._fpx)


    # returns updated dna
    def get_updated_dna(
            self,
            dna: dict=  None) -> dict:

        dna_mrg = ParaDict()
        dna_mrg.update(self._load_dna())    # update with dna from folder
        dna_mrg.update(dna)                 # update with given dna
        return dna_mrg

    # saves updated dna in a folder
    def update_dna(
            self,
            dna: dict) -> None:
        self.get_updated_dna(dna)
        self.save_dna(dna)

    # copies dna to target folder
    def copy(
            self,
            name_T: str,
            folder_T: str) -> None:

        dfd = f'{folder_T}/{name_T}'
        if not os.path.isdir(dfd): os.mkdir(dfd)
        dna = self._load_dna()
        if 'name' in dna: dna['name'] = name_T # update name
        dna.save(folder=dfd, fn_pfx=self._fpx)
