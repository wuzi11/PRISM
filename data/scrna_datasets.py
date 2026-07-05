import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scanpy as sc
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetMorganFeatureAtomInvGen

_MORGAN_GENERATORS = {}


def _get_morgan_fp_bitstring(mol, num_bits=1024):
    if num_bits not in _MORGAN_GENERATORS:
        _MORGAN_GENERATORS[num_bits] = GetMorganGenerator(
            radius=2,
            fpSize=num_bits,
            atomInvariantsGenerator=GetMorganFeatureAtomInvGen(),
        )
    return _MORGAN_GENERATORS[num_bits].GetFingerprint(mol).ToBitString()


def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1):
    """Encode SMILES + dose into Morgan fingerprint features."""
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits))

    if comb_num==1:
        for i, smiles in enumerate(drug_SMILES_list):
            smi = smiles
            mol = Chem.MolFromSmiles(smi)
            fcfp4 = _get_morgan_fp_bitstring(mol, num_Bits)
            fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
            fcfp4_list = fcfp4_list*np.log10(dose_list[i]+1)
            fcfp4_array[i] = fcfp4_list
    else:
        for i, smiles in enumerate(drug_SMILES_list):
            smiles_list = smiles.split('+')
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                fcfp4 = _get_morgan_fp_bitstring(mol, num_Bits)
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                fcfp4_list = fcfp4_list*np.log10(float(dose_list[i])+1)
                fcfp4_array[i] += fcfp4_list
    return fcfp4_array 

class AnnDataDataset(Dataset):
    def __init__(
        self,
        adata,
        control_adata=None,
        use_drug_structure=False,
        comb_num=1,
        pathway_priors=None,
        pathway_names=None,
    ):
        self.use_drug_structure = use_drug_structure
        self.pathway_priors = pathway_priors or {}
        self.pathway_names = pathway_names
        if type(adata.X)==np.ndarray:
            self.features = torch.tensor(adata.X, dtype=torch.float32)
        else:
            self.features = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        
        if self.use_drug_structure:
            if type(control_adata.X)==np.ndarray:
                self.control_features = torch.tensor(control_adata.X, dtype=torch.float32)
            else:
                self.control_features = torch.tensor(control_adata.X.toarray(), dtype=torch.float32)
                
            self.drug_type_list = adata.obs['SMILES'].to_list()
            self.dose_list = adata.obs['dose'].to_list()
            #self.encoded_obs_tensor = torch.tensor(adata.obs['Group'].copy().values, dtype=torch.float32)
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
            self.encode_drug_doses = Drug_dose_encoder(
                self.drug_type_list, self.dose_list, comb_num=comb_num
            )
            self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
        else:
            self.encoded_obs_tensor = adata.obs['Group'].copy().values

        if self.pathway_priors:
            from prism.model.prism_utils import prior_vector_for_group

            n_pathways = len(self.pathway_names) if self.pathway_names else len(
                next(iter(self.pathway_priors.values()))
            )
            prior_rows = []
            for group in self.encoded_obs_tensor:
                prior_rows.append(
                    prior_vector_for_group(
                        group,
                        self.pathway_priors,
                        pathway_names=self.pathway_names,
                        default=0.0,
                    )
                )
            self.pathway_prior_tensor = torch.tensor(
                np.stack(prior_rows, axis=0), dtype=torch.float32
            )
        else:
            self.pathway_prior_tensor = None
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
       
        if self.use_drug_structure:
            item = {
                'feature': self.features[idx],
                'drug_dose': self.encode_drug_doses[idx],
                'group': self.encoded_obs_tensor[idx],
                'control_feature': self.control_features[idx],
            }
        else:
            item = {'feature': self.features[idx], 'group': self.encoded_obs_tensor[idx]}
        if self.pathway_prior_tensor is not None:
            item['pathway_prior'] = self.pathway_prior_tensor[idx]
        return item
            
    

def prepared_data(
    data_dir=None,
    control_data_dir=None,
    batch_size=64,
    use_drug_structure=False,
    comb_num=1,
    pathway_prior_path=None,
):
     
    
    train_adata = sc.read_h5ad(data_dir)
    if use_drug_structure:
        control_adata = sc.read_h5ad(control_data_dir)
    else:
        control_adata = None

    pathway_priors = None
    pathway_names = None
    if pathway_prior_path:
        from prism.model.prism_utils import load_pathway_prior_store

        store = load_pathway_prior_store(pathway_prior_path)
        pathway_priors = store["priors"]
        pathway_names = store["pathway_names"]
    
    _data_dataset = AnnDataDataset(
        train_adata,
        control_adata,
        use_drug_structure,
        comb_num,
        pathway_priors=pathway_priors,
        pathway_names=pathway_names,
    )


    dataloader = DataLoader(
                _data_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                )
        
    return dataloader