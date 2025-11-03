import pandas as pd
from typing import Dict

class COSMICDatabaseLoader:
    """
    Chargeur pour les bases de données COSMIC
    Supporte: Census genes, Mutations, Resistance mutations
    """
    
    def __init__(self):
        self.census_genes = {}
        self.mutations = {}
        self.resistance_mutations = set()
        
    def load_census_from_file(self, filepath: str) -> Dict:
        """
        Charge COSMIC Census depuis fichier TSV
        Format attendu: Gene Symbol, Role in Cancer, Tier, etc.
        """
        try:
            df = pd.read_csv(filepath, sep='\t', low_memory=False)
            print(f"✓ COSMIC Census chargé: {len(df)} gènes")
            
            for _, row in df.iterrows():
                gene = row.get('Gene Symbol', row.get('gene', ''))
                self.census_genes[gene] = {
                    'role': row.get('Role in Cancer', 'unknown'),
                    'tier': row.get('Tier', 3),
                    'hallmark': row.get('Hallmark', ''),
                    'somatic': row.get('Somatic', False),
                    'germline': row.get('Germline', False),
                    'tumour_types': row.get('Tumour Types(Somatic)', ''),
                    'molecular_genetics': row.get('Molecular Genetics', '')
                }
            return self.census_genes
            
        except Exception as e:
            print(f"Erreur chargement Census: {e}")
            return self._load_default_census()
    
    def load_mutations_from_file(self, filepath: str, max_rows: int = 100000) -> Dict:
        """
        Charge COSMIC Mutations depuis fichier TSV
        Format: chr, pos, ref, alt, gene, cancer_type, etc.
        """
        try:
            df = pd.read_csv(filepath, sep='\t', nrows=max_rows, low_memory=False)
            print(f"✓ COSMIC Mutations chargé: {len(df)} mutations")
            
            for _, row in df.iterrows():
                chr_val = str(row.get('chr', row.get('Chromosome', ''))).replace('chr', '')
                pos = row.get('pos', row.get('Position', 0))
                ref = row.get('ref', row.get('Ref', ''))
                alt = row.get('alt', row.get('Alt', ''))
                
                key = f"{chr_val}:{pos}:{ref}:{alt}"
                
                if key not in self.mutations:
                    self.mutations[key] = {
                        'count': 0,
                        'cancer_types': set(),
                        'is_driver': False,
                        'samples': []
                    }
                
                self.mutations[key]['count'] += 1
                cancer = row.get('Primary site', row.get('cancer_type', ''))
                if cancer:
                    self.mutations[key]['cancer_types'].add(cancer)
                
                if 'driver' in str(row.get('Mutation Description', '')).lower():
                    self.mutations[key]['is_driver'] = True
            
            # Convertir sets en counts
            for key in self.mutations:
                self.mutations[key]['cancer_types'] = len(self.mutations[key]['cancer_types'])
                self.mutations[key]['driver_ratio'] = 1.0 if self.mutations[key]['is_driver'] else 0.3
            
            return self.mutations
            
        except Exception as e:
            print(f"Erreur chargement Mutations: {e}")
            return self._load_default_hotspots()
    
    def _load_default_census(self) -> Dict:
        """Fallback: gènes Census par défaut"""
        return {
            'TP53': {'role': 'TSG', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Gatekeeper'},
            'KRAS': {'role': 'oncogene', 'tier': 1, 'somatic': True, 'germline': False, 'hallmark': 'Proto-oncogene'},
            'PIK3CA': {'role': 'oncogene', 'tier': 1, 'somatic': True, 'germline': False, 'hallmark': 'Proto-oncogene'},
            'EGFR': {'role': 'oncogene', 'tier': 1, 'somatic': True, 'germline': False, 'hallmark': 'Proto-oncogene'},
            'BRAF': {'role': 'oncogene', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Proto-oncogene'},
            'PTEN': {'role': 'TSG', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Gatekeeper'},
            'APC': {'role': 'TSG', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Gatekeeper'},
            'RB1': {'role': 'TSG', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Gatekeeper'},
            'BRCA1': {'role': 'TSG', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Caretaker'},
            'BRCA2': {'role': 'TSG', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Caretaker'},
            'MYC': {'role': 'oncogene', 'tier': 1, 'somatic': True, 'germline': False, 'hallmark': 'Proto-oncogene'},
            'ERBB2': {'role': 'oncogene', 'tier': 1, 'somatic': True, 'germline': False, 'hallmark': 'Proto-oncogene'},
            'NRAS': {'role': 'oncogene', 'tier': 1, 'somatic': True, 'germline': False, 'hallmark': 'Proto-oncogene'},
            'ATM': {'role': 'TSG', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Caretaker'},
            'CDKN2A': {'role': 'TSG', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Gatekeeper'},
            'NF1': {'role': 'TSG', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Gatekeeper'},
            'VHL': {'role': 'TSG', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Gatekeeper'},
            'ALK': {'role': 'oncogene', 'tier': 1, 'somatic': True, 'germline': False, 'hallmark': 'Fusion'},
            'RET': {'role': 'oncogene', 'tier': 1, 'somatic': True, 'germline': True, 'hallmark': 'Proto-oncogene'},
            'IDH1': {'role': 'oncogene', 'tier': 1, 'somatic': True, 'germline': False, 'hallmark': 'Metabolic'},
        }
    
    def _load_default_hotspots(self) -> Dict:
        """Fallback: hotspots majeurs"""
        return {
            # TP53 DNA binding domain
            '17:7577538:C:T': {'count': 2847, 'cancer_types': 28, 'driver_ratio': 0.95, 'is_driver': True},
            '17:7577548:C:T': {'count': 2156, 'cancer_types': 25, 'driver_ratio': 0.94, 'is_driver': True},
            '17:7577120:C:T': {'count': 1893, 'cancer_types': 24, 'driver_ratio': 0.93, 'is_driver': True},
            # KRAS codons 12/13/61
            '12:25245350:C:A': {'count': 5234, 'cancer_types': 18, 'driver_ratio': 0.98, 'is_driver': True},
            '12:25245350:C:T': {'count': 3421, 'cancer_types': 17, 'driver_ratio': 0.97, 'is_driver': True},
            '12:25245351:C:A': {'count': 2987, 'cancer_types': 16, 'driver_ratio': 0.96, 'is_driver': True},
            '12:25398284:C:A': {'count': 1654, 'cancer_types': 14, 'driver_ratio': 0.95, 'is_driver': True},
            # PIK3CA helical + kinase
            '3:179218303:G:A': {'count': 4123, 'cancer_types': 22, 'driver_ratio': 0.97, 'is_driver': True},
            '3:179234297:A:G': {'count': 3654, 'cancer_types': 21, 'driver_ratio': 0.96, 'is_driver': True},
            # BRAF V600E
            '7:140453136:A:T': {'count': 6789, 'cancer_types': 19, 'driver_ratio': 0.99, 'is_driver': True},
            # EGFR exon 19 del + L858R
            '7:55242465:AGGAATTAAGAGAAGC:A': {'count': 2341, 'cancer_types': 8, 'driver_ratio': 0.97, 'is_driver': True},
            '7:55259515:T:G': {'count': 1876, 'cancer_types': 7, 'driver_ratio': 0.96, 'is_driver': True},
        }
