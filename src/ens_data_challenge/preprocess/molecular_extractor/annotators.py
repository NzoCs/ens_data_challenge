from typing import Dict, Optional
import pandas as pd

class ClinVarAnnotator:
    """
    Annotateur ClinVar pour pathogénicité
    """
    
    def __init__(self):
        self.variants = {}
        
    def load_from_file(self, filepath: str) -> Dict:
        """
        Charge ClinVar depuis VCF ou TSV
        Format: chr, pos, ref, alt, clinical_significance, review_status
        """
        try:
            df = pd.read_csv(filepath, sep='\t', low_memory=False)
            print(f"✓ ClinVar chargé: {len(df)} variants")
            
            for _, row in df.iterrows():
                chr_val = str(row.get('chr', row.get('Chromosome', ''))).replace('chr', '')
                pos = row.get('pos', row.get('Position', 0))
                ref = row.get('ref', row.get('ReferenceAllele', ''))
                alt = row.get('alt', row.get('AlternateAllele', ''))
                
                key = f"{chr_val}:{pos}:{ref}:{alt}"
                
                self.variants[key] = {
                    'clinical_significance': row.get('ClinicalSignificance', 'Uncertain'),
                    'review_status': row.get('ReviewStatus', 0),
                    'pathogenicity_score': self._calc_pathogenicity_score(
                        row.get('ClinicalSignificance', '')
                    )
                }
            
            return self.variants
            
        except Exception as e:
            print(f"Erreur chargement ClinVar: {e}")
            return self._load_default_clinvar()
    
    def _calc_pathogenicity_score(self, significance: str) -> float:
        """Convertit significance en score 0-1"""
        sig_lower = str(significance).lower()
        if 'pathogenic' in sig_lower and 'likely' not in sig_lower:
            return 1.0
        elif 'likely pathogenic' in sig_lower:
            return 0.8
        elif 'uncertain' in sig_lower or 'vus' in sig_lower:
            return 0.5
        elif 'likely benign' in sig_lower:
            return 0.2
        elif 'benign' in sig_lower:
            return 0.0
        else:
            return 0.5
    
    def _load_default_clinvar(self) -> Dict:
        """Fallback: variants ClinVar connus"""
        return {
            '17:7577548:C:T': {'clinical_significance': 'Pathogenic', 'review_status': 3, 'pathogenicity_score': 1.0},
            '17:7577538:C:T': {'clinical_significance': 'Pathogenic', 'review_status': 3, 'pathogenicity_score': 1.0},
            '13:32936732:C:T': {'clinical_significance': 'Pathogenic', 'review_status': 4, 'pathogenicity_score': 1.0},  # BRCA2
        }



class ConservationScorer:
    """
    Scores de conservation évolutive
    Utilise GERP, PhyloP, PhastCons
    """
    
    def __init__(self):
        self.scores = {}
        
    def load_from_file(self, filepath: str, score_type: str = 'GERP') -> Dict:
        """
        Charge scores de conservation depuis fichier
        Format: chr, pos, score
        """
        try:
            df = pd.read_csv(filepath, sep='\t', low_memory=False)
            print(f"✓ Conservation {score_type} chargé: {len(df)} positions")
            
            for _, row in df.iterrows():
                chr_val = str(row.get('chr', '')).replace('chr', '')
                pos = row.get('pos', 0)
                key = f"{chr_val}:{pos}"
                
                self.scores[key] = float(row.get('score', 0))
            
            return self.scores
            
        except Exception as e:
            print(f"Erreur chargement Conservation: {e}")
            return self._load_default_conservation()
    
    def get_score(self, chr_val: str, pos: int) -> float:
        """Récupère le score de conservation pour une position"""
        key = f"{chr_val}:{pos}"
        return self.scores.get(key, 0.0)
    
    def _load_default_conservation(self) -> Dict:
        """Fallback: scores élevés pour positions connues"""
        # Positions hautement conservées
        default_scores = {}
        
        # TP53 DNA binding domain
        for pos in range(7577100, 7577600):
            default_scores[f"17:{pos}"] = 5.2
        
        # KRAS GTPase domain
        for pos in range(25245300, 25245400):
            default_scores[f"12:{pos}"] = 4.8
        
        return default_scores


class ProteinDomainAnnotator:
    """
    Annotateur de domaines protéiques
    Utilise InterPro, Pfam, SMART
    """
    
    def __init__(self):
        self.gene_domains = {}
        
    def load_from_file(self, filepath: str) -> Dict:
        """
        Charge domaines depuis fichier
        Format: gene, domain_name, start_aa, end_aa, domain_type
        """
        try:
            df = pd.read_csv(filepath, sep='\t', low_memory=False)
            print(f"✓ Domaines protéiques chargés: {len(df)} domaines")
            
            for _, row in df.iterrows():
                gene = row.get('gene', '')
                if gene not in self.gene_domains:
                    self.gene_domains[gene] = []
                
                self.gene_domains[gene].append({
                    'name': row.get('domain_name', ''),
                    'start': int(row.get('start_aa', 0)),
                    'end': int(row.get('end_aa', 0)),
                    'type': row.get('domain_type', ''),
                    'importance': row.get('importance', 'medium')
                })
            
            return self.gene_domains
            
        except Exception as e:
            print(f"Erreur chargement Domaines: {e}")
            return self._load_default_domains()
    
    def get_domain_at_position(self, gene: str, aa_pos: int) -> Optional[Dict]:
        """Trouve le domaine à une position AA donnée"""
        if gene not in self.gene_domains:
            return None
        
        for domain in self.gene_domains[gene]:
            if domain['start'] <= aa_pos <= domain['end']:
                return domain
        
        return None
    
    def _load_default_domains(self) -> Dict:
        """Fallback: domaines majeurs"""
        return {
            'TP53': [
                {'name': 'Transactivation', 'start': 1, 'end': 61, 'type': 'functional', 'importance': 'high'},
                {'name': 'DNA_binding', 'start': 102, 'end': 292, 'type': 'functional', 'importance': 'critical'},
                {'name': 'Tetramerization', 'start': 324, 'end': 355, 'type': 'structural', 'importance': 'high'},
            ],
            'KRAS': [
                {'name': 'GTPase', 'start': 1, 'end': 166, 'type': 'catalytic', 'importance': 'critical'},
                {'name': 'Switch_I', 'start': 30, 'end': 38, 'type': 'functional', 'importance': 'critical'},
                {'name': 'Switch_II', 'start': 60, 'end': 76, 'type': 'functional', 'importance': 'critical'},
            ],
            'PIK3CA': [
                {'name': 'p85_binding', 'start': 1, 'end': 108, 'type': 'interaction', 'importance': 'medium'},
                {'name': 'RAS_binding', 'start': 190, 'end': 292, 'type': 'interaction', 'importance': 'high'},
                {'name': 'Helical', 'start': 533, 'end': 693, 'type': 'structural', 'importance': 'high'},
                {'name': 'Kinase', 'start': 713, 'end': 1068, 'type': 'catalytic', 'importance': 'critical'},
            ],
            'BRAF': [
                {'name': 'RAS_binding', 'start': 155, 'end': 227, 'type': 'interaction', 'importance': 'high'},
                {'name': 'Kinase', 'start': 457, 'end': 717, 'type': 'catalytic', 'importance': 'critical'},
            ],
            'EGFR': [
                {'name': 'Ligand_binding', 'start': 1, 'end': 621, 'type': 'interaction', 'importance': 'high'},
                {'name': 'Transmembrane', 'start': 622, 'end': 644, 'type': 'structural', 'importance': 'medium'},
                {'name': 'Kinase', 'start': 712, 'end': 979, 'type': 'catalytic', 'importance': 'critical'},
            ],
        }
