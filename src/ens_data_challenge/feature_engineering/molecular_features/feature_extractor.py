import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


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


class MolecularFeatureExtractor:
    """
    Extracteur de features moléculaires enrichi
    """
    
    def __init__(self, cosmic_loader: COSMICDatabaseLoader = None,
                 clinvar_annotator: ClinVarAnnotator = None,
                 conservation_scorer: ConservationScorer = None,
                 domain_annotator: ProteinDomainAnnotator = None):
        
        # Bases de données
        self.cosmic = cosmic_loader or COSMICDatabaseLoader()
        self.clinvar = clinvar_annotator or ClinVarAnnotator()
        self.conservation = conservation_scorer or ConservationScorer()
        self.domains = domain_annotator or ProteinDomainAnnotator()
        
        # Charger données par défaut si non fournies
        if not cosmic_loader:
            self.cosmic.census_genes = self.cosmic._load_default_census()
            self.cosmic.mutations = self.cosmic._load_default_hotspots()
        
        if not clinvar_annotator:
            self.clinvar.variants = self.clinvar._load_default_clinvar()
        
        if not conservation_scorer:
            self.conservation.scores = self.conservation._load_default_conservation()
        
        if not domain_annotator:
            self.domains.gene_domains = self.domains._load_default_domains()
        
        # Longueurs chromosomes (hg38)
        self.chr_lengths = {
            '1': 248956422, '2': 242193529, '3': 198295559, '4': 190214555,
            '5': 181538259, '6': 170805979, '7': 159345973, '8': 145138636,
            '9': 138394717, '10': 133797422, '11': 135086622, '12': 133275309,
            '13': 114364328, '14': 107043718, '15': 101991189, '16': 90338345,
            '17': 83257441, '18': 80373285, '19': 58617616, '20': 64444167,
            '21': 46709983, '22': 50818468, 'X': 156040895, 'Y': 57227415,
            'MT': 16569
        }
        
        # Trinucléotides pour signatures
        self.trinuc_types = self._generate_trinuc_types()
        
    def _generate_trinuc_types(self) -> List[str]:
        """Génère les 96 types de mutations trinucléotidiques"""
        bases = ['A', 'C', 'G', 'T']
        changes = [('C', 'A'), ('C', 'G'), ('C', 'T'), ('T', 'A'), ('T', 'C'), ('T', 'G')]
        
        trinucs = []
        for before in bases:
            for ref, alt in changes:
                for after in bases:
                    trinucs.append(f"{before}[{ref}>{alt}]{after}")
        
        return trinucs
    
    def normalize_chr(self, chr_value):
        """Normalise les noms de chromosomes"""
        chr_str = str(chr_value).upper().replace('CHR', '').replace('_', '')
        return chr_str if chr_str in self.chr_lengths else None
    
    def classify_mutation_type(self, ref: str, alt: str) -> Tuple[str, str]:
        """Classifie le type de mutation"""
        ref, alt = str(ref), str(alt)
        len_ref, len_alt = len(ref), len(alt)
        
        if len_ref == len_alt == 1:
            transitions = [('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')]
            if (ref, alt) in transitions:
                return 'SNV', 'transition'
            else:
                return 'SNV', 'transversion'
        elif len_ref > len_alt:
            diff = len_ref - len_alt
            if diff % 3 == 0:
                return 'deletion', 'inframe'
            else:
                return 'deletion', 'frameshift'
        elif len_alt > len_ref:
            diff = len_alt - len_ref
            if diff % 3 == 0:
                return 'insertion', 'inframe'
            else:
                return 'insertion', 'frameshift'
        else:
            return 'complex', 'unknown'
    
    def extract_cosmic_features(self, chr_str: str, start: int, ref: str, alt: str, gene: str) -> Dict:
        """Features COSMIC enrichies"""
        features = {}
        
        # Mutation exacte
        key = f"{chr_str}:{start}:{ref}:{alt}"
        if key in self.cosmic.mutations:
            mut_info = self.cosmic.mutations[key]
            features['cosmic_exact_count'] = mut_info['count']
            features['cosmic_cancer_types'] = mut_info['cancer_types']
            features['cosmic_driver_ratio'] = mut_info['driver_ratio']
            features['is_cosmic_hotspot'] = 1
            features['cosmic_log_count'] = np.log1p(mut_info['count'])
        else:
            features['cosmic_exact_count'] = 0
            features['cosmic_cancer_types'] = 0
            features['cosmic_driver_ratio'] = 0
            features['is_cosmic_hotspot'] = 0
            features['cosmic_log_count'] = 0
        
        # Mutations proches (±5bp)
        nearby_count = 0
        for other_key in self.cosmic.mutations:
            parts = other_key.split(':')
            if len(parts) >= 2 and parts[0] == chr_str:
                try:
                    other_pos = int(parts[1])
                    if abs(other_pos - start) <= 5:
                        nearby_count += 1
                except:
                    pass
        
        features['cosmic_nearby_5bp'] = nearby_count
        
        # Features gène
        if gene in self.cosmic.census_genes:
            census = self.cosmic.census_genes[gene]
            features['gene_is_census'] = 1
            features['gene_is_oncogene'] = 1 if census['role'] == 'oncogene' else 0
            features['gene_is_tsg'] = 1 if census['role'] == 'TSG' else 0
            features['gene_tier'] = census['tier']
            features['gene_is_somatic'] = 1 if census.get('somatic', False) else 0
            features['gene_is_germline'] = 1 if census.get('germline', False) else 0
            
            # Hallmark
            hallmark = census.get('hallmark', '')
            features['gene_hallmark_gatekeeper'] = 1 if 'Gatekeeper' in hallmark else 0
            features['gene_hallmark_caretaker'] = 1 if 'Caretaker' in hallmark else 0
        else:
            features['gene_is_census'] = 0
            features['gene_is_oncogene'] = 0
            features['gene_is_tsg'] = 0
            features['gene_tier'] = 3
            features['gene_is_somatic'] = 0
            features['gene_is_germline'] = 0
            features['gene_hallmark_gatekeeper'] = 0
            features['gene_hallmark_caretaker'] = 0
        
        return features
    
    def extract_clinvar_features(self, chr_str: str, start: int, ref: str, alt: str) -> Dict:
        """Features ClinVar"""
        features = {}
        
        key = f"{chr_str}:{start}:{ref}:{alt}"
        if key in self.clinvar.variants:
            var_info = self.clinvar.variants[key]
            features['clinvar_pathogenicity'] = var_info['pathogenicity_score']
            features['clinvar_review_status'] = var_info['review_status']
            features['in_clinvar'] = 1
        else:
            features['clinvar_pathogenicity'] = 0.5  # Uncertain
            features['clinvar_review_status'] = 0
            features['in_clinvar'] = 0
        
        return features
    
    def extract_conservation_features(self, chr_str: str, start: int, end: int) -> Dict:
        """Features de conservation"""
        features = {}
        
        # Score à la position exacte
        score = self.conservation.get_score(chr_str, start)
        features['conservation_gerp'] = score
        features['is_highly_conserved'] = 1 if score > 4.0 else 0
        
        # Score moyen sur fenêtre ±2bp
        window_scores = []
        for pos in range(start - 2, start + 3):
            window_scores.append(self.conservation.get_score(chr_str, pos))
        
        features['conservation_window_mean'] = np.mean(window_scores)
        features['conservation_window_max'] = np.max(window_scores)
        
        return features
    
    def extract_domain_features(self, gene: str, aa_position: Optional[int]) -> Dict:
        """Features de domaines protéiques"""
        features = {}
        
        if aa_position is None or gene not in self.domains.gene_domains:
            features['in_protein_domain'] = 0
            features['domain_importance_score'] = 0
            features['in_catalytic_domain'] = 0
            features['in_binding_domain'] = 0
            return features
        
        domain = self.domains.get_domain_at_position(gene, aa_position)
        
        if domain:
            features['in_protein_domain'] = 1
            
            # Score d'importance
            importance_scores = {'critical': 1.0, 'high': 0.75, 'medium': 0.5, 'low': 0.25}
            features['domain_importance_score'] = importance_scores.get(domain['importance'], 0.5)
            
            # Type de domaine
            features['in_catalytic_domain'] = 1 if domain['type'] in ['catalytic', 'kinase'] else 0
            features['in_binding_domain'] = 1 if 'binding' in domain['type'].lower() else 0
            features['in_structural_domain'] = 1 if domain['type'] == 'structural' else 0
            
            # Position relative dans le domaine
            domain_length = domain['end'] - domain['start'] + 1
            rel_pos = (aa_position - domain['start']) / domain_length
            features['domain_relative_position'] = rel_pos
        else:
            features['in_protein_domain'] = 0
            features['domain_importance_score'] = 0
            features['in_catalytic_domain'] = 0
            features['in_binding_domain'] = 0
            features['in_structural_domain'] = 0
            features['domain_relative_position'] = 0
        
        return features
    
    def extract_positional_features(self, chr_str: str, start: int, end: int) -> Dict:
        """Features positionnelles avancées"""
        features = {}
        
        if chr_str not in self.chr_lengths:
            return {
                'chr_normalized_pos': 0,
                'chr_percentile': 0,
                'is_telomeric': 0,
                'is_centromeric': 0,
                'chr_arm': 'unknown'
            }
        
        chr_len = self.chr_lengths[chr_str]
        
        # Position normalisée
        features['chr_normalized_pos'] = start / chr_len
        features['chr_percentile'] = int((start / chr_len) * 100)
        
        # Bras chromosomique (approximation)
        centromere_pos = chr_len * 0.5  # Simplifié
        if start < centromere_pos:
            features['chr_arm'] = 'p'
            features['distance_to_centromere'] = centromere_pos - start
        else:
            features['chr_arm'] = 'q'
            features['distance_to_centromere'] = start - centromere_pos
        
        features['distance_to_centromere_normalized'] = features['distance_to_centromere'] / chr_len
        
        # Régions spéciales
        telomere_threshold = 5000000
        features['is_telomeric'] = 1 if (start < telomere_threshold or start > chr_len - telomere_threshold) else 0
        
        centromere_region = (0.4 * chr_len, 0.6 * chr_len)
        features['is_centromeric'] = 1 if centromere_region[0] < start < centromere_region[1] else 0
        
        # Distance au télomère le plus proche
        dist_to_p_tel = start
        dist_to_q_tel = chr_len - start
        features['distance_to_nearest_telomere'] = min(dist_to_p_tel, dist_to_q_tel)
        features['distance_to_nearest_telomere_normalized'] = features['distance_to_nearest_telomere'] / chr_len
        
        return features
    
    def extract_structural_features(self, ref: str, alt: str, effect: str) -> Dict:
        """Features structurelles enrichies"""
        features = {}
        
        ref, alt = str(ref), str(alt)
        
        # Longueurs et ratios
        features['ref_length'] = len(ref)
        features['alt_length'] = len(alt)
        features['length_diff'] = len(alt) - len(ref)
        features['length_diff_abs'] = abs(len(alt) - len(ref))
        features['length_ratio'] = len(alt) / max(len(ref), 1)
        
        # Catégories de taille
        features['is_large_variant'] = 1 if max(len(ref), len(alt)) > 50 else 0
        features['is_medium_variant'] = 1 if 10 < max(len(ref), len(alt)) <= 50 else 0
        features['is_small_variant'] = 1 if max(len(ref), len(alt)) <= 10 else 0
        
        # Composition nucléotidique
        def calc_gc_content(seq):
            if len(seq) == 0:
                return 0
            return (seq.count('G') + seq.count('C')) / len(seq)
        
        features['ref_gc_content'] = calc_gc_content(ref)
        features['alt_gc_content'] = calc_gc_content(alt)
        features['gc_content_change'] = features['alt_gc_content'] - features['ref_gc_content']
        
        # Homopolymers (répétitions)
        def has_homopolymer(seq, min_length=3):
            for base in 'ACGT':
                if base * min_length in seq:
                    return 1
            return 0
        
        features['ref_has_homopolymer'] = has_homopolymer(ref)
        features['alt_has_homopolymer'] = has_homopolymer(alt)
        
        # Dinucléotide repeats
        def has_dinuc_repeat(seq):
            for i in range(len(seq) - 5):
                dinuc = seq[i:i+2]
                if seq[i:i+6] == dinuc * 3:
                    return 1
            return 0
        
        features['ref_has_dinuc_repeat'] = has_dinuc_repeat(ref)
        features['alt_has_dinuc_repeat'] = has_dinuc_repeat(alt)
        
        # Complexité de séquence (entropie de Shannon simplifiée)
        def calc_complexity(seq):
            if len(seq) == 0:
                return 0
            counts = Counter(seq)
            entropy = 0
            for count in counts.values():
                p = count / len(seq)
                entropy -= p * np.log2(p) if p > 0 else 0
            return entropy / 2  # Normaliser (max = 2 pour 4 bases)
        
        features['ref_complexity'] = calc_complexity(ref)
        features['alt_complexity'] = calc_complexity(alt)
        
        # Impact fonctionnel basé sur l'effet
        effect_lower = str(effect).lower() if pd.notna(effect) else ''
        
        impact_scores = {
            'frameshift': 5,
            'nonsense': 5,
            'stop_gained': 5,
            'splice_acceptor': 4.5,
            'splice_donor': 4.5,
            'start_lost': 4,
            'stop_lost': 4,
            'missense': 3,
            'inframe_deletion': 3,
            'inframe_insertion': 3,
            'splice_region': 2,
            'synonymous': 1,
            'coding_sequence': 1.5,
            '5_prime_utr': 1,
            '3_prime_utr': 1,
            'intronic': 0.5,
            'intergenic': 0.3
        }
        
        features['predicted_impact_score'] = 2.0  # Default medium
        for key, score in impact_scores.items():
            if key in effect_lower:
                features['predicted_impact_score'] = score
                break
        
        # Catégories binaires d'effet
        features['is_lof'] = 1 if any(x in effect_lower for x in ['frameshift', 'nonsense', 'stop_gained', 'splice']) else 0
        features['is_missense'] = 1 if 'missense' in effect_lower else 0
        features['is_synonymous'] = 1 if 'synonymous' in effect_lower else 0
        features['is_splice'] = 1 if 'splice' in effect_lower else 0
        
        return features
    
    def extract_signature_features(self, mut_type: str, mut_subtype: str, ref: str, alt: str, chr_str: str) -> Dict:
        """Features de signatures mutationnelles"""
        features = {}
        
        # Signatures basiques
        features['is_transition'] = 1 if mut_subtype == 'transition' else 0
        features['is_transversion'] = 1 if mut_subtype == 'transversion' else 0
        features['is_indel'] = 1 if mut_type in ['insertion', 'deletion'] else 0
        
        # Signatures COSMIC spécifiques
        if mut_subtype == 'transition' and ref == 'C' and alt == 'T':
            features['sig_aging_cpg'] = 1  # Signature 1
        else:
            features['sig_aging_cpg'] = 0
        
        if mut_subtype == 'transversion' and ref in ['C', 'G']:
            features['sig_oxidative'] = 1  # Signature 18
        else:
            features['sig_oxidative'] = 0
        
        if mut_type in ['insertion', 'deletion'] and len(ref + alt) <= 5:
            features['sig_mmr_indel'] = 1  # Signature 6
        else:
            features['sig_mmr_indel'] = 0
        
        # Pattern C>A (smoking signature 4)
        if ref == 'C' and alt == 'A':
            features['sig_smoking'] = 1
        else:
            features['sig_smoking'] = 0
        
        # Pattern T>C (APOBEC signature 2/13)
        if ref == 'T' and alt == 'C':
            features['sig_apobec'] = 1
        else:
            features['sig_apobec'] = 0
        
        # UV signature (C>T at dipyrimidine)
        if mut_subtype == 'transition' and ref == 'C':
            features['sig_uv_like'] = 1
        else:
            features['sig_uv_like'] = 0
        
        return features
    
    def extract_mutation_context_features(self, chr_str: str, start: int, ref: str, alt: str, all_mutations_df: pd.DataFrame = None) -> Dict:
        """Features de contexte mutationnel (clustering, kataegis)"""
        features = {}
        
        if all_mutations_df is None or len(all_mutations_df) == 0:
            return {
                'n_mutations_100kb': 0,
                'n_mutations_1mb': 0,
                'mutation_density_local': 0,
                'is_clustered': 0,
                'min_distance_to_other_mut': 1e9
            }
        
        # Filtrer mutations sur même chromosome
        same_chr = all_mutations_df[all_mutations_df['chr'] == chr_str].copy()
        
        if len(same_chr) == 0:
            return {
                'n_mutations_100kb': 0,
                'n_mutations_1mb': 0,
                'mutation_density_local': 0,
                'is_clustered': 0,
                'min_distance_to_other_mut': 1e9
            }
        
        # Calculer distances
        same_chr['distance'] = np.abs(same_chr['start'] - start)
        same_chr_filtered = same_chr[same_chr['distance'] > 0]  # Exclure mutation elle-même
        
        # Comptages dans fenêtres
        features['n_mutations_100kb'] = len(same_chr_filtered[same_chr_filtered['distance'] <= 100000])
        features['n_mutations_1mb'] = len(same_chr_filtered[same_chr_filtered['distance'] <= 1000000])
        features['n_mutations_10mb'] = len(same_chr_filtered[same_chr_filtered['distance'] <= 10000000])
        
        # Densité locale
        features['mutation_density_local'] = features['n_mutations_100kb'] / 100  # Par 100kb
        
        # Clustering (≥6 mutations dans 1kb = kataegis)
        n_in_1kb = len(same_chr_filtered[same_chr_filtered['distance'] <= 1000])
        features['is_clustered'] = 1 if n_in_1kb >= 6 else 0
        features['n_mutations_1kb'] = n_in_1kb
        
        # Distance minimale à une autre mutation
        if len(same_chr_filtered) > 0:
            features['min_distance_to_other_mut'] = same_chr_filtered['distance'].min()
            features['log_min_distance'] = np.log1p(features['min_distance_to_other_mut'])
        else:
            features['min_distance_to_other_mut'] = 1e9
            features['log_min_distance'] = np.log1p(1e9)
        
        return features
    
    def extract_all_features(self, mutations_df: pd.DataFrame, include_context: bool = True) -> pd.DataFrame:
        """
        Pipeline complet d'extraction de features
        
        Parameters:
        -----------
        mutations_df : DataFrame avec colonnes [chr, start, end, ref, alt, gene, effect]
        include_context : Si True, calcule features de contexte mutationnel (plus lent)
        
        Returns:
        --------
        DataFrame avec toutes les features extraites
        """
        all_features = []
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION DE FEATURES MOLÉCULAIRES")
        print(f"{'='*60}")
        print(f"Mutations à traiter: {len(mutations_df)}")
        print(f"Contexte mutationnel: {'Oui' if include_context else 'Non'}")
        
        # Préparer données pour contexte
        if include_context:
            context_df = mutations_df[['chr', 'start']].copy()
            for col in ['chr', 'start']:
                if col in context_df.columns:
                    context_df[col] = context_df[col].astype(str) if col == 'chr' else pd.to_numeric(context_df[col], errors='coerce')
        else:
            context_df = None
        
        # Extraire AA position si disponible
        aa_positions = {}
        if 'aa_position' in mutations_df.columns:
            for idx, row in mutations_df.iterrows():
                if pd.notna(row.get('aa_position')):
                    aa_positions[idx] = int(row['aa_position'])
        
        # Extraction
        for idx, row in mutations_df.iterrows():
            if idx % 500 == 0:
                print(f"  Progression: {idx}/{len(mutations_df)} ({idx/len(mutations_df)*100:.1f}%)")
            
            features = {'mutation_id': idx}
            
            # Extraire valeurs
            chr_str = self.normalize_chr(row.get('chr', ''))
            if chr_str is None:
                continue
            
            start = int(row.get('start', 0))
            end = int(row.get('end', start))
            ref = str(row.get('ref', ''))
            alt = str(row.get('alt', ''))
            gene = str(row.get('gene', ''))
            effect = row.get('effect', '')
            
            # Classification mutation
            mut_type, mut_subtype = self.classify_mutation_type(ref, alt)
            features['mutation_type'] = mut_type
            features['mutation_subtype'] = mut_subtype
            
            # 1. Features COSMIC (base de données externe)
            cosmic_feat = self.extract_cosmic_features(chr_str, start, ref, alt, gene)
            features.update(cosmic_feat)
            
            # 2. Features ClinVar (pathogénicité)
            clinvar_feat = self.extract_clinvar_features(chr_str, start, ref, alt)
            features.update(clinvar_feat)
            
            # 3. Features conservation évolutive
            conservation_feat = self.extract_conservation_features(chr_str, start, end)
            features.update(conservation_feat)
            
            # 4. Features domaines protéiques
            aa_pos = aa_positions.get(idx)
            domain_feat = self.extract_domain_features(gene, aa_pos)
            features.update(domain_feat)
            
            # 5. Features positionnelles
            pos_feat = self.extract_positional_features(chr_str, start, end)
            features.update(pos_feat)
            
            # 6. Features structurelles
            struct_feat = self.extract_structural_features(ref, alt, effect)
            features.update(struct_feat)
            
            # 7. Features signatures mutationnelles
            sig_feat = self.extract_signature_features(mut_type, mut_subtype, ref, alt, chr_str)
            features.update(sig_feat)
            
            # 8. Features contexte mutationnel (optionnel)
            if include_context and context_df is not None:
                context_feat = self.extract_mutation_context_features(chr_str, start, ref, alt, context_df)
                features.update(context_feat)
            
            # 9. Features composites
            features['chr'] = chr_str
            features['gene'] = gene
            features['mutation_signature'] = f"{chr_str}:{start}:{ref}:{alt}"
            features['position_hash'] = hash(f"{chr_str}:{start}") % 100000
            
            # 10. Score composite de "driver likelihood"
            driver_score = (
                features['cosmic_driver_ratio'] * 0.3 +
                features['is_cosmic_hotspot'] * 0.2 +
                features['gene_is_census'] * 0.15 +
                features['conservation_gerp'] / 10 * 0.15 +
                features['domain_importance_score'] * 0.1 +
                features['predicted_impact_score'] / 5 * 0.1
            )
            features['composite_driver_score'] = driver_score
            
            all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        
        print(f"\n{'='*60}")
        print(f"✓ EXTRACTION TERMINÉE")
        print(f"{'='*60}")
        print(f"Mutations traitées: {len(features_df)}")
        print(f"Features générées: {len(features_df.columns)}")
        print(f"\nCatégories de features:")
        print(f"  - COSMIC: {sum(1 for c in features_df.columns if 'cosmic' in c)}")
        print(f"  - ClinVar: {sum(1 for c in features_df.columns if 'clinvar' in c)}")
        print(f"  - Conservation: {sum(1 for c in features_df.columns if 'conservation' in c)}")
        print(f"  - Domaines: {sum(1 for c in features_df.columns if 'domain' in c)}")
        print(f"  - Signatures: {sum(1 for c in features_df.columns if 'sig_' in c)}")
        print(f"  - Structurelles: {sum(1 for c in features_df.columns if any(x in c for x in ['length', 'gc', 'complexity']))}")
        
        return features_df
    
    def aggregate_patient_features(self, features_df: pd.DataFrame, patient_col: str = 'patient_id') -> pd.DataFrame:
        """Agrège features au niveau patient"""
        
        if patient_col not in features_df.columns:
            print(f"Warning: Colonne {patient_col} non trouvée")
            return features_df
        
        print(f"\n{'='*60}")
        print(f"AGRÉGATION AU NIVEAU PATIENT")
        print(f"{'='*60}")
        
        agg_features = []
        
        for patient, group in features_df.groupby(patient_col):
            feat = {'patient_id': patient}
            
            # Comptages basiques
            feat['total_mutations'] = len(group)
            feat['n_unique_genes'] = group['gene'].nunique()
            feat['n_unique_chromosomes'] = group['chr'].nunique()
            
            # Distribution types de mutations
            for mtype in ['SNV', 'insertion', 'deletion']:
                n = (group['mutation_type'] == mtype).sum()
                feat[f'n_{mtype}'] = n
                feat[f'ratio_{mtype}'] = n / len(group)
            
            for subtype in ['transition', 'transversion', 'frameshift', 'inframe']:
                n = (group['mutation_subtype'] == subtype).sum()
                feat[f'n_{subtype}'] = n
                feat[f'ratio_{subtype}'] = n / len(group) if len(group) > 0 else 0
            
            # Features COSMIC agrégées
            feat['n_cosmic_hotspots'] = group['is_cosmic_hotspot'].sum()
            feat['ratio_cosmic_hotspots'] = group['is_cosmic_hotspot'].mean()
            feat['mean_cosmic_count'] = group['cosmic_exact_count'].mean()
            feat['max_cosmic_count'] = group['cosmic_exact_count'].max()
            feat['sum_cosmic_log_count'] = group['cosmic_log_count'].sum()
            feat['mean_driver_ratio'] = group['cosmic_driver_ratio'].mean()
            feat['weighted_driver_score'] = (group['cosmic_driver_ratio'] * group['cosmic_exact_count']).sum() / max(group['cosmic_exact_count'].sum(), 1)
            
            # Gènes cancer
            feat['n_census_genes'] = group['gene_is_census'].sum()
            feat['n_oncogenes'] = group['gene_is_oncogene'].sum()
            feat['n_tsg'] = group['gene_is_tsg'].sum()
            feat['n_tier1_genes'] = (group['gene_tier'] == 1).sum()
            feat['ratio_census_genes'] = group['gene_is_census'].mean()
            
            # Conservation
            feat['mean_conservation'] = group['conservation_gerp'].mean()
            feat['n_highly_conserved'] = group['is_highly_conserved'].sum()
            feat['max_conservation'] = group['conservation_gerp'].max()
            
            # Domaines protéiques
            feat['n_in_domains'] = group['in_protein_domain'].sum()
            feat['n_in_catalytic'] = group['in_catalytic_domain'].sum()
            feat['mean_domain_importance'] = group['domain_importance_score'].mean()
            
            # Impact fonctionnel
            feat['n_lof'] = group['is_lof'].sum()
            feat['n_missense'] = group['is_missense'].sum()
            feat['ratio_lof'] = group['is_lof'].mean()
            feat['mean_impact_score'] = group['predicted_impact_score'].mean()
            feat['max_impact_score'] = group['predicted_impact_score'].max()
            
            # Signatures mutationnelles
            for sig in ['sig_aging_cpg', 'sig_oxidative', 'sig_mmr_indel', 'sig_smoking', 'sig_apobec', 'sig_uv_like']:
                if sig in group.columns:
                    feat[f'{sig}_count'] = group[sig].sum()
                    feat[f'{sig}_ratio'] = group[sig].mean()
            
            # ClinVar
            feat['n_in_clinvar'] = group['in_clinvar'].sum()
            feat['mean_clinvar_pathogenicity'] = group['clinvar_pathogenicity'].mean()
            
            # Clustering / contexte
            if 'is_clustered' in group.columns:
                feat['n_clustered_mutations'] = group['is_clustered'].sum()
                feat['mean_mutation_density'] = group['mutation_density_local'].mean()
                feat['max_mutations_100kb'] = group['n_mutations_100kb'].max()
            
            # Score composite
            feat['mean_driver_score'] = group['composite_driver_score'].mean()
            feat['max_driver_score'] = group['composite_driver_score'].max()
            feat['sum_driver_score'] = group['composite_driver_score'].sum()
            
            # Distribution chromosomique
            chr_dist = group['chr'].value_counts()
            feat['max_mutations_single_chr'] = chr_dist.max()
            feat['entropy_chr_distribution'] = -sum((chr_dist / len(group)) * np.log2(chr_dist / len(group)))
            
            # Complexité
            feat['mean_ref_length'] = group['ref_length'].mean()
            feat['mean_alt_length'] = group['alt_length'].mean()
            feat['mean_length_diff'] = group['length_diff_abs'].mean()
            feat['n_large_variants'] = group['is_large_variant'].sum()
            
            agg_features.append(feat)
        
        agg_df = pd.DataFrame(agg_features)
        
        print(f"✓ Agrégation terminée")
        print(f"  Patients: {len(agg_df)}")
        print(f"  Features: {len(agg_df.columns)}")
        print(f"{'='*60}\n")
        
        return agg_df