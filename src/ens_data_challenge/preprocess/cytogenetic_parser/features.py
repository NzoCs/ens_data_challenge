import logging
import re
import math  # Importation de math au lieu de numpy
import pandas as pd
from typing import List, Optional, DefaultDict, TypedDict, Tuple
from enum import StrEnum

from .parser import ParsedKaryotype, CytogeneticsParser

# =================================================================
# FEATURE ENGINEERING - Séparé du parsing
# =================================================================

class CytogeneticsFeatureExtractor:
    """Extrait des features prédictives à partir d'un karyotype parsé"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, parsed_list: Optional[List[ParsedKaryotype]]) -> dict:
        """
        Interface principale pour extraire toutes les features
        
        Args:
            parsed_list: Liste des clones parsés
            
        Returns:
            Dictionnaire de features pour ML
        """
        if parsed_list is None or not parsed_list:
            return self._empty_features()
        
        if len(parsed_list) > 1:
            return self._mosaic_features(parsed_list)
        
        # Clone unique
        parsed = parsed_list[0]
        if parsed.is_normal:
            return self._normal_features(parsed)
        
        return self._single_clone_features(parsed)
    
    def _empty_features(self) -> dict:
        """Features pour données manquantes"""
        return {
            'is_normal': None,
            'ploidy': None,
            'has_tp53_deletion': None,
            'has_complex_chr3': None,
            'n_abnormalities': None,
            'n_chromosomes_affected': None,
            'has_monosomy_7': None,
            'has_del_5q': None,
            'has_del_7q': None,
            'has_monosomy_y': None,
            'n_deletions': None,
            'n_critical_regions_deleted': None,
            'has_large_deletion': None,
            'is_mosaic': None,
            'n_clones': None,
            'abnormal_clone_percentage': None,
            # Nouveaux scores de risque
            'computed_risk_score': None,
            'mds_ipss_r_cyto_risk': 'Unknown',
            'mds_ipss_cyto_risk': 'Unknown',
            'aml_eln_2022_cyto_risk': 'Unknown',
            'cll_cyto_risk': 'Unknown',
            'mm_riss_cyto_risk': 'Unknown'
        }
    
    def _normal_features(self, parsed: ParsedKaryotype) -> dict:
        """Features pour caryotype normal"""
        return {
            'is_normal': True,
            'ploidy': parsed.ploidy,
            'has_tp53_deletion': False,
            'has_complex_chr3': False,
            'n_abnormalities': 0,
            'n_chromosomes_affected': 0,
            'has_monosomy_7': False,
            'has_del_5q': False,
            'has_del_7q': False,
            'has_monosomy_y': False,
            'n_deletions': 0,
            'n_critical_regions_deleted': 0,
            'has_large_deletion': False,
            'is_mosaic': False,
            'n_clones': 1,
            'abnormal_clone_percentage': 0.0,
            # Nouveaux scores de risque (catégories favorables)
            'computed_risk_score': 0.0,
            'mds_ipss_r_cyto_risk': 'Good',
            'mds_ipss_cyto_risk': 'Good',
            'aml_eln_2022_cyto_risk': 'Intermediate', # Normal est intermédiaire pour ELN
            'cll_cyto_risk': 'Very Low',
            'mm_riss_cyto_risk': 'Standard'
        }
    
    def _single_clone_features(self, parsed: ParsedKaryotype) -> dict:
        """Features pour un clone unique"""
        
        # Calcul des features de base
        feats = {
            'is_normal': False,
            'ploidy': parsed.ploidy,
            'has_tp53_deletion': self._has_tp53_deletion(parsed),
            'has_complex_chr3': self._has_complex_chr3(parsed),
            'n_abnormalities': self._count_abnormalities(parsed),
            'n_chromosomes_affected': self._count_affected_chromosomes(parsed),
            'has_monosomy_7': ('7' in parsed.monosomies),
            'has_del_5q': self._has_del_5q(parsed),
            'has_del_7q': self._has_del_7q(parsed),
            'has_monosomy_y': ('y' in parsed.monosomies),
            'n_deletions': len(parsed.deletions),
            'n_critical_regions_deleted': self._count_critical_deletions(parsed),
            'has_large_deletion': self._has_large_deletion(parsed),
            'is_mosaic': False,
            'n_clones': 1,
            'abnormal_clone_percentage': 100.0,
        }
        
        # Calcul des scores de risque basés sur les features
        feats['mds_ipss_r_cyto_risk'] = self._get_mds_ipss_r_cytogenetic_risk(parsed, feats)
        feats['mds_ipss_cyto_risk'] = self._get_mds_ipss_cytogenetic_risk(parsed, feats)
        feats['aml_eln_2022_cyto_risk'] = self._get_aml_eln_2022_cytogenetic_risk(parsed, feats)
        feats['cll_cyto_risk'] = self._get_cll_hierarchical_risk(parsed, feats)
        feats['mm_riss_cyto_risk'] = self._get_mm_riss_cytogenetic_risk(parsed, feats)
        
        # Calcul du score numérique personnalisé
        feats['computed_risk_score'] = self._compute_risk_score(parsed, feats)
        
        return feats
    
    def _mosaic_features(self, clones: List[ParsedKaryotype]) -> dict:
        """Features pour caryotype mosaïque"""
        # Identifier le clone le plus anormal
        most_abnormal = self._get_most_abnormal_clone(clones)
        
        if most_abnormal is None:
             # S'il n'y a pas de clone anormal (ex: 46,XX[10]/46,XY[10]),
             # on prend les features normales
            if any(c.is_normal for c in clones):
                features = self._normal_features(clones[0])
            else:
                return self._empty_features()
        else:
            features = self._single_clone_features(most_abnormal)
        
        # Ajouter les infos mosaïque
        total_cells = sum(c.cell_count or 0 for c in clones)
        
        features['is_mosaic'] = True
        features['n_clones'] = len(clones)

        abnormal_cell_count = most_abnormal.cell_count or 0
        if most_abnormal.is_normal:
            # Si le "plus anormal" est en fait normal (ex: 46,XX[10]/45,X,-Y[2])
            # On cherche le premier clone VRAIMENT anormal pour le comptage
            abnormal_clone = next((c for c in clones if not c.is_normal), None)
            if abnormal_clone:
                abnormal_cell_count = abnormal_clone.cell_count or 0
            else:
                abnormal_cell_count = 0 # Cas ex: 46,XX / 46,XY

        features['abnormal_clone_percentage'] = (
            (abnormal_cell_count / total_cells * 100) 
            if total_cells > 0 
            else 0
        )
        
        return features
    
    # === Feature calculation methods ===
    
    def _has_tp53_deletion(self, parsed: ParsedKaryotype) -> int:
        """Détecte délétion TP53 (17p13)"""
        for del_info in parsed.deletions:
            if (del_info.get('chromosome') == '17' and 
                del_info.get('arm') == 'p' and 
                del_info.get('start')):
                # TP53 est en 17p13.1, donc toute délétion commençant
                # à p13 ou avant (p12, p11...) l'inclut
                pos = int(re.sub(r'\..*', '', del_info['start'])) # '13.1' -> '13'
                if pos <= 13:
                    return True
        return False
    
    def _has_complex_chr3(self, parsed: ParsedKaryotype) -> int:
        """Détecte anomalies complexes chromosome 3 (inv(3) ou t(3q))"""
        # Inversions on chr3
        for inv in parsed.inversions:
            if inv.get('chromosome') == '3':
                return True

        # Translocations: support new structure ('chromosomes' tuple + 'breakpoints' tuples)
        for trans in parsed.translocations:
            # extract chromosome list
            chrs = []
            if trans.get('chromosomes'):
                val = trans.get('chromosomes')
                if val:
                    chrs = [str(c) for c in val]
            elif trans.get('chromosome'):
                chrs = re.findall(r'(\d+|x|y)', str(trans.get('chromosome')).lower())

            # if 3 involved, check corresponding breakpoint on q-arm
            if '3' in chrs:
                # new structured breakpoints
                bps = trans.get('breakpoints')
                if bps and isinstance(bps, (list, tuple)):
                    for i, c in enumerate(chrs):
                        if c == '3' and i < len(bps):
                            bp = bps[i]
                            # bp can be tuple (arm, band) or string
                            if isinstance(bp, (list, tuple)) and len(bp) >= 1:
                                arm = bp[0]
                            else:
                                arm = str(bp)[0] if bp else ''
                            if str(arm).lower().startswith('q'):
                                return True

                # fallback to legacy 'breakpoint' string
                bp_str = trans.get('breakpoint')
                if bp_str:
                    parts = str(bp_str).split(';')
                    for i, c in enumerate(chrs):
                        if c == '3' and i < len(parts) and parts[i].startswith('q'):
                            return 1

        return False
    
    def _has_del_5q(self, parsed: ParsedKaryotype) -> int:
        """Détecte délétion 5q"""
        return any(
            d.get('chromosome') == '5' and d.get('arm') == 'q'
            for d in parsed.deletions
        )
    
    def _has_del_7q(self, parsed: ParsedKaryotype) -> int:
        """Détecte délétion 7q"""
        return any(
            d.get('chromosome') == '7' and d.get('arm') == 'q'
            for d in parsed.deletions
        )

    def _has_del_11q(self, parsed: ParsedKaryotype) -> int:
        """Détecte délétion 11q"""
        return any(
            d.get('chromosome') == '11' and d.get('arm') == 'q'
            for d in parsed.deletions
        )

    def _has_del_13q(self, parsed: ParsedKaryotype) -> int:
        """Détecte délétion 13q"""
        return any(
            d.get('chromosome') == '13' and d.get('arm') == 'q'
            for d in parsed.deletions
        )

    def _has_trisomy(self, parsed: ParsedKaryotype, chromosomes: List[str]) -> int:
        """Détecte si une trisomie est présente dans la liste"""
        return any(tri in chromosomes for tri in parsed.trisomies)

    def _has_translocation(self, parsed: ParsedKaryotype, chr_pairs: List[tuple[str, str]]) -> int:
        """Détecte une translocation dans une liste de paires"""
        for trans in parsed.translocations:
            # Support new structured 'chromosomes' tuple
            chrs = []
            if trans.get('chromosomes'):
                val = trans.get('chromosomes')
                if val:
                    chrs = [str(c) for c in val]
            elif trans.get('chromosome'):
                chrs = re.findall(r'(\d+|x|y)', str(trans.get('chromosome')).lower())
            elif trans.get('translocation'):
                # derivative might store translocation as tuple
                val = trans.get('translocation')
                if val:
                    chrs = [str(c) for c in val]

            if len(chrs) < 2:
                continue

            chr_set = set(chrs)
            for pair in chr_pairs:
                if pair[0] in chr_set and pair[1] in chr_set:
                    return True
        return False

    def _has_large_deletion(self, parsed: ParsedKaryotype) -> int:
        """Détecte délétions larges (>20 bandes)"""
        for del_info in parsed.deletions:
            start = del_info.get('start')
            end = del_info.get('end')
            if start and end:
                try:
                    # Gérer 'p11' vs 'q22' (pas comparable)
                    if del_info.get('arm'): 
                        # Simplification : si start/end sont sur le même bras
                        size = int(re.sub(r'\..*', '', end)) - int(re.sub(r'\..*', '', start))
                        if size > 20:
                            return True
                    # Si 'start' est 'p' et 'end' est 'q', c'est une large délétion
                    elif del_info.get('start_arm') == 'p' and del_info.get('end_arm') == 'q':
                         return True
                except ValueError:
                    continue # Ignorer si les positions ne sont pas numériques
        return False
    
    def _count_abnormalities(self, parsed: ParsedKaryotype) -> int:
        """Compte le nombre total d'anomalies"""
        return (
            len(parsed.deletions) +
            len(parsed.additions) +
            len(parsed.translocations) +
            len(parsed.inversions) +
            len(parsed.derivatives) +
            len(parsed.monosomies) +
            len(parsed.trisomies)
        )
    
    def _count_affected_chromosomes(self, parsed: ParsedKaryotype) -> int:
        """Compte les chromosomes uniques affectés"""
        affected = set()
        
        for anomaly_list in [parsed.deletions, parsed.additions, 
                             parsed.translocations, parsed.inversions, 
                             parsed.derivatives]:
            for anom in anomaly_list:
                chr_field = anom.get('chromosome', '')
                chr_matches = re.findall(r'(\d+|x|y)', 
                                         str(chr_field).lower())
                affected.update(chr_matches)
        
        affected.update(parsed.monosomies)
        affected.update(parsed.trisomies)
        
        return len(affected)
    
    def _count_critical_deletions(self, parsed: ParsedKaryotype) -> int:
        """Compte les délétions dans les régions critiques"""
        critical_regions = {
            '5': (31, 33),  # 5q31-33
            '7': (22, 36),  # 7q22-36
            '17': (13, 13), # 17p13 (TP53)
        }
        
        count = 0
        for del_info in parsed.deletions:
            chr_num = del_info.get('chromosome', '')
            arm = del_info.get('arm', '')
            
            if chr_num in critical_regions and arm in ('p', 'q'):
                crit_start, crit_end = critical_regions[chr_num]
                start = del_info.get('start')
                end = del_info.get('end')
                
                # Gérer les délétions de bras entier (ex: del(5q))
                if not start:
                    count +=1
                    continue

                if start and end:
                    try:
                        del_start = int(re.sub(r'\..*', '', start))
                        del_end = int(re.sub(r'\..*', '', end))
                        
                        # Chevauchement avec région critique
                        if not (del_end < crit_start or del_start > crit_end):
                            count += 1
                    except ValueError:
                        continue
                elif start: # Délétion terminale (ex: del(5)(q31))
                    try:
                        del_start = int(re.sub(r'\..*', '', start))
                        # Si la délétion commence dans ou avant la région critique
                        if del_start <= crit_end:
                            count += 1
                    except ValueError:
                        continue
        return count
    
    # === NOUVELLES FONCTIONS DE SCORING ===

    def _get_mds_ipss_r_cytogenetic_risk(self, parsed: ParsedKaryotype, feats: dict) -> str:
        """Classification cytogénétique IPSS-R (MDS)"""
        n_abn = feats['n_abnormalities']
        
        # Very Poor
        if n_abn > 3 or feats['has_tp53_deletion']:
            return 'Very Poor'
        
        # Poor
        if feats['has_monosomy_7'] or feats['has_del_7q'] or feats['has_complex_chr3'] or n_abn == 3:
            return 'Poor'
        
        # Very Good
        has_del_11q = self._has_del_11q(parsed)
        if n_abn == 1 and (feats['has_monosomy_y'] or has_del_11q):
            return 'Very Good'
        
        # Good
        has_del_12p = any(d.get('chromosome') == '12' and d.get('arm') == 'p' for d in parsed.deletions)
        has_del_20q = any(d.get('chromosome') == '20' and d.get('arm') == 'q' for d in parsed.deletions)
        
        if n_abn == 0:
            return 'Good' 
        if n_abn == 1 and (feats['has_del_5q'] or has_del_12p or has_del_20q):
            return 'Good'
        if n_abn == 2 and feats['has_del_5q']: # Double incluant del(5q)
            return 'Good'
        
        # Intermediate
        # (inclut +8 seule, del(7q) seule [déjà 'Poor' ?], +19 seule, i(17q), autres)
        # La logique IPSS-R est complexe, 'Poor' prime sur 'Intermediate'
        return 'Intermediate'

    def _get_mds_ipss_cytogenetic_risk(self, parsed: ParsedKaryotype, feats: dict) -> str:
        """Classification cytogénétique IPSS (MDS)"""
        n_abn = feats['n_abnormalities']
        
        # Poor
        if n_abn >= 3 or feats['has_monosomy_7'] or feats['has_del_7q']:
            return 'Poor'
        
        # Good
        has_del_20q = any(d.get('chromosome') == '20' and d.get('arm') == 'q' for d in parsed.deletions)
        
        if n_abn == 0: # Normal
            return 'Good'
        if n_abn == 1 and (feats['has_monosomy_y'] or feats['has_del_5q'] or has_del_20q):
            return 'Good'
        
        # Intermediate
        return 'Intermediate'

    def _get_aml_eln_2022_cytogenetic_risk(self, parsed: ParsedKaryotype, feats: dict) -> str:
        """Classification cytogénétique ELN 2022 (AML)"""
        n_abn = feats['n_abnormalities']
        
        # Favorable
        if self._has_translocation(parsed, [('8', '21'), ('16', '16'), ('15', '17')]):
            return 'Favorable'
        if any(inv.get('chromosome') == '16' for inv in parsed.inversions):
            return 'Favorable'

        # Adverse
        if self._has_translocation(parsed, [('3', '3'), ('6', '9'), ('9', '22')]):
            return 'Adverse'
        if any(inv.get('chromosome') == '3' for inv in parsed.inversions):
            return 'Adverse'
        if '5' in parsed.monosomies or feats['has_del_5q']:
            return 'Adverse'
        if '7' in parsed.monosomies: # -7 (pas del(7q) pour ELN)
            return 'Adverse'
        if '17' in parsed.monosomies or feats['has_tp53_deletion']:
            return 'Adverse'
        if n_abn >= 3: # Complex karyotype
            return 'Adverse'
            
        # Intermediate
        return 'Intermediate'
        
    def _get_cll_hierarchical_risk(self, parsed: ParsedKaryotype, feats: dict) -> str:
        """Classification cytogénétique hiérarchique (CLL)"""
        
        if feats['has_tp53_deletion']:
            return 'Very High'
        
        if self._has_del_11q(parsed):
            return 'High'
        
        if '12' in parsed.trisomies:
            return 'Intermediate'
        
        n_abn = feats['n_abnormalities']
        if self._has_del_13q(parsed) and n_abn == 1:
            return 'Low'
        
        if n_abn == 0:
            return 'Very Low'
        
        return 'Intermediate' # Fallback pour autres anomalies

    def _get_mm_riss_cytogenetic_risk(self, parsed: ParsedKaryotype, feats: dict) -> str:
        """Classification cytogénétique R-ISS (Multiple Myeloma)"""
        
        # High Risk
        if feats['has_tp53_deletion']:
            return 'High'
        if self._has_translocation(parsed, [('4', '14'), ('14', '16')]):
            return 'High'
            
        # Standard Risk
        return 'Standard'

    # === Fin des nouvelles fonctions ===

    def _get_most_abnormal_clone(self, clones: List[ParsedKaryotype]) -> ParsedKaryotype:
        """Identifie le clone le plus anormal dans un mosaïque"""
        
        
        def abnormality_score(clone: ParsedKaryotype) -> float:
            if clone.is_normal:
                return -1.0 # Le clone normal a toujours le score le plus bas
            
            score = 0.0
            score += self._has_tp53_deletion(clone) * 10
            score += (1 if '7' in clone.monosomies else 0) * 10
            score += self._has_complex_chr3(clone) * 8
            score += (1 if '5' in clone.monosomies else 0) * 7
            score += self._has_del_7q(clone) * 6
            score += self._has_del_5q(clone) * 5
            score += self._count_critical_deletions(clone) * 4
            score += self._has_large_deletion(clone) * 3
            score += self._count_abnormalities(clone) * 2
            # Gérer -Y (très bon pronostic) comme étant "moins anormal"
            if 'y' in clone.monosomies and self._count_abnormalities(clone) == 1:
                return 0.1
            return score
        
        # Retourne le clone avec le score max, ou None si tous sont normaux
        max_clone = max(clones, key=abnormality_score)
        return max_clone
    
    def _compute_risk_score(self, parsed: ParsedKaryotype, feats: dict) -> float:
        """
        Calcule un score de risque numérique basé sur les anomalies détectées.
        Score normalisé entre 0 (bon pronostic) et 1 (très mauvais pronostic).
        """
        weights = {
            'has_tp53_deletion': 3.0,
            'has_complex_chr3': 2.0,
            'has_del_5q': 1.0, # Moins d'impact que -7
            'has_del_7q': 1.5,
            'has_monosomy_7': 2.5,
            'n_abnormalities': 0.3,
            'n_critical_regions_deleted': 1.0,
            'has_large_deletion': 1.0,
        }
        
        # calcul pondéré
        raw_score = 0.0
        for feat, w in weights.items():
            value = feats.get(feat)
            if value is not None:
                raw_score += w * float(value)
        
        # Pénalité pour -Y (bon pronostic)
        if feats.get('has_monosomy_y') == 1 and feats.get('n_abnormalities') == 1:
             raw_score = -2.0 # Force un score bas

        # normalisation (sigmoïde)
        normalized = 1 / (1 + math.exp(-0.4 * (raw_score - 4))) # Utilisation de math.exp
        return round(float(normalized), 3)

    def gen_features_to_dataframe(
        self,
        df: pd.DataFrame,
        cyto_col: str = 'CYTOGENETICS',
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Ajoute les colonnes de features au DataFrame à partir de la colonne cytogénétique.

        Args:
            df: DataFrame source contenant une colonne avec les caryotypes ISCN.
            cyto_col: Nom de la colonne contenant la chaîne cytogénétique.
            inplace: Si True, modifie `df` en place et le retourne. Sinon retourne une copie étendue.

        Returns:
            DataFrame avec les nouvelles colonnes de features (une colonne par feature).
        """
        parser = CytogeneticsParser()

        features_rows: List[dict] = []
        # Iterate in order and build list of feature dicts
        for cyto in df[cyto_col].tolist():
            parsed = parser.parse(cyto)
            feats = self.extract_features(parsed)
            features_rows.append(feats)

        # Create features dataframe; ensure consistent columns by using _empty_features keys
        empty = self._empty_features()
        feat_cols = list(empty.keys())
        features_df = pd.DataFrame(
            [{k: row.get(k, empty[k]) for k in feat_cols} for row in features_rows]
        )


        # Return only the features dataframe
        return features_df, feat_cols