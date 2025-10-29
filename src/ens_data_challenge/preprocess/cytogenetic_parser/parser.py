import re
import pandas as pd
import logging
from typing import List, Optional, Any, Union, Dict
from collections import defaultdict

from .types import (
    CytogeneticsPatterns
)


class ParsedKaryotype:
    """Représentation clean d'un caryotype parsé, sans features calculées"""
    
    def __init__(self):
        self.ploidy: Optional[int] = None
        self.sex_chromosomes: Optional[str] = None
        
        # Anomalies structurelles détaillées
        self.deletions: List[Dict[str, Any]] = []
        self.additions: List[Dict[str, Any]] = []
        self.translocations: List[Dict[str, Any]] = []
        self.inversions: List[Dict[str, Any]] = []
        self.derivatives: List[Dict[str, Any]] = []
        self.duplications: List[Dict[str, Any]] = []
        self.triplications: List[Dict[str, Any]] = []
        self.isochromosomes: List[Dict[str, Any]] = []
        self.markers: List[str] = []
        
        # Anomalies numériques
        self.monosomies: List[str] = []
        self.trisomies: List[str] = []
        
        self.cell_count: Optional[int] = None
    
    @property
    def is_normal(self) -> bool:
        """Détermine si le karyotype est normal"""
        return (
            self.ploidy == 46 and
            not self.deletions and
            not self.additions and
            not self.translocations and
            not self.inversions and
            not self.derivatives and
            not self.duplications and
            not self.triplications and
            not self.isochromosomes and
            not self.markers and
            not self.monosomies and
            not self.trisomies
        )
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour faciliter l'usage"""
        return {
            'is_normal': self.is_normal,
            'ploidy': self.ploidy,
            'sex_chromosomes': self.sex_chromosomes,
            'deletions': self.deletions,
            'additions': self.additions,
            'translocations': self.translocations,
            'inversions': self.inversions,
            'derivatives': self.derivatives,
            'duplications': self.duplications,
            'triplications': self.triplications,
            'isochromosomes': self.isochromosomes,
            'markers': self.markers,
            'monosomies': self.monosomies,
            'trisomies': self.trisomies,
            'cell_count': self.cell_count,
        }
    

class CytogeneticsParser:
    """Parser clean pour caryotypes ISCN - parsing uniquement, pas de features"""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.seen_special_types = set()
    
    def parse(self, cyto_string: Any) -> Optional[List[ParsedKaryotype]]:
        """
        Point d'entrée principal pour parser un caryotype
        
        Returns:
            Liste de ParsedKaryotype si parsing réussi, None si invalide/vide
        """
        if pd.isna(cyto_string):
            return None
        
        cyto_str = str(cyto_string).strip()
        if not cyto_str:
            return None
        
        # 1. Vérifier les cas spéciaux d'abord
        special_obj = self._identify_special_case(cyto_str)
        if special_obj is not None:
            # determine a short type string for logging (set by _identify_special_case)
            type_name = getattr(special_obj, 'karyotype_type', None)  or 'special'
            if type_name not in self.seen_special_types:
                self.logger.info(f"Encountered new special cytogenetic type: {type_name}")
                self.seen_special_types.add(type_name)
            return [special_obj]
            # Per requirement: for textual special cases we return missing (None)
            return None
            
        cyto_lower = cyto_str.lower()
        
        # Cas 1: Normal
        if self._is_normal_karyotype(cyto_lower):
            return self._parse_normal(cyto_lower)
        
        # Cas 2: Mosaïque
        if '/' in cyto_lower:
            return self._parse_mosaic(cyto_lower)
        
        # Cas 3: Clone unique avec anomalies
        return self._parse_single_clone(cyto_lower)
    
    # =================================================================
    # PARSING DE BASE
    # =================================================================
    
    def _is_normal_karyotype(self, cyto: str) -> bool:
        """Vérifie si le caryotype est normal (46,XX ou 46,XY)"""
        return bool(re.match(CytogeneticsPatterns.NORMAL_KARYOTYPE, cyto, re.IGNORECASE))
    
    def _parse_normal(self, cyto: str) -> List[ParsedKaryotype]:
        """Parse un caryotype normal"""
        result = ParsedKaryotype()
        result.ploidy = self._extract_ploidy(cyto)
        result.sex_chromosomes = self._extract_sex_chromosomes(cyto)
        return [result]
    
    def _parse_mosaic(self, cyto: str) -> List[ParsedKaryotype]:
        """
        Parse un caryotype mosaïque
        Format: clone1[n]/clone2[m]/clone3[k]
        """
        clones = []
        
        for clone_str in cyto.split('/'):
            match = re.match(CytogeneticsPatterns.MOSAIC_CLONE, clone_str.strip())
            if match:
                karyotype = match.group(1).strip()
                cell_count = int(match.group(2))
                
                parsed_clones = self._parse_single_clone(karyotype)
                if parsed_clones:
                    clone = parsed_clones[0]
                    clone.cell_count = cell_count
                    clones.append(clone)
        
        return clones
    
    def _parse_single_clone(self, cyto: str) -> List[ParsedKaryotype]:
        """Parse un seul clone (sans mosaïcisme)"""
        result = ParsedKaryotype()
        
        # Extraction de base
        result.ploidy = self._extract_ploidy(cyto)
        result.sex_chromosomes = self._extract_sex_chromosomes(cyto)
        
        # Anomalies numériques
        result.monosomies = self._extract_monosomies(cyto)
        result.trisomies = self._extract_trisomies(cyto)
        
        # Anomalies structurelles
        result.deletions = self._extract_deletions(cyto)
        result.additions = self._extract_additions(cyto)
        result.translocations = self._extract_translocations(cyto)
        result.inversions = self._extract_inversions(cyto)
        result.derivatives = self._extract_derivatives(cyto)
        result.duplications = self._extract_duplications(cyto)
        result.triplications = self._extract_triplications(cyto)
        result.isochromosomes = self._extract_isochromosomes(cyto)
        result.markers = self._extract_markers(cyto)
        
        return [result]
    
    # =================================================================
    # EXTRACTION DES ÉLÉMENTS DE BASE
    # =================================================================
    
    def _extract_ploidy(self, cyto: str) -> Optional[int]:
        """Extrait le nombre de chromosomes"""
        match = re.match(CytogeneticsPatterns.PLOIDY, cyto)
        return int(match.group(1)) if match else None
    

    def _identify_special_case(self, cyto_str: str) -> Optional[ParsedKaryotype]:
        """
        Identifie les cas spéciaux (texte seul) et retourne un instance
        de `ParsedKaryotype` construite avec le constructeur par défaut
        puis peuplée via ses attributs.
        """
        if not cyto_str:
            return None

        cyto_lower = cyto_str.strip().lower()

        # --- Cas 1: Échec de l'analyse / Non évalué ---
        fail_patterns = [
            r'not evaluated', r'no result', r'echec', r'failed',
            r'unsuitable', r'pas d.analyse', r'ininterpretable',
            r'no growth', r'pas de mitose'
        ]

        if any(re.search(pat, cyto_lower) for pat in fail_patterns):
            return None

        # --- Cas 2: Explicitement Normal ---
        normal_patterns = [
            r'^normal$', r'^normal karyotype$', r'^karyotype normal$',
            r'^46,xx$', r'^46,xy$'
        ]
        if any(re.fullmatch(pat, cyto_lower) for pat in normal_patterns):
            sex = 'XX' if 'xx' in cyto_lower else ('XY' if 'xy' in cyto_lower else None)
            p = ParsedKaryotype()
            p.sex_chromosomes = sex
            p.ploidy = 46
            return p

        # Helper to create dummy anomalies for "complex" textual cases
        def _get_dummy_complex(n=3):
            return [{'raw': f'der({i+1})', 'chromosome': str(i+1)} for i in range(n)]

        # Case: complex with -7 or del(7q)
        if re.search(r'complex.*(-7|del.*7q)', cyto_lower):
            monosomies = ['7'] if '-7' in cyto_lower else []
            deletions = []
            if 'del(7q)' in cyto_lower:
                deletions.append({
                    'type': 'deletion',
                    'chromosome': '7',
                    'arm': 'q',
                    'start_arm': None,
                    'start': None,
                    'end_arm': None,
                    'end': None
                })
            derivatives = _get_dummy_complex(3)
            p = ParsedKaryotype()
            p.monosomies = monosomies
            p.deletions = deletions
            p.derivatives = derivatives
            return p

        # Case: complex with -5 or del(5q)
        if re.search(r'complex.*(-5|del.*5q)', cyto_lower):
            monosomies = ['5'] if '-5' in cyto_lower else []
            deletions = []
            if 'del(5q)' in cyto_lower:
                deletions.append({
                    'type': 'deletion',
                    'chromosome': '5',
                    'arm': 'q',
                    'start_arm': None,
                    'start': None,
                    'end_arm': None,
                    'end': None
                })
            derivatives = _get_dummy_complex(3)
            p = ParsedKaryotype()
            p.monosomies = monosomies
            p.deletions = deletions
            p.derivatives = derivatives
            return p

        # Generic 'complex'
        if 'complex' in cyto_lower:
            derivatives = _get_dummy_complex(4)
            p = ParsedKaryotype()
            p.derivatives = derivatives
            return p

        # Treat 'abnormal' as a complex-like textual label
        if 'abnormal' in cyto_lower:
            derivatives = _get_dummy_complex(4)
            p = ParsedKaryotype()
            p.derivatives = derivatives
            return p

        # Simple high-risk text rules
        if re.fullmatch(r'-7', cyto_lower):
            p = ParsedKaryotype()
            p.monosomies = ['7']
            return p

        if re.fullmatch(r'del\(5q\)', cyto_lower):
            deletions = [{
                'type': 'deletion',
                'chromosome': '5',
                'arm': 'q',
                'start_arm': None,
                'start': None,
                'end_arm': None,
                'end': None
            }]
            p = ParsedKaryotype()
            p.deletions = deletions
            return p

        if re.fullmatch(r'\+8', cyto_lower):
            p = ParsedKaryotype()
            p.trisomies = ['8']
            return p

        return None
    
    def _extract_sex_chromosomes(self, cyto: str) -> Optional[str]:
        """Extrait les chromosomes sexuels"""
        match = re.search(CytogeneticsPatterns.SEX_CHROMOSOMES, cyto, re.IGNORECASE)
        return match.group(1).upper() if match else None
    
    def _extract_monosomies(self, cyto: str) -> List[str]:
        """Extrait les monosomies"""
        return re.findall(CytogeneticsPatterns.MONOSOMY, cyto)
    
    def _extract_trisomies(self, cyto: str) -> List[str]:
        """Extrait les trisomies"""
        return re.findall(CytogeneticsPatterns.TRISOMY, cyto)
    
    # =================================================================
    # EXTRACTION DES ANOMALIES STRUCTURELLES
    # =================================================================
    
    def _extract_deletions(self, cyto: str) -> List[dict]:
        """Extrait les délétions avec positions détaillées si disponibles"""
        deletions = []
        
        # Essayer pattern détaillé: del(5)(q12q34)
        matches = re.findall(CytogeneticsPatterns.DELETION_POSITIONS, cyto, re.IGNORECASE)
        if matches:
            for chr_num, start_arm, start_band, end_arm, end_band in matches:
                deletions.append({
                    'type': 'deletion',
                    'chromosome': chr_num,
                    'arm': start_arm.lower() if start_arm else None,
                    'start_arm': start_arm.lower() if start_arm else None,
                    'start': start_band,
                    'end_arm': end_arm.lower() if end_arm else None,
                    'end': end_band if end_band else None
                })
            return deletions
        
        # Fallback pattern simple: del(5)
        matches = re.findall(CytogeneticsPatterns.DELETION, cyto, re.IGNORECASE)
        for chr_num in matches:
            deletions.append({'type': 'deletion', 'chromosome': chr_num})
        
        return deletions
    
    def _extract_additions(self, cyto: str) -> List[dict]:
        """Extrait les additions"""
        additions = []
        
        # Pattern détaillé: add(5)(q12)
        matches = re.findall(CytogeneticsPatterns.ADDITION_POSITIONS, cyto, re.IGNORECASE)
        if matches:
            for chr_num, arm, band in matches:
                data = {
                    'type': 'addition',
                    'chromosome': chr_num,
                    'arm': arm.lower() if arm else None,
                    'start': band
                }
                additions.append(data)
            return additions
        
        # Pattern simple: add(5)
        matches = re.findall(CytogeneticsPatterns.ADDITION, cyto, re.IGNORECASE)
        for chr_num in matches:
            additions.append({'type': 'addition', 'chromosome': chr_num})
        
        return additions
    
    def _extract_translocations(self, cyto: str) -> List[dict]:
        """Extrait les translocations"""
        translocations = []
        # Pattern détaillé: t(8;21)(q22;q22)
        matches = re.findall(CytogeneticsPatterns.TRANSLOCATION_POSITIONS, cyto, re.IGNORECASE)
        if matches:
            for chr1, chr2, arm1, band1, arm2, band2 in matches:
                # Uniform structure: use tuples for chromosomes and breakpoints
                translocations.append({
                    'type': 'translocation',
                    'chromosomes': (chr1, chr2),
                    'breakpoints': ((arm1, band1), (arm2, band2))
                })
            return translocations

        # Pattern simple: t(8;21)
        matches = re.findall(CytogeneticsPatterns.TRANSLOCATION, cyto, re.IGNORECASE)
        for chr1, chr2 in matches:
            translocations.append({'type': 'translocation', 'chromosomes': (chr1, chr2)})

        return translocations

    def _extract_inversions(self, cyto: str) -> List[dict]:
        """Extrait les inversions"""
        inversions = []
        
        # Pattern détaillé: inv(3)(q21q26)
        matches = re.findall(CytogeneticsPatterns.INVERSION_POSITIONS, cyto, re.IGNORECASE)
        if matches:
            for chr_num, start_arm, start_band, end_arm, end_band in matches:
                data = {
                    'type': 'inversion',
                    'chromosome': chr_num,
                    'arm': start_arm.lower() if start_arm else None,
                    'start_arm': start_arm.lower() if start_arm else None,
                    'start': start_band,
                }
                if end_band:
                    data['end_arm'] = end_arm.lower() if end_arm else None
                    data['end'] = end_band
                inversions.append(data)
            return inversions
        
        # Pattern simple: inv(3)
        matches = re.findall(CytogeneticsPatterns.INVERSION, cyto, re.IGNORECASE)
        for chr_num in matches:
            inversions.append({'type': 'inversion', 'chromosome': chr_num})
        
        return inversions
    
    def _extract_derivatives(self, cyto: str) -> List[dict]:
        """Extrait les chromosomes dérivés"""
        derivatives = []
        # Pattern détaillé: der(9)t(9;22)(q34;q11)
        matches = re.findall(CytogeneticsPatterns.DERIVATIVE_POSITIONS, cyto, re.IGNORECASE)
        if matches:
            for der_chr, chr1, chr2, break1, break2 in matches:
                derivatives.append({
                    'type': 'derivative',
                    'chromosome': der_chr,
                    'translocation': (chr1, chr2),
                    'breakpoints': (break1, break2)
                })
            return derivatives

        # Pattern simple: der(9)
        matches = re.findall(CytogeneticsPatterns.DERIVATIVE, cyto, re.IGNORECASE)
        for chr_num in matches:
            derivatives.append({'type': 'derivative', 'chromosome': chr_num})

        return derivatives
    
    def _extract_duplications(self, cyto: str) -> List[dict]:
        """Extrait les duplications avec positions détaillées si disponibles"""
        duplications = []
        
        # Pattern détaillé: dup(1)(q21q31)
        matches = re.findall(CytogeneticsPatterns.DUPLICATION_POSITIONS, cyto, re.IGNORECASE)
        if matches:
            for chr_num, start_arm, start_band, end_arm, end_band in matches:
                data = {
                    'type': 'duplication',
                    'chromosome': chr_num,
                    'arm': start_arm.lower() if start_arm else None,
                    'start_arm': start_arm.lower() if start_arm else None,
                    'start': start_band,
                }
                if end_band:
                    data['end_arm'] = end_arm.lower() if end_arm else None
                    data['end'] = end_band
                duplications.append(data)
            return duplications
        
        # Fallback pattern simple: dup(1)
        matches = re.findall(CytogeneticsPatterns.DUPLICATION, cyto, re.IGNORECASE)
        for chr_num in matches:
            duplications.append({'type': 'duplication', 'chromosome': chr_num})
        
        return duplications
    
    def _extract_triplications(self, cyto: str) -> List[dict]:
        """Extrait les triplications avec positions détaillées si disponibles"""
        triplications = []
        
        # Pattern détaillé: trp(1)(q21q31)
        matches = re.findall(CytogeneticsPatterns.TRIPLICATION_POSITIONS, cyto, re.IGNORECASE)
        if matches:
            for chr_num, start_arm, start_band, end_arm, end_band in matches:
                data = {
                    'type': 'triplication',
                    'chromosome': chr_num,
                    'arm': start_arm.lower() if start_arm else None,
                    'start_arm': start_arm.lower() if start_arm else None,
                    'start': start_band,
                }
                if end_band:
                    data['end_arm'] = end_arm.lower() if end_arm else None
                    data['end'] = end_band
                triplications.append(data)
            return triplications
        
        # Fallback pattern simple: trp(1)
        matches = re.findall(CytogeneticsPatterns.TRIPLICATION, cyto, re.IGNORECASE)
        for chr_num in matches:
            triplications.append({'type': 'triplication', 'chromosome': chr_num})
        
        return triplications
    
    def _extract_isochromosomes(self, cyto: str) -> List[dict]:
        """Extrait les isochromosomes avec positions détaillées si disponibles"""
        isochromosomes = []
        
        # Pattern détaillé: i(17)(q10)
        matches = re.findall(CytogeneticsPatterns.ISOCHROMOSOME_POSITIONS, cyto, re.IGNORECASE)
        if matches:
            for chrom, arm, band in matches:
                data = {
                    'type': 'isochromosome',
                    'chromosome': chrom,
                    'arm': arm.lower() if arm else None,
                }
                if band:
                    data['band'] = band
                isochromosomes.append(data)
            return isochromosomes
        
        # Fallback pattern simple: i(17)
        matches = re.findall(CytogeneticsPatterns.ISOCHROMOSOME, cyto, re.IGNORECASE)
        for chr_num in matches:
            isochromosomes.append({'type': 'isochromosome', 'chromosome': chr_num})
        
        return isochromosomes
    
    def _extract_markers(self, cyto: str) -> List[str]:
        """Extrait les chromosomes marqueurs"""
        return re.findall(CytogeneticsPatterns.MARKER, cyto, re.IGNORECASE)