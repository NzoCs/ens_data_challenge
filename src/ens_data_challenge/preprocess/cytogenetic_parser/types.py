from typing import List, Optional, TypedDict, Union, Tuple
from enum import StrEnum



class CytogeneticsPatterns(StrEnum):
    """Enumération robuste des motifs regex pour le parsing cytogénétique"""

    # --- Structure et base ---
    PLOIDY = r"^(?P<ploidy>\d+)"  # Ex: 46, 47, 45
    SEX_CHROMOSOMES = r",?\s*(?P<sex>XY|XYY|X{1,3})"  # Ex: 46,XY ou 47,XXX
    NORMAL_KARYOTYPE = r"^46,\s*(XX|XY)$"
    MOSAIC_CLONE = r"(.+?)\[(\d+)\]"  # clone[cell_count]

    # --- Anomalies numériques ---
    MONOSOMY = r"-\s*(?P<chrom>\d{1,2}|X|Y)(?:,|$|\[)"
    TRISOMY = r"\+\s*(?P<chrom>\d{1,2}|X|Y)(?:,|$|\[)"

    # --- Additions ---
    ADDITION = r"add\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)"
    ADDITION_POSITIONS = r"add\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)\s*\(\s*(?P<arm>[pq])(?P<band>\d{1,2}(?:\.\d+)?\??)?\s*\)"

    # --- Délétions ---
    DELETION = r"del\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)"
    DELETION_POSITIONS = r"del\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)\s*\(\s*(?P<start_arm>[pq])(?P<start_band>\d{1,2}(?:\.\d+)?\??)\s*(?P<end_arm>[pq])?(?P<end_band>\d{1,2}(?:\.\d+)?\??)?\s*\)"
    DELETION_17P = r"del\(17\)\(p(?P<band>\d{1,2})"

    # --- Duplications ---
    DUPLICATION = r"dup\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)"
    DUPLICATION_POSITIONS = r"dup\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)\s*\(\s*(?P<start_arm>[pq])(?P<start_band>\d{1,2}(?:\.\d+)?\??)\s*(?P<end_arm>[pq])?(?P<end_band>\d{1,2}(?:\.\d+)?\??)?\s*\)"

    # --- Triplications ---
    TRIPLICATION = r"trp\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)"
    TRIPLICATION_POSITIONS = r"trp\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)\s*\(\s*(?P<start_arm>[pq])(?P<start_band>\d{1,2}(?:\.\d+)?\??)\s*(?P<end_arm>[pq])?(?P<end_band>\d{1,2}(?:\.\d+)?\??)?\s*\)"

    # --- Inversions ---
    INVERSION = r"inv\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)"
    INVERSION_POSITIONS = r"inv\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)\s*\(\s*(?P<start_arm>[pq])(?P<start_band>\d{1,2}(?:\.\d+)?\??)\s*(?P<end_arm>[pq])?(?P<end_band>\d{1,2}(?:\.\d+)?\??)?\s*\)"

    # --- Isochromosomes ---
    ISOCHROMOSOME = r"i\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)"
    ISOCHROMOSOME_POSITIONS = r"i\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)\s*\(\s*(?P<arm>[pq])(?P<band>\d{1,2}(?:\.\d+)?\??)?\s*\)"

    # --- Translocations ---
    TRANSLOCATION = r"t\(\s*(?P<chrom1>\d{1,2}|X|Y)\s*;\s*(?P<chrom2>\d{1,2}|X|Y)\s*\)"
    TRANSLOCATION_POSITIONS = r"t\(\s*(?P<chrom1>\d{1,2}|X|Y)\s*;\s*(?P<chrom2>\d{1,2}|X|Y)\s*\)\s*\(\s*(?P<arm1>[pq])(?P<band1>\d{1,2}(?:\.\d+)?\??)\s*;\s*(?P<arm2>[pq])(?P<band2>\d{1,2}(?:\.\d+)?\??)\s*\)"

    # --- Insertions ---
    INSERTION = r"ins\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)"
    INSERTION_POSITIONS = r"ins\(\s*(?P<src_chrom>\d{1,2}|X|Y)\s*;\s*(?P<dest_chrom>\d{1,2}|X|Y)\s*\)\s*\(\s*(?P<arm1>[pq])(?P<band1>\d{1,2}(?:\.\d+)?\??)\s*;\s*(?P<arm2>[pq])(?P<band2>\d{1,2}(?:\.\d+)?\??)\s*\)"

    # --- Marker chromosomes ---
    MARKER = r"mar(?P<index>\d+)?"

    # --- Ring chromosomes ---
    RING = r"r\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)"
    RING_POSITIONS = r"r\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)\s*\(\s*(?P<arm>[pq])(?P<band>\d{1,2}(?:\.\d+)?\??)?\s*\)"

    # --- Derivative chromosomes ---
    DERIVATIVE = r"der\(\s*(?P<chrom>\d{1,2}|X|Y)\s*\)"
    DERIVATIVE_POSITIONS = r"der\(\s*(?P<der_chr>\d{1,2}|X|Y)\s*\)t\(\s*(?P<chr1>\d{1,2}|X|Y)\s*;\s*(?P<chr2>\d{1,2}|X|Y)\s*\)\s*\(\s*(?P<break1>[pq]\d{1,2}(?:\.\d+)?\??)\s*;\s*(?P<break2>[pq]\d{1,2}(?:\.\d+)?\??)\s*\)"

    # --- Dicentric chromosomes ---
    DICENTRIC = r"dic\(\s*(?P<chrom1>\d{1,2}|X|Y)\s*;\s*(?P<chrom2>\d{1,2}|X|Y)\s*\)"

    # --- Anomalies spécifiques (pronostiques) ---
    COMPLEX_CHR3 = r"(?:inv\(3\)|t\(3[;q])"
    T_8_21 = r"t\(8;21\)"
    T_15_17 = r"t\(15;17\)"
    INV_16 = r"inv\(16\)"
    T_9_22 = r"t\(9;22\)"  # Philadelphia chromosome

    # --- Caryotypes complexes ---
    MULTIPLE_ABNORMALITIES = r"[,/].+[,/].+[,/]"
    HYPODIPLOIDY = r"^[2-4]\d,"
    HYPERDIPLOIDY = r"^4[8-9],"

class RiskCategory(StrEnum):
    """Catégories de risque cytogénétique selon IPSS-R"""
    VERY_GOOD = 'Very Good'
    GOOD = 'Good'
    INTERMEDIATE = 'Intermediate'
    POOR = 'Poor'
    VERY_POOR = 'Very Poor'
    UNKNOWN = 'Unknown'


class MutationInfo(TypedDict):
    """Information détaillée sur une anomalie structurelle"""
    chromosome: Union[str, Tuple[str, str]]
    type: str  # 'deletion', 'addition', 'translocation', etc.
    arm: Optional[str]  # 'p' ou 'q'
    start_position: Optional[str]
    end_position: Optional[str]


# Types pour le parsing (données brutes)
class ParsedKaryotypeDict(TypedDict, total=False):
    """Représentation typée d'un karyotype parsé (données brutes uniquement)"""
    is_normal: bool
    ploidy: Optional[int]
    sex_chromosomes: Optional[str]
    deletions: List[MutationInfo]
    additions: List[MutationInfo]
    translocations: List[MutationInfo]
    inversions: List[MutationInfo]
    derivatives: List[MutationInfo]
    duplications: List[MutationInfo]
    triplications: List[MutationInfo]
    isochromosomes: List[MutationInfo]
    markers: List[str]
    monosomies: List[str]
    trisomies: List[str]
    cell_count: Optional[int]


# Types pour les features (données calculées)
class CytogeneticsFeatures(TypedDict):
    """Features prédictives calculées à partir du parsing"""
    is_normal: Optional[int]
    ploidy: Optional[int]
    has_tp53_deletion: Optional[int]
    has_complex_chr3: Optional[int]
    n_abnormalities: Optional[int]
    n_chromosomes_affected: Optional[int]
    has_monosomy_7: Optional[int]
    has_del_5q: Optional[int]
    has_del_7q: Optional[int]
    has_monosomy_y: Optional[int]
    n_deletions: Optional[int]
    n_critical_regions_deleted: Optional[int]
    has_large_deletion: Optional[int]
    is_mosaic: Optional[int]
    risk_category: str  # RiskCategory as string


class MosaicFeatures(CytogeneticsFeatures):
    """Features pour caryotypes mosaïques (extend base features)"""
    n_clones: int
    abnormal_clone_percentage: float