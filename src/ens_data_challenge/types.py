from enum import StrEnum, Enum


class Columns(StrEnum):
    ID = "ID"
    CENTER = "CENTER"
    BM_BLAST = "BM_BLAST"
    WBC = "WBC"
    ANC = "ANC"
    MONOCYTES = "MONOCYTES"
    HB = "HB"
    PLT = "PLT"
    CYTOGENETICS = "CYTOGENETICS"


class CytoColumns(StrEnum):
    IS_NORMAL = "is_normal"
    HAS_TP53_DELETION = "has_tp53_deletion"
    HAS_COMPLEX_CHR3 = "has_complex_chr3"
    N_ABNORMALITIES = "n_abnormalities"
    N_CHROMOSOMES_AFFECTED = "n_chromosomes_affected"
    N_DELETIONS = "n_deletions"
    N_CRITICAL_REGIONS_DELETED = "n_critical_regions_deleted"
    HAS_LARGE_DELETION = "has_large_deletion"
    IS_MOSAIC = "is_mosaic"
    N_CLONES = "n_clones"
    ABNORMAL_CLONE_PERCENTAGE = "abnormal_clone_percentage"
    COMPUTED_RISK_SCORE = "computed_risk_score"
    MDS_IPSS_R_CYTO_RISK = "mds_ipss_r_cyto_risk"
    AML_ELN_2022_CYTO_RISK = "aml_eln_2022_cyto_risk"
    CLL_CYTO_RISK = "cll_cyto_risk"
    MM_RISS_CYTO_RISK = "mm_riss_cyto_risk"


class MolecularColumns(StrEnum):
    GENE = "GENE"
    VARIANT_ALLELE_FREQUENCY = "VAF"
    DEPTH = "DEPTH"
    START = "START"
    END = "END"
    CHR = "CHR"
    REF = "REF"
    ALT = "ALT"
    PROT_CHANGE = "PROTEIN_CHANGE"
    EFFECT = "EFFECT"


class CytoStructColumns(StrEnum):
    ID = "ID"
    CLONE_INDEX = "clone_index"
    CLONE_CELL_COUNT = "clone_cell_count"
    PLOIDY = "ploidy"
    SEX_CHROMOSOMES = "sex_chromosomes"
    MUTATION_TYPE = "mutation_type"
    CHROMOSOME = "chromosome"
    ARM = "arm"
    START = "start"
    END = "end"
    START_ARM = "start_arm"
    END_ARM = "end_arm"
    RAW = "raw"
    MONOSOMY_CHROMOSOME = "monosomy_chromosome"
    MONOSOMIES_COUNT = "monosomies_count"
    TRISOMIES_COUNT = "trisomies_count"
    DELETIONS_COUNT = "deletions_count"
    DELETION_CHROMOSOME = "deletion_chromosome"
    DELETION_ARM = "deletion_arm"
    CELL_COUNT_TOTAL = "cell_count_total"


# Gènes pronostiques établis dans MDS/AML
class HighRiskGenes(StrEnum):
    # ========== TRÈS MAUVAIS PRONOSTIC ==========
    TP53 = "TP53"  # ⚠️ Le pire - résistance thérapeutique
    ASXL1 = "ASXL1"  # Mauvais pronostic établi
    RUNX1 = "RUNX1"  # Mauvais pronostic
    EZH2 = "EZH2"  # Mauvais pronostic
    SETBP1 = "SETBP1"  # Mauvais pronostic (surtout si co-mutation ASXL1)

    # Signaling (RAS pathway) - Mauvais pronostic
    NRAS = "NRAS"  # Mauvais pronostic
    KRAS = "KRAS"  # Mauvais pronostic
    PTPN11 = "PTPN11"  # ⚠️ MANQUAIT - Mauvais pronostic
    CBL = "CBL"  # ⚠️ MANQUAIT - Mauvais/Intermédiaire

    # ========== INTERMÉDIAIRE/CONTEXTE-DÉPENDANT ==========
    # Spliceosome
    SRSF2 = "SRSF2"  # Intermédiaire/mauvais
    U2AF1 = "U2AF1"  # ⚠️ MANQUAIT - Intermédiaire
    ZRSR2 = "ZRSR2"  # ⚠️ MANQUAIT - Intermédiaire

    # Epigénétique
    DNMT3A = "DNMT3A"  # Contexte-dépendant (âge, co-mutations)
    TET2 = "TET2"  # Plutôt favorable/neutre
    IDH1 = "IDH1"  # Intermédiaire (mais cible thérapeutique)
    IDH2 = "IDH2"  # Intermédiaire (mais cible thérapeutique)

    # Tyrosine kinases
    FLT3 = "FLT3"  # ITD = mauvais, TKD = intermédiaire
    KIT = "KIT"  # ⚠️ MANQUAIT - Intermédiaire/mauvais (surtout CBF-AML)
    JAK2 = "JAK2"  # ⚠️ MANQUAIT - Intermédiaire (fréquent en MPN)

    # Transcription factors
    CEBPA = "CEBPA"  # ⚠️ MANQUAIT - Biallélique = BON pronostic
    GATA2 = "GATA2"  # ⚠️ MANQUAIT - Mauvais pronostic

    # Cohesin complex
    STAG2 = "STAG2"  # ⚠️ MANQUAIT - Mauvais pronostic
    RAD21 = "RAD21"  # ⚠️ MANQUAIT - Intermédiaire
    SMC1A = "SMC1A"  # ⚠️ MANQUAIT - Intermédiaire
    SMC3 = "SMC3"  # ⚠️ MANQUAIT - Intermédiaire

    # Autres
    BCOR = "BCOR"  # ⚠️ MANQUAIT - Mauvais/Intermédiaire
    BCORL1 = "BCORL1"  # ⚠️ MANQUAIT - Mauvais pronostic
    PHF6 = "PHF6"  # ⚠️ MANQUAIT - Mauvais pronostic
    WT1 = "WT1"  # ⚠️ MANQUAIT - Mauvais pronostic

    # ========== BON PRONOSTIC ==========
    SF3B1 = "SF3B1"  # Généralement bon (sauf si co-mutations)
    NPM1 = "NPM1"  # Favorable (si sans FLT3-ITD)


class Pathway(StrEnum):
    SIGNALISATION = "Signalisation / Prolifération"
    EPIGENETIQUE = "Épigénétique"
    EPISSAGE = "Épissage"
    TRANSCRIPTION = "Transcription / Différenciation"
    COHESINE = "Complexe de cohésine"
    TUMOR_SUPPRESSORS = "Suppresseurs de tumeurs / Réparation de l'ADN"
    AUTRES = "Autres ou non classés"


PATHWAY_GENES = {
    Pathway.SIGNALISATION: {
        "GNB1",
        "CSF3R",
        "MPL",
        "NRAS",
        "KRAS",
        "FLT3",
        "KIT",
        "JAK2",
        "PTPN11",
        "CBL",
        "SH2B3",
        "NF1",
        "BRAF",
        "GNAS",
        "STAT3",
        "ETNK1",
        "CSNK1A1",
        "PPM1D",
    },
    Pathway.EPIGENETIQUE: {
        "TET2",
        "DNMT3A",
        "IDH1",
        "IDH2",
        "ASXL1",
        "ASXL2",
        "EZH2",
        "KDM6A",
        "JARID2",
        "BCOR",
        "BCORL1",
        "SUZ12",
        "CREBBP",
    },
    Pathway.EPISSAGE: {"SF3B1", "SRSF2", "U2AF1", "U2AF2", "ZRSR2", "PRPF8"},
    Pathway.TRANSCRIPTION: {"RUNX1", "CEBPA", "GATA2", "ETV6", "NFE2", "IRF1", "CUX1"},
    Pathway.COHESINE: {"STAG2", "SMC1A", "SMC3", "RAD21"},
    Pathway.TUMOR_SUPPRESSORS: {"TP53", "CHEK2", "CDKN2A", "BRCC3", "PHF6"},
    Pathway.AUTRES: {
        "NPM1",
        "DDX41",
        "TERT",
        "NOTCH1",
        "MLL",
        "ARID2",
        "WT1",
        "SETBP1",
        "OTHER",
    },
}


class MdsIpssRCytoRisk(Enum):
    VERY_GOOD = 5
    GOOD = 4
    INTERMEDIATE = 3
    POOR = 2
    VERY_POOR = 1
    UNKNOWN = 2.5


class AmlEln2022CytoRisk(Enum):
    FAVORABLE = 3
    INTERMEDIATE = 2
    ADVERSE = 1
    UNKNOWN = 2


class CllCytoRisk(Enum):
    VERY_LOW = 5
    LOW = 4
    INTERMEDIATE = 3
    HIGH = 2
    VERY_HIGH = 1
    UNKNOWN = 2.5


class MmRissCytoRisk(Enum):
    HIGH = 2
    INTERMEDIATE = 1
    UNKNOWN = 1.5
