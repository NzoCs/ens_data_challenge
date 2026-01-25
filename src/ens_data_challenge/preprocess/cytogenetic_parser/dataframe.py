import logging
import re
import math  # Importation de math au lieu de numpy
import pandas as pd
from typing import List, Optional, TypedDict, cast

from .parser import ParsedKaryotype, CytogeneticsParser
from ens_data_challenge.types import (
    MdsIpssRCytoRisk,
    AmlEln2022CytoRisk,
    CllCytoRisk,
    MmRissCytoRisk,
)

# =================================================================
# FEATURE ENGINEERING - Séparé du parsing
# =================================================================


class CytogeneticFeatures(TypedDict):
    """Dictionnaire typé pour les features cytogénétiques extraites."""

    is_normal: bool
    ploidy: int
    has_tp53_deletion: bool
    has_complex_chr3: bool
    n_abnormalities: int
    n_chromosomes_affected: int
    n_deletions: int
    n_critical_regions_deleted: int
    has_large_deletion: bool
    is_mosaic: bool
    n_clones: int
    abnormal_clone_percentage: float
    computed_risk_score: float
    mds_ipss_r_cyto_risk: MdsIpssRCytoRisk
    aml_eln_2022_cyto_risk: AmlEln2022CytoRisk
    cll_cyto_risk: CllCytoRisk
    mm_riss_cyto_risk: MmRissCytoRisk


class CytogeneticsExtractor:
    """Extrait des features prédictives à partir d'un karyotype parsé"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_features(
        self, parsed_list: Optional[List[ParsedKaryotype]]
    ) -> CytogeneticFeatures:
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

    def _empty_features(self) -> CytogeneticFeatures:
        """Features pour données manquantes"""
        return CytogeneticFeatures(
            is_normal=True,
            ploidy=0,
            has_tp53_deletion=False,
            has_complex_chr3=False,
            n_abnormalities=0,
            n_chromosomes_affected=0,
            n_deletions=0,
            n_critical_regions_deleted=0,
            has_large_deletion=False,
            is_mosaic=False,
            n_clones=0,
            abnormal_clone_percentage=0.0,
            computed_risk_score=0.0,
            mds_ipss_r_cyto_risk=MdsIpssRCytoRisk.UNKNOWN,
            aml_eln_2022_cyto_risk=AmlEln2022CytoRisk.UNKNOWN,
            cll_cyto_risk=CllCytoRisk.UNKNOWN,
            mm_riss_cyto_risk=MmRissCytoRisk.UNKNOWN,
        )

    def _normal_features(self, parsed: ParsedKaryotype) -> CytogeneticFeatures:
        """Features pour caryotype normal"""
        return CytogeneticFeatures(
            is_normal=True,
            ploidy=parsed.ploidy,
            has_tp53_deletion=False,
            has_complex_chr3=False,
            n_abnormalities=0,
            n_chromosomes_affected=0,
            n_deletions=0,
            n_critical_regions_deleted=0,
            has_large_deletion=False,
            is_mosaic=False,
            n_clones=1,
            abnormal_clone_percentage=0.0,
            # Nouveaux scores de risque (catégories favorables)
            computed_risk_score=0.0,
            mds_ipss_r_cyto_risk=MdsIpssRCytoRisk.GOOD,
            aml_eln_2022_cyto_risk=AmlEln2022CytoRisk.INTERMEDIATE,  # Normal est intermédiaire pour ELN
            cll_cyto_risk=CllCytoRisk.VERY_LOW,
            mm_riss_cyto_risk=MmRissCytoRisk.INTERMEDIATE,  # Standard -> intermediate
        )

    def _single_clone_features(self, parsed: ParsedKaryotype) -> CytogeneticFeatures:
        """Features pour un clone unique"""

        # Calcul des features de base
        feats = CytogeneticFeatures(
            is_normal=False,
            ploidy=parsed.ploidy,
            has_tp53_deletion=self._has_tp53_deletion(parsed),
            has_complex_chr3=self._has_complex_chr3(parsed),
            n_abnormalities=self._count_abnormalities(parsed),
            n_chromosomes_affected=self._count_affected_chromosomes(parsed),
            n_deletions=len(parsed.deletions),
            n_critical_regions_deleted=self._count_critical_deletions(parsed),
            has_large_deletion=self._has_large_deletion(parsed),
            is_mosaic=False,
            n_clones=1,
            abnormal_clone_percentage=100.0,
            # Nouveaux scores de risque
            computed_risk_score=0.0,
            mds_ipss_r_cyto_risk=MdsIpssRCytoRisk.UNKNOWN,
            aml_eln_2022_cyto_risk=AmlEln2022CytoRisk.UNKNOWN,
            cll_cyto_risk=CllCytoRisk.UNKNOWN,
            mm_riss_cyto_risk=MmRissCytoRisk.UNKNOWN,
        )

        # Calcul des scores de risque basés sur les features
        feats["mds_ipss_r_cyto_risk"] = self._get_mds_ipss_r_cytogenetic_risk(
            parsed, feats
        )
        feats["aml_eln_2022_cyto_risk"] = self._get_aml_eln_2022_cytogenetic_risk(
            parsed, feats
        )
        feats["cll_cyto_risk"] = self._get_cll_hierarchical_risk(parsed, feats)
        feats["mm_riss_cyto_risk"] = self._get_mm_riss_cytogenetic_risk(parsed, feats)

        # Calcul du score numérique personnalisé
        feats["computed_risk_score"] = self._compute_risk_score(parsed, feats)

        return feats

    def _mosaic_features(self, clones: List[ParsedKaryotype]) -> CytogeneticFeatures:
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

        # ploidy reported from the most abnormal or first clone
        ploidy_val = (
            getattr(most_abnormal, "ploidy", None)
            if most_abnormal
            else (clones[0].ploidy if clones else None)
        )
        features["ploidy"] = ploidy_val if ploidy_val is not None else 0

        features["is_mosaic"] = True
        features["n_clones"] = len(clones)

        abnormal_cell_count = most_abnormal.cell_count or 0
        if most_abnormal.is_normal:
            # Si le "plus anormal" est en fait normal (ex: 46,XX[10]/45,X,-Y[2])
            # On cherche le premier clone VRAIMENT anormal pour le comptage
            abnormal_clone = next((c for c in clones if not c.is_normal), None)
            if abnormal_clone:
                abnormal_cell_count = abnormal_clone.cell_count or 0
            else:
                abnormal_cell_count = 0  # Cas ex: 46,XX / 46,XY

        features["abnormal_clone_percentage"] = (
            (abnormal_cell_count / total_cells * 100) if total_cells > 0 else 0
        )

        return features

    # === Feature calculation methods ===

    def _has_tp53_deletion(self, parsed: ParsedKaryotype) -> bool:
        """Détecte délétion TP53 (17p13)"""
        for del_info in parsed.deletions:
            if (
                del_info.get("chromosome") == "17"
                and del_info.get("arm") == "p"
                and del_info.get("start")
            ):
                # TP53 est en 17p13.1, donc toute délétion commençant
                # à p13 ou avant (p12, p11...) l'inclut
                pos = int(re.sub(r"\..*", "", del_info["start"]))  # '13.1' -> '13'
                if pos <= 13:
                    return True
        return False

    def _has_complex_chr3(self, parsed: ParsedKaryotype) -> bool:
        """Détecte anomalies complexes chromosome 3 (inv(3) ou t(3q))"""
        # Inversions on chr3
        for inv in parsed.inversions:
            if inv.get("chromosome") == "3":
                return True

        # Translocations: support new structure ('chromosomes' tuple + 'breakpoints' tuples)
        for trans in parsed.translocations:
            # extract chromosome list
            chrs = []
            if trans.get("chromosomes"):
                val = trans.get("chromosomes")
                if val:
                    chrs = [str(c) for c in val]
            elif trans.get("chromosome"):
                chrs = re.findall(r"(\d+|x|y)", str(trans.get("chromosome")).lower())

            # if 3 involved, check corresponding breakpoint on q-arm
            if "3" in chrs:
                # new structured breakpoints
                bps = trans.get("breakpoints")
                if bps and isinstance(bps, (list, tuple)):
                    for i, c in enumerate(chrs):
                        if c == "3" and i < len(bps):
                            bp = bps[i]
                            # bp can be tuple (arm, band) or string
                            if isinstance(bp, (list, tuple)) and len(bp) >= 1:
                                arm = bp[0]
                            else:
                                arm = str(bp)[0] if bp else ""
                            if str(arm).lower().startswith("q"):
                                return True

                # fallback to legacy 'breakpoint' string
                bp_str = trans.get("breakpoint")
                if bp_str:
                    parts = str(bp_str).split(";")
                    for i, c in enumerate(chrs):
                        if c == "3" and i < len(parts) and parts[i].startswith("q"):
                            return True

        return False

    def _has_del_5q(self, parsed: ParsedKaryotype) -> bool:
        """Détecte délétion 5q"""
        return any(
            d.get("chromosome") == "5" and d.get("arm") == "q" for d in parsed.deletions
        )

    def _has_del_7q(self, parsed: ParsedKaryotype) -> bool:
        """Détecte délétion 7q"""
        return any(
            d.get("chromosome") == "7" and d.get("arm") == "q" for d in parsed.deletions
        )

    def _has_del_11q(self, parsed: ParsedKaryotype) -> bool:
        """Détecte délétion 11q"""
        return any(
            d.get("chromosome") == "11" and d.get("arm") == "q"
            for d in parsed.deletions
        )

    def _has_del_13q(self, parsed: ParsedKaryotype) -> bool:
        """Détecte délétion 13q"""
        return any(
            d.get("chromosome") == "13" and d.get("arm") == "q"
            for d in parsed.deletions
        )

    def _has_trisomy(self, parsed: ParsedKaryotype, chromosomes: List[str]) -> bool:
        """Détecte si une trisomie est présente dans la liste"""
        return any(tri in chromosomes for tri in parsed.trisomies)

    def _has_monosomy(self, parsed: ParsedKaryotype, chromosome: str) -> bool:
        """Détecte si une monosomie pour un chromosome donné est présente"""
        return str(chromosome) in set(str(m) for m in parsed.monosomies)

    def _has_translocation(
        self, parsed: ParsedKaryotype, chr_pairs: List[tuple[str, str]]
    ) -> bool:
        """Détecte une translocation dans une liste de paires"""
        for trans in parsed.translocations:
            # Support new structured 'chromosomes' tuple
            chrs = []
            if trans.get("chromosomes"):
                val = trans.get("chromosomes")
                if val:
                    chrs = [str(c) for c in val]
            elif trans.get("chromosome"):
                chrs = re.findall(r"(\d+|x|y)", str(trans.get("chromosome")).lower())
            elif trans.get("translocation"):
                # derivative might store translocation as tuple
                val = trans.get("translocation")
                if val:
                    chrs = [str(c) for c in val]

            if len(chrs) < 2:
                continue

            chr_set = set(chrs)
            for pair in chr_pairs:
                if pair[0] in chr_set and pair[1] in chr_set:
                    return True
        return False

    def _has_large_deletion(self, parsed: ParsedKaryotype) -> bool:
        """Détecte délétions larges (>20 bandes)"""
        for del_info in parsed.deletions:
            start = del_info.get("start")
            end = del_info.get("end")
            if start and end:
                try:
                    # Gérer 'p11' vs 'q22' (pas comparable)
                    if del_info.get("arm"):
                        # Simplification : si start/end sont sur le même bras
                        size = int(re.sub(r"\..*", "", end)) - int(
                            re.sub(r"\..*", "", start)
                        )
                        if size > 20:
                            return True
                    # Si 'start' est 'p' et 'end' est 'q', c'est une large délétion
                    elif (
                        del_info.get("start_arm") == "p"
                        and del_info.get("end_arm") == "q"
                    ):
                        return True
                except ValueError:
                    continue  # Ignorer si les positions ne sont pas numériques
        return False

    def _count_abnormalities(self, parsed: ParsedKaryotype) -> int:
        """Compte le nombre total d'anomalies"""
        return (
            len(parsed.deletions)
            + len(parsed.additions)
            + len(parsed.translocations)
            + len(parsed.inversions)
            + len(parsed.derivatives)
            + len(parsed.monosomies)
            + len(parsed.trisomies)
        )

    def _count_affected_chromosomes(self, parsed: ParsedKaryotype) -> int:
        """Compte les chromosomes uniques affectés"""
        affected = set()

        for anomaly_list in [
            parsed.deletions,
            parsed.additions,
            parsed.translocations,
            parsed.inversions,
            parsed.derivatives,
        ]:
            for anom in anomaly_list:
                chr_field = anom.get("chromosome", "")
                chr_matches = re.findall(r"(\d+|x|y)", str(chr_field).lower())
                affected.update(chr_matches)

        affected.update(parsed.monosomies)
        affected.update(parsed.trisomies)

        return len(affected)

    def _count_critical_deletions(self, parsed: ParsedKaryotype) -> int:
        """Compte les délétions dans les régions critiques"""
        critical_regions = {
            "5": (31, 33),  # 5q31-33
            "7": (22, 36),  # 7q22-36
            "17": (13, 13),  # 17p13 (TP53)
        }

        count = 0
        for del_info in parsed.deletions:
            chr_num = del_info.get("chromosome", "")
            arm = del_info.get("arm", "")

            if chr_num in critical_regions and arm in ("p", "q"):
                crit_start, crit_end = critical_regions[chr_num]
                start = del_info.get("start")
                end = del_info.get("end")

                # Gérer les délétions de bras entier (ex: del(5q))
                if not start:
                    count += 1
                    continue

                if start and end:
                    try:
                        del_start = int(re.sub(r"\..*", "", start))
                        del_end = int(re.sub(r"\..*", "", end))

                        # Chevauchement avec région critique
                        if not (del_end < crit_start or del_start > crit_end):
                            count += 1
                    except ValueError:
                        continue
                elif start:  # Délétion terminale (ex: del(5)(q31))
                    try:
                        del_start = int(re.sub(r"\..*", "", start))
                        # Si la délétion commence dans ou avant la région critique
                        if del_start <= crit_end:
                            count += 1
                    except ValueError:
                        continue
        return count

    # === NOUVELLES FONCTIONS DE SCORING ===

    def _get_mds_ipss_r_cytogenetic_risk(
        self, parsed: ParsedKaryotype, feats: CytogeneticFeatures
    ) -> MdsIpssRCytoRisk:
        """Classification cytogénétique IPSS-R (MDS) - standardisée"""
        n_abn = feats.get("n_abnormalities")

        if n_abn is None:
            return MdsIpssRCytoRisk.UNKNOWN

        # Very Poor: TP53 deletion or many abnormalities
        if feats.get("has_tp53_deletion") or (n_abn is not None and n_abn > 3):
            return MdsIpssRCytoRisk.VERY_POOR

        # Poor: monosomy 7 / del(7q) / complex chr3 or exactly 3 abnormalities
        if (
            self._has_monosomy(parsed, "7")
            or self._has_del_7q(parsed)
            or feats.get("has_complex_chr3")
            or n_abn == 3
        ):
            return MdsIpssRCytoRisk.POOR

        # Good: no abnormalities or single del(5q) or double including del(5q)
        if n_abn == 0:
            return MdsIpssRCytoRisk.GOOD
        if n_abn == 1 and self._has_del_5q(parsed):
            return MdsIpssRCytoRisk.GOOD
        if n_abn == 2 and self._has_del_5q(parsed):
            return MdsIpssRCytoRisk.GOOD

        # Very Good: specific low-risk single events (11q or -Y)
        if n_abn == 1 and (
            self._has_monosomy(parsed, "y") or self._has_del_11q(parsed)
        ):
            return MdsIpssRCytoRisk.VERY_GOOD

        return MdsIpssRCytoRisk.INTERMEDIATE

    def _get_aml_eln_2022_cytogenetic_risk(
        self, parsed: ParsedKaryotype, feats: CytogeneticFeatures
    ) -> AmlEln2022CytoRisk:
        """Classification cytogénétique ELN 2022 (AML) - standardisée"""
        n_abn = feats["n_abnormalities"]

        # Very Low (Favorable)
        if self._has_translocation(parsed, [("8", "21"), ("16", "16"), ("15", "17")]):
            return AmlEln2022CytoRisk.FAVORABLE
        if any(inv.get("chromosome") == "16" for inv in parsed.inversions):
            return AmlEln2022CytoRisk.FAVORABLE

        # Very High (Adverse)
        if self._has_translocation(parsed, [("3", "3"), ("6", "9"), ("9", "22")]):
            return AmlEln2022CytoRisk.ADVERSE
        if any(inv.get("chromosome") == "3" for inv in parsed.inversions):
            return AmlEln2022CytoRisk.ADVERSE
        if "5" in parsed.monosomies or self._has_del_5q(parsed):
            return AmlEln2022CytoRisk.ADVERSE
        if "7" in parsed.monosomies:  # -7 (pas del(7q) pour ELN)
            return AmlEln2022CytoRisk.ADVERSE
        if "17" in parsed.monosomies or feats["has_tp53_deletion"]:
            return AmlEln2022CytoRisk.ADVERSE
        if n_abn is not None and n_abn >= 3:  # Complex karyotype
            return AmlEln2022CytoRisk.ADVERSE

        # Intermediate
        return AmlEln2022CytoRisk.INTERMEDIATE

    def _get_cll_hierarchical_risk(
        self, parsed: ParsedKaryotype, feats: CytogeneticFeatures
    ) -> CllCytoRisk:
        """Classification cytogénétique hiérarchique (CLL) - standardisée"""

        if feats["has_tp53_deletion"]:
            return CllCytoRisk.VERY_HIGH

        if self._has_del_11q(parsed):
            return CllCytoRisk.HIGH

        if "12" in parsed.trisomies:
            return CllCytoRisk.INTERMEDIATE

        n_abn = feats["n_abnormalities"]
        if self._has_del_13q(parsed) and n_abn == 1:
            return CllCytoRisk.LOW

        if n_abn == 0:
            return CllCytoRisk.VERY_LOW

        return CllCytoRisk.INTERMEDIATE  # Fallback pour autres anomalies

    def _get_mm_riss_cytogenetic_risk(
        self, parsed: ParsedKaryotype, feats: CytogeneticFeatures
    ) -> MmRissCytoRisk:
        """Classification cytogénétique R-ISS (Multiple Myeloma) - standardisée"""

        # High
        if feats["has_tp53_deletion"]:
            return MmRissCytoRisk.HIGH
        if self._has_translocation(parsed, [("4", "14"), ("14", "16")]):
            return MmRissCytoRisk.HIGH

        # Intermediate (Standard)
        return MmRissCytoRisk.INTERMEDIATE

    # === Fin des nouvelles fonctions ===

    def _get_most_abnormal_clone(
        self, clones: List[ParsedKaryotype]
    ) -> ParsedKaryotype:
        """Identifie le clone le plus anormal dans un mosaïque"""

        def abnormality_score(clone: ParsedKaryotype) -> float:
            if clone.is_normal:
                return -1.0  # Le clone normal a toujours le score le plus bas

            score = 0.0
            score += self._has_tp53_deletion(clone) * 10
            score += (1 if "7" in clone.monosomies else 0) * 10
            score += self._has_complex_chr3(clone) * 8
            score += (1 if "5" in clone.monosomies else 0) * 7
            score += self._has_del_7q(clone) * 6
            score += self._has_del_5q(clone) * 5
            score += self._count_critical_deletions(clone) * 4
            score += self._has_large_deletion(clone) * 3
            score += self._count_abnormalities(clone) * 2
            # Gérer -Y (très bon pronostic) comme étant "moins anormal"
            if "y" in clone.monosomies and self._count_abnormalities(clone) == 1:
                return 0.1
            return score

        # Retourne le clone avec le score max, ou None si tous sont normaux
        max_clone = max(clones, key=abnormality_score)
        return max_clone

    def _compute_risk_score(
        self, parsed: ParsedKaryotype, feats: CytogeneticFeatures
    ) -> float:
        """
        Calcule un score de risque numérique basé sur les anomalies détectées.
        Score normalisé entre 0 (bon pronostic) et 1 (très mauvais pronostic).
        """
        # Poids manuels — évaluer depuis `parsed` pour éviter dépendance aux flags exposés
        raw_score = 0.0
        # TP53
        if feats.get("has_tp53_deletion"):
            raw_score += 3.0
        # complex chr3
        if feats.get("has_complex_chr3"):
            raw_score += 2.0
        # del(5q) / del(7q)
        if self._has_del_5q(parsed):
            raw_score += 1.0
        if self._has_del_7q(parsed):
            raw_score += 1.5
        # monosomy 7
        if self._has_monosomy(parsed, "7"):
            raw_score += 2.5
        # n_abnormalities / n_critical
        if feats["n_abnormalities"] is not None:
            raw_score += 0.3 * float(feats["n_abnormalities"])
        if feats["n_critical_regions_deleted"] is not None:
            raw_score += 1.0 * float(feats["n_critical_regions_deleted"])
        # large deletion
        if feats["has_large_deletion"]:
            raw_score += 1.0

        # Pénalité pour -Y (bon pronostic)
        if self._has_monosomy(parsed, "y") and feats["n_abnormalities"] == 1:
            raw_score = -2.0  # Force un score bas

        # normalisation (sigmoïde)
        normalized = 1 / (
            1 + math.exp(-0.4 * (raw_score - 4))
        )  # Utilisation de math.exp
        return round(float(normalized), 3)

    def gen_structured_dataframe(
        self,
        clinical_data: pd.DataFrame,
        cyto_col: str = "CYTOGENETICS",
    ) -> pd.DataFrame:
        """Génère un DataFrame structuré à partir des données cytogénétiques brutes qui résume chaque anomalie
        Args:
            df: DataFrame d'entrée qui est extrais du dataset clinique
            cyto_col: Nom de la colonne contenant les karyotypes bruts
        Returns:
            DataFrame structuré avec une ligne par anomalie cytogénétique
        """

        parser = CytogeneticsParser()
        rows: list[dict] = []

        # colonnes fixes pour toutes les mutations
        base_cols = [
            "ID",
            "ploidy",
            "sex_chromosomes",
            "clone_index",
            "clone_cell_count",
            "mutation_type",
            "chromosome",
            "arm",
            "start",
            "end",
            "start_arm",
            "end_arm",
            "raw",
        ]

        for idx, row in clinical_data.iterrows():
            patient_id = row["ID"]
            cyto = row[cyto_col]
            parsed_list = parser.parse(cyto)
            if not parsed_list:
                continue

            ploidy = parsed_list[0].ploidy
            sex_chrom = parsed_list[0].sex_chromosomes

            for clone_i, clone in enumerate(parsed_list):
                base = {
                    "ID": patient_id,
                    "ploidy": ploidy,
                    "sex_chromosomes": sex_chrom,
                    "clone_index": clone_i,
                    "clone_cell_count": clone.cell_count or 0,
                }

                def make_row(extra: dict, mut_type: str):
                    row = base.copy()
                    for col in base_cols:
                        if col not in row:
                            row[col] = None
                    row.update(extra)
                    row["mutation_type"] = mut_type
                    return row

                # deletions
                for d in clone.deletions:
                    rows.append(
                        make_row(
                            {
                                "chromosome": d.get("chromosome"),
                                "arm": d.get("arm"),
                                "start": d.get("start"),
                                "end": d.get("end"),
                                "start_arm": d.get("start_arm"),
                                "end_arm": d.get("end_arm"),
                                "raw": d,
                            },
                            "deletion",
                        )
                    )

                # additions
                for a in clone.additions:
                    rows.append(
                        make_row(
                            {
                                "chromosome": a.get("chromosome"),
                                "arm": a.get("arm"),
                                "start": a.get("start"),
                                "end": a.get("end"),
                                "start_arm": a.get("start_arm"),
                                "end_arm": a.get("end_arm"),
                                "raw": a,
                            },
                            "addition",
                        )
                    )

                # inversions
                for inv in clone.inversions:
                    rows.append(
                        make_row(
                            {
                                "chromosome": inv.get("chromosome"),
                                "arm": inv.get("arm"),
                                "start": inv.get("start"),
                                "end": inv.get("end"),
                                "start_arm": inv.get("start_arm"),
                                "end_arm": inv.get("end_arm"),
                                "raw": inv,
                            },
                            "inversion",
                        )
                    )

                # derivatives
                for der in clone.derivatives:
                    rows.append(
                        make_row(
                            {
                                "chromosome": der.get("chromosome")
                                or der.get("derivative"),
                                "raw": der,
                            },
                            "derivative",
                        )
                    )

                # triplications / duplications / isochromosomes / markers
                for dup in clone.duplications:
                    rows.append(
                        make_row(
                            {
                                "chromosome": dup.get("chromosome"),
                                "arm": dup.get("arm"),
                                "start": dup.get("start"),
                                "end": dup.get("end"),
                                "start_arm": dup.get("start_arm"),
                                "end_arm": dup.get("end_arm"),
                                "raw": dup,
                            },
                            "duplication",
                        )
                    )

                for _ in range(len(clone.triplications)):
                    rows.append(make_row({"raw": None}, "triplication"))

                for iso in clone.isochromosomes:
                    rows.append(
                        make_row(
                            {
                                "chromosome": iso.get("chromosome"),
                                "arm": iso.get("arm"),
                                "start_arm": iso.get("start_arm"),
                                "end_arm": iso.get("end_arm"),
                                "raw": iso,
                            },
                            "isochromosome",
                        )
                    )

                for m in clone.markers:
                    rows.append(make_row({"raw": m}, "marker"))

                # trisomies
                for tri in clone.trisomies:
                    rows.append(make_row({"chromosome": tri, "raw": tri}, "trisomy"))

                # monosomies
                for mono in clone.monosomies:
                    rows.append(make_row({"chromosome": mono, "raw": mono}, "monosomy"))

                # translocations → on stocke chr1, chr2 dans chromosome séparé par une virgule, breakpoints dans arm/start/end optionnel
                for t in clone.translocations:
                    chr1 = chr2 = None
                    start_arm = end_arm = None
                    if t.get("chromosomes"):
                        chrs = t.get("chromosomes")
                        if isinstance(chrs, (list, tuple)) and len(chrs) >= 2:
                            chr1, chr2 = chrs[0], chrs[1]
                    elif t.get("chromosome"):
                        chrs = re.findall(
                            r"(\d+|x|y)", str(t.get("chromosome")).lower()
                        )
                        if len(chrs) >= 2:
                            chr1, chr2 = chrs[0], chrs[1]

                    # Extract arms from breakpoints if available
                    bps = t.get("breakpoints")
                    if bps and isinstance(bps, (list, tuple)) and len(bps) >= 2:
                        if isinstance(bps[0], (list, tuple)) and len(bps[0]) >= 1:
                            start_arm = bps[0][0]
                        if isinstance(bps[1], (list, tuple)) and len(bps[1]) >= 1:
                            end_arm = bps[1][0]

                    rows.append(
                        make_row(
                            {
                                "chromosome": f"{chr1},{chr2}"
                                if chr1 and chr2
                                else None,
                                "arm": None,
                                "start": None,
                                "end": None,
                                "start_arm": start_arm,
                                "end_arm": end_arm,
                                "raw": t,
                            },
                            "translocation",
                        )
                    )

        struct_df = pd.DataFrame(rows)
        struct_df = struct_df[base_cols]  # assure l'ordre des colonnes
        return struct_df

    def gen_features_to_dataframe(
        self,
        df: pd.DataFrame,
        cyto_col: str = "CYTOGENETICS",
    ) -> pd.DataFrame:
        """Generate cytogenetics features from a DataFrame containing ISCN karyotypes. Ne retourne que les features.

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
            features_rows.append(cast(dict, feats))

        # Create features dataframe; ensure consistent columns by using _empty_features keys
        empty = cast(dict, self._empty_features())
        feat_cols = list(empty.keys())

        # Convert Enums to strings explicitly
        final_rows = []
        for row in features_rows:
            new_row = {}
            for k in feat_cols:
                val = row.get(k, empty[k])
                # Convert Enums to string values
                if hasattr(val, "value"):
                    new_row[k] = val.value
                else:
                    new_row[k] = val
            final_rows.append(new_row)

        features_df = pd.DataFrame(final_rows)

        # Return only the features dataframe
        return features_df
