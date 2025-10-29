import numpy as np
from typing import Literal, Optional

class GeneEnhancedEncoder:

    def __init__(
            self, 
            method: Literal['confidence_weighted', 'bayesian', 'vaf_score', 'log_vaf', 'depth_score', 'constant'], 
            apply_effect_weighting: bool, 
            vaf_mean: float, 
            depth_mean: float
            ) -> None:

        self.method = method
        self.vaf_mean = vaf_mean
        self.depth_mean = depth_mean
        self.apply_effect_weighting = apply_effect_weighting

    def compute(
            self, 
            vaf: float,
            depth: float,
            effect: Optional[str] = None
            ) -> float:
        """
        Calcule le VAF pondéré selon la méthode choisie
        
        Args:
            method: Méthode de pondération ('confidence_weighted', 'bayesian', 'vaf_score', 'log_vaf')
        
        Returns:
            VAF pondéré (float)
        """
        if self.method == 'confidence_weighted':
            result = self._confidence_weighted_vaf(vaf, depth, self.depth_mean)
        elif self.method == 'bayesian':
            result = self._bayesian_vaf_score(vaf, depth, self.vaf_mean, self.depth_mean)
        elif self.method == 'vaf_score':
            result = self._vaf_score(vaf)
        elif self.method == 'log_vaf':
            result = self.log_vaf(vaf, depth)
        elif self.method == 'depth_score':
            result = self.depth_score(depth)
        elif self.method == 'constant':
            result = 1.0
        else:
            raise ValueError(f"Méthode inconnue: {self.method}")
        
        if self.apply_effect_weighting and effect is not None:
            effect_weight = self.effect_weighting(effect)
            result *= effect_weight

        return result
        
        
    @staticmethod
    def log_vaf(vaf: float, depth: float) -> float:

        """
        Calcule le log du VAF pour atténuer l'impact des valeurs élevées
        """
        return np.log(vaf * depth + 1e-6)
    
    @staticmethod
    def _confidence_weighted_vaf(vaf: float, depth: float, depth_mean: float) -> float:
        """
        Calcule un VAF pondéré par la confiance (sigmoid)
        
        Returns:
            VAF ajusté par confiance (0 à vaf)
        """

        confidence = 1/2 + 1 / (1 + np.exp(-0.05 * (depth - depth_mean)))
        return vaf * confidence

    @staticmethod
    def _bayesian_vaf_score(vaf: float, depth: float, prior_vaf_mean: float, prior_vaf_strength: float) -> float:
        """
        Estimateur bayésien du VAF
        
        Args:
            prior_vaf_mean: VAF a priori (0.5 = distribution neutre)
            prior_vaf_strength: Force du prior (équivalent à N lectures fictives)
        
        Returns:
            VAF ajusté (shrinkage vers le prior si depth faible)
        """
        # Posterior mean avec conjugate prior (Beta-Binomial)
        variant_reads = depth * vaf
        
        # Posterior = (prior + data) / (prior_strength + depth)
        posterior_mean = (prior_vaf_strength * prior_vaf_mean + variant_reads) / (prior_vaf_strength + depth)

        return posterior_mean

    @staticmethod
    def _vaf_score(vaf: float) -> float:
        """
        Calcule un score basé sur le VAF et la profondeur
        
        Returns:
            Score VAF ajusté (0 à vaf)
        """
        return vaf
    
    @staticmethod
    def depth_score(depth: float) -> float:
        """
        Calcule un score basé sur la profondeur de lecture
        
        Returns:
            Score de profondeur (float)
        """
        return depth 
    
    
    def effect_weighting(self, effect: str) -> float:
        """
        Attribue un poids à un type d'effet de mutation
        
        Returns:
            Poids numérique (float)
        """
        EFFECT_WEIGHTS = {
            'stop_gained': 1.00,            # codon stop prématuré
            'frameshift_variant': 0.95,     # décalage de lecture
            'stop_lost': 0.90,              # perte du stop -> extension
            'ITD': 0.90,                    # Internal Tandem Duplication (ex: FLT3-ITD)
            'PTD': 0.85,                    # Partial Tandem Duplication
            'inframe_codon_gain': 0.75,     # insertion in-frame (gain de codons)
            'inframe_codon_loss': 0.75,     # deletion in-frame (perte de codons)
            'non_synonymous_codon': 0.60,   # missense / substitution non synonyme
            'OTHER': 0.20                   # tout autre effet (peu probable d'être fort délétère)
        }

        return EFFECT_WEIGHTS.get(effect.lower(), 0.5)  # Poids par défaut si inconnu
