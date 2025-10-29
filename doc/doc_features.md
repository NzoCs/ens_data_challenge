# Voici une analyse détaillée de l'intérêt de chaque feature clinique pour prédire la survie :

## **ID (Identifiant patient)**

- **Intérêt** : Aucun (clé de jointure avec les données moléculaires)

## **CENTER (Centre clinique)**

- **Intérêt modéré** : Potentiellement informatif
- **Pourquoi** :
  - **Qualité des soins** : Variations entre centres (expertise, protocoles)
  - **Biais de sélection** : Certains centres peuvent recruter des cas plus complexes
  - **Facteurs socio-économiques** : Accès aux traitements, suivi
- **Exploitation** :
  - Encoding catégoriel ou groupement par taille/expertise
  - Peut servir de variable de stratification
- **Attention** : Risque de sur-apprentissage si peu de patients par centre

## **BM_BLAST (Blastes médullaires en %)** ⭐ **EXTRÊMEMENT IMPORTANT**

- **Intérêt critique** : Un des marqueurs pronostiques les plus puissants
- **Pourquoi** :
  - **Critère diagnostique** : Définit le type et le stade de la maladie (ex: MDS vs AML si >20%)
  - **Charge tumorale** : % élevé = maladie plus agressive
  - **Classification WHO** : Seuils pronostiques établis (5%, 10%, 20%)
  - Corrélé directement à la survie : plus le % est élevé, pire le pronostic
- **Exploitation** :
  - Feature continue très prédictive
  - Créer des catégories selon classifications cliniques
  - Interaction avec cytogénétique (blast + mauvaise cyto = très mauvais pronostic)

## **WBC (Globules blancs en G/L)** ⭐ **TRÈS IMPORTANT**

- **Intérêt majeur** : Marqueur pronostique établi
- **Pourquoi** :
  - **WBC élevé** (>20-30 G/L) = souvent mauvais pronostic dans les leucémies
  - Reflète la **prolifération tumorale**
  - **Leucocytose** peut indiquer une maladie agressive (ex: leucémie myélomonocytaire chronique)
  - Risque de complications (leucostase, syndrome de lyse tumorale)
- **Exploitation** :
  - Feature continue
  - Transformation log possible (distribution souvent asymétrique)
  - Seuils cliniques : <4, 4-10, 10-30, >30 G/L

## **ANC (Neutrophiles absolus en G/L)** ⭐ **TRÈS IMPORTANT**

- **Intérêt majeur** : Indicateur de fonction médullaire
- **Pourquoi** :
  - **Neutropénie** (<1.5 G/L) = risque infectieux, réserve médullaire faible
  - Reflète la capacité de la moelle à produire des cellules normales
  - ANC bas = moelle "envahie" par les cellules tumorales
  - **Pronostic** : ANC bas corrélé à survie réduite
- **Exploitation** :
  - Feature continue
  - Catégories cliniques : sévère (<0.5), modérée (0.5-1.0), normale (>1.5)

## **MONOCYTES (Monocytes en G/L)** ⭐ **IMPORTANT**

- **Intérêt élevé** : Spécifique selon le type de cancer
- **Pourquoi** :
  - **Monocytose** (>1 G/L) : critère diagnostique de certains cancers (LMMC = leucémie myélomonocytaire chronique)
  - Marqueur de sous-type de maladie
  - Peut indiquer une différenciation monocytaire anormale
- **Exploitation** :
  - Feature continue
  - Ratio monocytes/lymphocytes (si lymphocytes disponibles)
  - Seuil à 1 G/L pour monocytose

## **HB (Hémoglobine en g/dL)** ⭐ **TRÈS IMPORTANT**

- **Intérêt majeur** : Marqueur de sévérité
- **Pourquoi** :
  - **Anémie** = symptôme clé des cancers du sang (insuffisance médullaire)
  - HB bas (<10 g/dL) = mauvais pronostic établi
  - Reflète la capacité de production érythrocytaire
  - Impact sur qualité de vie et performance status
  - Critère majeur dans les scores pronostiques (IPSS, IPSS-R)
- **Exploitation** :
  - Feature continue très prédictive
  - Catégories WHO : sévère (<8), modérée (8-10), légère (10-12)
  - Dépendance aux transfusions (si HB très bas)

## **PLT (Plaquettes en G/L)** ⭐ **EXTRÊMEMENT IMPORTANT**

- **Intérêt critique** : Un des plus puissants prédicteurs
- **Pourquoi** :
  - **Thrombopénie** (<100 G/L) = marqueur pronostique majeur
  - PLT bas = insuffisance médullaire sévère, risque hémorragique
  - **Sévérité** : PLT <50 = mauvais, PLT <20 = très mauvais pronostic
  - Critère central dans tous les scores pronostiques (IPSS, WPSS, IPSS-R)
  - Reflète la fonction mégacaryocytaire
- **Exploitation** :
  - Feature continue extrêmement prédictive
  - Catégories : sévère (<50), modérée (50-100), normale (>150)
  - Transformation log possible (A voir)

En cours

## **CYTOGENETICS (Caryotype)** ⭐ **EXTRÊMEMENT IMPORTANT**

- **Intérêt critique** : LE facteur pronostique le plus puissant dans beaucoup de cancers du sang
- **Pourquoi** :
  - **Classification de risque** : Base des scores pronostiques (IPSS-R)
  - Certaines anomalies sont **très défavorables** : -7/del(7q), -5/del(5q), caryotype complexe (≥3 anomalies), anomalies chromosome 3
  - Certaines sont **favorables** : del(5q) isolée, del(20q), -Y
  - **Caryotype complexe** (≥3 anomalies) = très mauvais pronostic
  - **Caryotype normal** vs **anormal** = différence pronostique majeure
- **Exploitation** :
  - **Parsing ISCN** : Complexe mais essentiel !
    - Identifier : nombre de chromosomes (46 = normal, autres = aneuploidie)
    - Détecter délétions (del, -), additions (+), translocations t()
    - Compter nombre d'anomalies (complexité)
  - **Features à créer** :
    - **Catégorie de risque cytogénétique** : Très bon / Bon / Intermédiaire / Mauvais / Très mauvais (selon classification IPSS-R)
    - Présence d'anomalies spécifiques : -7, del(5q), -5, anomalie 3, etc.
    - **Complexité** : nombre d'anomalies (0, 1, 2, 3+)
    - Caryotype normal (46,XX ou 46,XY) vs anormal
    - Monosomies/trisomies
- **Exemples de classification** :
  - **Très favorable** : -Y isolé, del(11q) isolé
  - **Favorable** : Normal, del(5q) isolé, del(12p) isolé, del(20q) isolé, double incluant del(5q)
  - **Intermédiaire** : del(7q) isolé, +8 isolé, +19 isolé, i(17q) isolé, autres simples/doubles
  - **Défavorable** : -7, inv(3)/t(3q), double incluant -7/del(7q), caryotype complexe (3 anomalies)
  - **Très défavorable** : Caryotype complexe (>3 anomalies)

---

## **Features dérivées recommandées**

### 1. **Scores pronostiques établis** (à recréer)

- **IPSS-R** (International Prognostic Scoring System - Revised) :
  - Combine : BM_BLAST, cytogénétique, HB, PLT, ANC
  - Classification : Très bas / Bas / Intermédiaire / Élevé / Très élevé risque
- **WPSS** (WHO Prognostic Scoring System)

### 2. **Ratios et interactions**

- WBC/ANC (proportion de cellules anormales)
- PLT/HB (sévérité de l'atteinte médullaire)
- BM_BLAST × complexité cytogénétique
- Charge tumorale composite : (BM_BLAST + WBC_log) × risque_cyto

### 3. **Sévérité globale**

- Nombre de cytopénies (HB bas + PLT bas + ANC bas)
- Score composite d'insuffisance médullaire

### 4. **Catégorisation cytogénétique**

- **Essentiel** : Parser le caryotype pour extraire :
  - Risque cytogénétique (selon IPSS-R)
  - Nombre d'anomalies
  - Présence d'anomalies à haut risque (-7, -5, complexe)

---

## **Priorités pour le ML**

**Top 3 features les plus prédictives** :

1. **CYTOGENETICS** (une fois parsé et catégorisé)
2. **BM_BLAST**
3. **PLT**

Ces trois features constituent la base des scores pronostiques cliniques validés et sont probablement les plus importants pour votre modèle.

**Attention** : La cytogénétique nécessite un travail de preprocessing substantiel (parsing ISCN, classification de risque) mais c'est un investissement crucial pour la performance du modèle !
