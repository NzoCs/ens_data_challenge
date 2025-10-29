import pytest
import pandas as pd
from typing import cast, List


from ens_data_challenge.feature_engineering.cytogenetic_parser.parser import (
    CytogeneticsParser,
    CytogeneticsPatterns,
)
from ens_data_challenge.feature_engineering.cytogenetic_parser.types import (
    ParsedKaryotypeDict,
)
from ens_data_challenge.gloabls import TRAIN_CLINICAL_DATA_PATH, TEST_CLINICAL_DATA_PATH


@pytest.fixture
def parser() -> CytogeneticsParser:
    return CytogeneticsParser()


@pytest.fixture
def clinical_data_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CLINICAL_DATA_PATH)

@pytest.fixture
def clinical_data_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CLINICAL_DATA_PATH)

def test_parser_initialization(parser: CytogeneticsParser) -> None:
    # Patterns are centralized in the `CytogeneticsPatterns` enum
    assert hasattr(CytogeneticsPatterns, 'PLOIDY')
    assert isinstance(CytogeneticsPatterns.PLOIDY, str)


def test_parse_normal_karyotype(parser: CytogeneticsParser) -> None:
    parsed = parser.parse("46,XY")
    assert parsed is not None
    assert len(parsed) == 1
    result = parsed[0].to_dict()
    assert result['is_normal'] == True
    assert result['ploidy'] == 46
    assert result['sex_chromosomes'] == 'XY'


def test_parse_empty_string(parser: CytogeneticsParser) -> None:
    result = parser.parse("")
    assert result is None


def test_parse_nan(parser: CytogeneticsParser) -> None:
    result = parser.parse(pd.NA)
    assert result is None


def test_parse_complex_karyotype(parser: CytogeneticsParser) -> None:
    # Example complex karyotype with mosaic
    cyto = "47,XY,+8,del(5)(q31q33),-7[5]/46,XY[15]"
    parsed = parser.parse(cyto)
    assert parsed is not None
    assert len(parsed) == 2  # Mosaic with 2 clones
    
    # Check first clone (abnormal)
    clone1 = parsed[0]
    d1 = clone1.to_dict()
    assert d1['ploidy'] == 47
    assert d1['sex_chromosomes'] == 'XY'
    assert '7' in d1['monosomies']
    assert any(d['chromosome'] == '5' for d in d1['deletions'])
    assert '8' in d1['trisomies']
    assert clone1.cell_count == 5
    
    # Check second clone (normal)
    clone2 = parsed[1]
    d2 = clone2.to_dict()
    assert d2['ploidy'] == 46
    assert d2['sex_chromosomes'] == 'XY'
    assert d2['is_normal'] == True
    assert clone2.cell_count == 15


def test_parse_mosaic_karyotype(parser: CytogeneticsParser) -> None:
    # Example mosaic karyotype: abnormal clone with deletion in minority
    cyto = "46,XX,del(5)(q13q33~34)[5]/46,XX[15]"
    parsed = parser.parse(cyto)
    assert parsed is not None
    assert len(parsed) == 2
    
    # Check abnormal clone
    abnormal = next((p for p in parsed if not p.is_normal), None)
    assert abnormal is not None
    d_ab = abnormal.to_dict()
    assert d_ab['ploidy'] == 46
    assert d_ab['sex_chromosomes'] == 'XX'
    assert any(d['chromosome'] == '5' for d in d_ab['deletions'])
    assert abnormal.cell_count == 5
    
    # Check normal clone
    normal = next((p for p in parsed if p.is_normal), None)
    assert normal is not None
    d_norm = normal.to_dict()
    assert d_norm['ploidy'] == 46
    assert d_norm['sex_chromosomes'] == 'XX'
    assert normal.cell_count == 15


def test_edge_cases(parser: CytogeneticsParser) -> None:
    # Example mosaic karyotype: abnormal clone with deletion in minority
    edge_case_1 = "46,XY,inv(1)(p22q42)[1]/47,XY,+Y[1]/46,XY[18]"
    parsed = parser.parse(edge_case_1)
    assert parsed is not None
    assert len(parsed) == 3
    # Check ploidies
    ploidies = [p.ploidy for p in parsed]
    assert 46 in ploidies
    assert 47 in ploidies
    # Check sex chromosomes
    for p in parsed:
        assert p.sex_chromosomes == 'XY'
    # Check inversions
    inv_clones = [p for p in parsed if p.inversions]
    assert len(inv_clones) == 1
    inv = inv_clones[0].inversions[0]
    assert inv['chromosome'] == '1'
    assert inv['arm'] == 'p'
    assert inv['start'] == '22'
    assert inv['end'] == '42'

    edge_case_2 = "47,XY,dup(1)(q21q32),+8[2]/48,idem,-dup(1),+trp(1)(q21q32),+mar1[5]"
    parsed = parser.parse(edge_case_2)
    assert parsed is not None
    assert len(parsed) == 2
    # Check ploidies
    ploidies = [p.ploidy for p in parsed]
    assert 47 in ploidies
    assert 48 in ploidies
    # Check sex chromosomes for clones that have them
    for p in parsed:
        if p.sex_chromosomes is not None:
            assert p.sex_chromosomes == 'XY'
    # Check duplications in first clone
    clone1 = parsed[0]
    assert len(clone1.duplications) == 1
    dup = clone1.duplications[0]
    assert dup['chromosome'] == '1'
    assert dup['arm'] == 'q'
    assert dup['start'] == '21'
    assert dup['end'] == '32'
    # Check trisomy 8
    assert '8' in clone1.trisomies
    # Second clone has -dup(1), but parser extracts it anyway
    clone2 = parsed[1]
    assert len(clone2.duplications) == 1  # -dup(1) is still extracted
    # But +trp(1) is parsed as triplication
    assert len(clone2.triplications) == 1
    trp = clone2.triplications[0]
    assert trp['chromosome'] == '1'
    assert trp['arm'] == 'q'
    assert trp['start'] == '21'
    assert trp['end'] == '32'
    # Check marker
    assert '1' in clone2.markers

def test_is_normal_karyotype(parser: CytogeneticsParser) -> None:
    assert parser._is_normal_karyotype("46,XY") == True
    assert parser._is_normal_karyotype("46,XX") == True
    assert parser._is_normal_karyotype("47,XY,+8") == False


def test_extract_ploidy(parser: CytogeneticsParser) -> None:
    assert parser._extract_ploidy("46,XY") == 46
    assert parser._extract_ploidy("47,XY,+8") == 47
    assert parser._extract_ploidy("invalid") is None


def test_structural_anomalies(parser: CytogeneticsParser) -> None:
    # Test duplication
    cyto_dup = "46,XY,dup(1)(q21q32)"
    parsed = parser.parse(cyto_dup)
    assert parsed is not None
    assert len(parsed) == 1
    p = parsed[0]
    assert len(p.duplications) == 1
    dup = p.duplications[0]
    assert dup['chromosome'] == '1'
    assert dup['arm'] == 'q'
    assert dup['start'] == '21'
    assert dup['end'] == '32'

    # Test inversion
    cyto_inv = "46,XX,inv(3)(q21q26)"
    parsed = parser.parse(cyto_inv)
    assert parsed is not None
    assert len(parsed) == 1
    p = parsed[0]
    assert len(p.inversions) == 1
    inv = p.inversions[0]
    assert inv['chromosome'] == '3'
    assert inv['arm'] == 'q'
    assert inv['start'] == '21'
    assert inv['end'] == '26'

    # Test isochromosome
    cyto_iso = "45,X,i(17)(q10)"
    parsed = parser.parse(cyto_iso)
    assert parsed is not None
    assert len(parsed) == 1
    p = parsed[0]
    assert len(p.isochromosomes) == 1
    iso = p.isochromosomes[0]
    assert iso['chromosome'] == '17'
    assert iso['arm'] == 'q'
    assert iso['band'] == '10'

    # Test translocation
    cyto_t = "46,XY,t(9;22)(q34;q11)"
    parsed = parser.parse(cyto_t)
    assert parsed is not None
    assert len(parsed) == 1
    p = parsed[0]
    assert len(p.translocations) == 1
    t = p.translocations[0]
    assert t['chromosomes'] == ('9', '22')
    assert t['breakpoints'] == (('q', '34'), ('q', '11'))

    # Test triplication
    cyto_trp = "46,XY,trp(1)(q21q32)"
    parsed = parser.parse(cyto_trp)
    assert parsed is not None
    assert len(parsed) == 1
    p = parsed[0]
    assert len(p.triplications) == 1
    trp = p.triplications[0]
    assert trp['chromosome'] == '1'
    assert trp['arm'] == 'q'
    assert trp['start'] == '21'
    assert trp['end'] == '32'


def test_parse_clinical_train_cytogenetics(parser: CytogeneticsParser, clinical_data_train: pd.DataFrame) -> None:
    """Try to parse every value in the clinical `CYTOGENETICS` column.

    The test passes if parsing does not raise and returns either None
    or a list of ParsedKaryotype objects for every row.
    """
    # Import here to avoid unused import at module top
    from ens_data_challenge.feature_engineering.cytogenetic_parser.parser import ParsedKaryotype

    assert 'CYTOGENETICS' in clinical_data_train.columns

    total = 0
    parsed_count = 0
    none_count = 0

    for i, val in clinical_data_train['CYTOGENETICS'].items():
        total += 1

        try:
            res = parser.parse(val)
        except Exception as exc:  # pragma: no cover - surface any unexpected exceptions
            pytest.fail(f"Parsing raised on row {i}: {exc!r}")

        if res is None:
            none_count += 1
            continue

        # Expect a list of ParsedKaryotype
        assert isinstance(res, list), f"Expected list or None, got {type(res)} for row {i}"
        parsed_count += 1
        for p in res:
            assert isinstance(p, ParsedKaryotype)

    # Basic sanity: processed all rows
    assert parsed_count + none_count == total



def test_parse_clinical_test_cytogenetics(parser: CytogeneticsParser, clinical_data_test: pd.DataFrame) -> None:
    """Try to parse every value in the clinical `CYTOGENETICS` column.

    The test passes if parsing does not raise and returns either None
    or a list of ParsedKaryotype objects for every row.
    """
    # Import here to avoid unused import at module top
    from ens_data_challenge.feature_engineering.cytogenetic_parser.parser import ParsedKaryotype

    assert 'CYTOGENETICS' in clinical_data_test.columns

    total = 0
    parsed_count = 0
    none_count = 0

    for i, val in clinical_data_test['CYTOGENETICS'].items():
        total += 1

        try:
            res = parser.parse(val)
        except Exception as exc:  # pragma: no cover - surface any unexpected exceptions
            pytest.fail(f"Parsing raised on row {i}: {exc!r}")

        if res is None:
            none_count += 1
            continue

        # Expect a list of ParsedKaryotype
        assert isinstance(res, list), f"Expected list or None, got {type(res)} for row {i}"
        parsed_count += 1
        for p in res:
            assert isinstance(p, ParsedKaryotype)

    # Basic sanity: processed all rows
    assert parsed_count + none_count == total
