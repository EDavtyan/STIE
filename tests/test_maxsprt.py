from pyspark.sql.types import StructType, TimestampType, LongType, DoubleType, IntegerType, BooleanType

from ST_Methods.maxsprt import MaxSPRT
from tests.conftest import *


@pytest.fixture()
def create_tester():
    tester = MaxSPRT(sid_columns=["sid_1", "sid_2"],
                     grouping_column="hours",
                     treatment_variant="B",
                     control_variant="A",
                     item_id_column="item_id",
                     variant_column="variant")

    # tester.transform(simulate_dataset)

    return tester


@pytest.mark.parametrize('simulate_dataset', [False],
                         indirect=True)  # 'indirect' means the param goes to the fixture
def test_calculate_thresholds(create_tester, simulate_dataset):
    tester = create_tester

    # Run calculate_thresholds
    result_threshold = tester.calculate_threshold(simulate_dataset,
                                                  num_experiments=3)

    # Assert result
    assert isinstance(result_threshold, float)


@pytest.mark.parametrize('simulate_dataset', [False],
                         indirect=True)  # 'indirect' means the param goes to the fixture
def test_run_test(create_tester, simulate_dataset):
    tester = create_tester

    tester.transform(simulate_dataset)

    # Parameters for the calculate_threshold function, if needed
    n = 42
    iterations = 10000
    threshold = 42.

    # Assume that run_test takes `O_i_threshold` as a parameter and returns a DataFrame
    output_df = tester.run_test(L_i_threshold=threshold)

    # Expected output schema

    expected_output_schema = [
        ('hours', TimestampType()),
        ('T_i_sngl', LongType()),
        ('m_i_sngl', DoubleType()),
        ('T_i', LongType()),
        ('m_i', DoubleType()),
        ('p_1i', DoubleType()),
        ('numerator_1', DoubleType()),
        ('numerator_2', DoubleType()),
        ('denominator', LongType()),
        ('L_i', DoubleType()),
        ('threshold', DoubleType()),
        ('can_stop', BooleanType())
    ]

    # Extracting the actual schema from the DataFrame
    actual_output_schema = [(field.name, field.dataType) for field in output_df.schema.fields]

    # Asserting that the actual schema matches the expected schema
    assert actual_output_schema == expected_output_schema, f"Schema mismatch. Expected {expected_output_schema}, got {actual_output_schema}"
