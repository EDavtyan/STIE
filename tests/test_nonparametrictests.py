from ST_Methods.non_parametric_tests import NonParametricTests
from tests.conftest import *


@pytest.fixture()
def create_tester():
    # Instantiate the NonParametricTests class
    tester = NonParametricTests(
        sid_columns=['sid_1', 'sid_2'],
        grouping_column='hours',
        treatment_variant='A',
        control_variant='B'
    )

    return tester


@pytest.mark.parametrize('simulate_dataset', [True],
                         indirect=True)  # 'indirect' means the param goes to the fixture
def test_run_permutation_test(create_tester, simulate_dataset):
    tester = create_tester

    # Perform the transform method
    tester.transform(simulate_dataset)

    result = tester.run_test(test_type='permutation')

    # Basic checks
    assert isinstance(result, dict)
    assert all(
        key in result for key in ['null_distribution', 'pvalue', 'statistic', 'within_ci', 'confidence_interval'])

    # Check null_distribution
    null_distribution = result['null_distribution']
    assert isinstance(null_distribution, np.ndarray)
    assert null_distribution.shape[1] == 1  # Ensure it's a column vector
    assert all(isinstance(val, (float, np.float_)) for val in null_distribution.flatten())

    # Check pvalue
    pvalue = result['pvalue']
    assert isinstance(pvalue, np.ndarray)
    assert pvalue.shape == (1,)  # Ensure it's a single value in array format
    assert 0 <= pvalue[0] <= 1

    # Check statistic
    statistic = result['statistic']
    assert isinstance(statistic, np.ndarray)
    assert statistic.shape == (1,)  # Ensure it's a single value in array format

    # Check within_ci
    assert isinstance(result['within_ci'], np.bool_)

    # Check confidence_interval
    confidence_interval = result['confidence_interval']
    assert isinstance(confidence_interval, tuple)
    assert len(confidence_interval) == 2
    assert all(isinstance(val, (float, np.float_)) for val in confidence_interval)


@pytest.mark.parametrize('simulate_dataset', [True],
                         indirect=True)  # 'indirect' means the param goes to the fixture
def test_run_bootstrap_test(create_tester, simulate_dataset):
    tester = create_tester

    # Perform the transform method
    tester.transform(simulate_dataset)

    result = tester.run_test(test_type='bootstrap')

    # Basic checks
    assert isinstance(result, dict)
    assert all(
        key in result for key in ['bootstrap_distribution', 'confidence_interval', 'standard_error', 'statistic'])

    # Check bootstrap_distribution
    bootstrap_distribution = result['bootstrap_distribution']
    assert isinstance(bootstrap_distribution, np.ndarray)
    assert bootstrap_distribution.shape[1] > 0  # Ensure it has at least one element
    assert all(isinstance(val, (float, np.float_)) for val in bootstrap_distribution.flatten())

    # Check confidence_interval
    confidence_interval = result['confidence_interval']
    assert hasattr(confidence_interval, 'low') and hasattr(confidence_interval, 'high')

    ci_low = confidence_interval.low
    ci_high = confidence_interval.high

    assert isinstance(ci_low, np.ndarray) and ci_low.shape == (1,)  # Ensure it's a single value in array format
    assert isinstance(ci_high, np.ndarray) and ci_high.shape == (1,)  # Ensure it's a single value in array format

    # Check standard_error
    standard_error = result['standard_error']
    assert isinstance(standard_error, np.ndarray)
    assert standard_error.shape == (1,)  # Ensure it's a single value in array format
    assert standard_error[0] >= 0  # Standard error should be non-negative

    # Check statistic
    statistic = result['statistic']
    assert isinstance(statistic, (float, np.float_))
