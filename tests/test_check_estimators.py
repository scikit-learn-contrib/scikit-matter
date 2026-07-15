from sklearn.utils.estimator_checks import parametrize_with_checks

from skmatter.decomposition import KernelPCovR, PCovC, PCovR, SketchMap
from skmatter.feature_selection import CUR as fCUR
from skmatter.feature_selection import FPS as fFPS
from skmatter.feature_selection import PCovCUR as fPCovCUR
from skmatter.feature_selection import PCovFPS as fPCovFPS
from skmatter.linear_model import Ridge2FoldCV  # OrthogonalRegression,
from skmatter.preprocessing import KernelNormalizer, StandardFlexibleScaler


def _expected_failed_checks(estimator):
    if isinstance(estimator, SketchMap):
        return {
            "check_sample_weight_equivalence_on_dense_data": (
                "Sketch-Map minimizes a non-convex stress; weighting samples and "
                "repeating them give the same objective only at the exact global "
                "optimum, which the iterative optimizer is not guaranteed to reach."
            ),
        }
    return {}


@parametrize_with_checks(
    [
        KernelPCovR(mixing=0.5),
        PCovR(mixing=0.5),
        PCovC(mixing=0.5),
        SketchMap(),
        fCUR(),
        fFPS(),
        fPCovCUR(),
        fPCovFPS(),
        Ridge2FoldCV(),
        KernelNormalizer(),
        StandardFlexibleScaler(),
    ],
    expected_failed_checks=_expected_failed_checks,
)
def test_sklearn_compatible_estimator(estimator, check):
    """Test of the estimators are compatible with sklearn."""
    check(estimator)
