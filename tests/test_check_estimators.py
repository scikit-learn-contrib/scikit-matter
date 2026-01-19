from sklearn.utils.estimator_checks import parametrize_with_checks

from skmatter.decomposition import KernelPCovR, PCovC, PCovR, SketchMap
from skmatter.feature_selection import CUR as fCUR
from skmatter.feature_selection import FPS as fFPS
from skmatter.feature_selection import PCovCUR as fPCovCUR
from skmatter.feature_selection import PCovFPS as fPCovFPS
from skmatter.linear_model import Ridge2FoldCV  # OrthogonalRegression,
from skmatter.preprocessing import KernelNormalizer, StandardFlexibleScaler


@parametrize_with_checks(
    [
        fCUR(),
        fFPS(),
        fPCovCUR(),
        fPCovFPS(),
        KernelNormalizer(),
        KernelPCovR(mixing=0.5),
        PCovC(mixing=0.5),
        PCovR(mixing=0.5),
        Ridge2FoldCV(),
        SketchMap(),
        StandardFlexibleScaler(),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    """Test of the estimators are compatible with sklearn."""
    check(estimator)
