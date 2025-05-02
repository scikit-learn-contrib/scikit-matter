from sklearn.utils.estimator_checks import parametrize_with_checks

from skmatter.decomposition import KernelPCovR
from skmatter.feature_selection import CUR as fCUR
from skmatter.feature_selection import FPS as fFPS
from skmatter.feature_selection import PCovCUR as fPCovCUR
from skmatter.feature_selection import PCovFPS as fPCovFPS
from skmatter.linear_model import Ridge2FoldCV  # OrthogonalRegression,
from skmatter.preprocessing import KernelNormalizer, StandardFlexibleScaler

import sys
sys.path.append('scikit-matter')
from src.skmatter.decomposition.pcovr_new import PCovR
from src.skmatter.decomposition.pcovc_new import PCovC

@parametrize_with_checks(
    [
        KernelPCovR(mixing=0.5),
        PCovR(mixing=0.5),
        PCovC(mixing=0.5),
        fCUR(),
        fFPS(),
        fPCovCUR(),
        fPCovFPS(),
        Ridge2FoldCV(),
        KernelNormalizer(),
        StandardFlexibleScaler(),
        #put PCovC/KPCovC once ready
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    """Test of the estimators are compatible with sklearn."""
    check(estimator)
