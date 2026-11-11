"""Independent analytical gates for the unpublished delta plate reference."""
import numpy as np
import pytest

from validation.validate_clear_glazing import plate_fresnel


def test_nonabsorbing_plate_normal_limit_energy_and_grazing():
    r,t = plate_fresnel(np.array([1,.8,.3,0]))
    interface = ((1.52-1)/(1.52+1))**2
    assert r[0]==pytest.approx(2*interface/(1+interface))
    assert t[0]==pytest.approx((1-interface)/(1+interface))
    np.testing.assert_array_equal(r+t,1)
    assert r[-1]==1 and t[-1]==0
    assert np.all(np.diff(r)>0)
    no_interface = plate_fresnel([0,.5,1],1)
    np.testing.assert_array_equal(no_interface[0],0)
    np.testing.assert_array_equal(no_interface[1],1)


@pytest.mark.parametrize("cosine,index",[(np.nan,1.52),(-.1,1.52),(1.1,1.52),(.5,.8),(.5,np.inf)])
def test_plate_rejects_unsupported_inputs(cosine,index):
    with pytest.raises(ValueError):
        plate_fresnel(cosine,index)
