"""Material ABI rejection and handle ownership across optional geometry updates."""
import ctypes as ct

import pytest

from raystrack.backends import native


@pytest.mark.parametrize("abi", [2, 3])
def test_old_material_abi_is_rejected_before_any_scene(monkeypatch, abi):
    class Function:
        def __call__(self):
            return abi

    class OldLibrary:
        def __init__(self):
            for name in ("rs_abi", "rs_backends", "rs_error", "rs_create", "rs_destroy",
                         "rs_shortwave", "rs_longwave"):
                setattr(self, name, Function())

    monkeypatch.setattr(native.ct, "CDLL", lambda path: OldLibrary())
    with pytest.raises(native.CapabilityError, match="requires material ABI 4"):
        native._load("old-material-abi-test")


def test_old_library_update_fallback_transfers_library_with_handle(monkeypatch):
    class Library:
        def __init__(self, handle):
            self.handle = handle
            self.destroyed = []

        def rs_create(self, backend, entries, count, result):
            ct.cast(result, ct.POINTER(ct.c_void_p))[0] = self.handle
            return 0

        def rs_destroy(self, handle):
            assert handle.value == self.handle
            self.destroyed.append(handle.value)

    original, replacement = Library(123), Library(456)
    libraries = iter((original,replacement))
    monkeypatch.setattr(native,"native_library",lambda backend: next(libraries))
    geometry = native.NativeGeometry("embree",[])
    try:
        assert geometry.update([]) is False
        assert geometry._library is replacement
        assert original.destroyed == [123]
    finally:
        geometry.close()
    assert replacement.destroyed == [456]
