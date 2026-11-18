import numpy as np


def _texture(w_m, seed=0):
    import cv2
    from curb_occupancy.register import PXM
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (216, int(w_m * PXM), 3)).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 3)


def _strip(img, x0):
    from curb_occupancy.register import Strip, prep
    return Strip(prep(img), np.ones(img.shape[1], bool), x0)


def test_register_recovers_shift_up_to_the_ends():
    from curb_occupancy.register import register, smooth
    img = _texture(200)
    a, b = _strip(img, 0.0), _strip(img[:, :] , 3.4)  # same scene, axis offset 3.4 m
    d = smooth(register(a, b, search_m=10.0))
    assert len(d) and d.ok.all()
    assert np.allclose(d["shift"], -3.4, atol=0.05)
    assert d.x_anchor.min() < 20  # the search window is cut at the strip ends, not skipped


def test_fit_map_reversed_passes_through_points():
    from curb_occupancy.register import fit_map
    xc = np.linspace(0, 300, 61)
    xr = 500 - 0.98 * xc + 2.0 * np.sin(xc / 40)
    m = fit_map(xc, xr)
    assert m.reverse
    assert np.abs(m(xc[3:-3]) - xr[3:-3]).max() < 0.3
