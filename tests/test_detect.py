import numpy as np

from curb_occupancy import detect as De


class FakeDetector:
    """Returns the boxes of `cars` (mosaic x0, x1) that fall inside each tile, as COCO cars."""

    def __init__(self, cars, H):
        self.cars, self.H, self.calls = cars, H, []

    def __call__(self, tiles):
        out = []
        for t in tiles:
            x = int(t[0, 0, 0]) * 256 + int(t[0, 0, 1])   # tile origin encoded in pixel (0, 0)
            self.calls.append(x)
            bx = [(max(a - x, 0), 10, min(b - x, t.shape[1]), self.H - 5)
                  for a, b in self.cars if b > x and a < x + t.shape[1]]
            out.append((np.array(bx, float).reshape(-1, 4), np.full(len(bx), 0.9),
                        np.full(len(bx), 2)))
        return out


def test_tile_detect_drops_cut_boxes_and_keeps_whole_cars():
    H, W, tile, stride = 64, 3000, 1000, 500
    img = np.zeros((H, W, 3), np.uint8)
    for x in range(0, W, 1):   # pixel (0, x) encodes x so the fake detector knows its tile
        img[0, x, 0], img[0, x, 1] = x // 256, x % 256
    cars = [(100, 400), (900, 1300), (2700, 2990)]
    det = FakeDetector(cars, H)
    df = De.tile_detect(img, det, tile_w=tile, stride=stride, batch=2)
    assert det.calls == [0, 500, 1000, 1500, 2000]
    got = sorted(zip(df.x0.round().astype(int), df.x1.round().astype(int)))
    assert got == cars                      # each car once, never a piece cut by a tile edge
    assert set(df.cls_name) == {"car"}


def test_detector_id_tracks_what_the_boxes_depend_on():
    dc = {"backend": "hf", "model": "m", "tile_w": 1280, "stride": 640, "conf": 0.15,
          "min_conf": 0.35}
    assert De.detector_id(dc) == {"backend": "hf", "model": "m", "tile_w": 1280,
                                  "stride": 640, "conf": 0.15}   # not the acceptance settings
