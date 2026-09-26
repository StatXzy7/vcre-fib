from __future__ import annotations

import pandas as pd
import pytest

from sfibai_b.preprocessing_analysis import per_image_geometry


def test_letterbox_padding_fraction_and_aspect_ratio_are_exact() -> None:
    frame = pd.DataFrame(
        {
            "image_uid": ["wide", "square"],
            "split": ["train", "val"],
            "center_id": ["c1", "c2"],
            "crop_image_width": [20, 12],
            "crop_image_height": [10, 12],
        }
    )

    result = per_image_geometry(frame)

    assert result.loc[0, "aspect_ratio_width_over_height"] == pytest.approx(2.0)
    assert result.loc[0, "letterbox_padding_fraction"] == pytest.approx(0.5)
    assert result.loc[1, "letterbox_padding_fraction"] == pytest.approx(0.0)
