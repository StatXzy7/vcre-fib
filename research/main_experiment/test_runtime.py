import itertools
import numpy as np
import pandas as pd
import pytest
from sfibai_b.data import EpochShuffleSampler
from runtime import GlobalBatchShard, extended_metrics


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_native_training_shards_have_exact_order_and_no_padding(world):
    size = 83722
    expected = list(EpochShuffleSampler(size=size, seed=2026))
    ranks = [list(GlobalBatchShard(size, 2026, rank, world)) for rank in range(world)]
    reconstructed = []
    for batches in zip(*ranks):
        reconstructed.extend(x for block in itertools.zip_longest(*batches) for x in block if x is not None)
    assert reconstructed == expected
    assert len({i for i, _ in reconstructed}) == size
    assert len(ranks[0][-1]) == (size % 24 + world - 1) // world


def test_auxiliary_metrics_unavailable_for_baseline_and_all_core_groups_exist():
    frame = pd.DataFrame(dict(image_uid=['a','b','c','d'], patient_uid=['p1','p1','p2','p2'],
                              center_id=['c1']*4, true_score=[0.,1.,2.,3.], pred_score=[.1,.8,2.1,2.8]))
    m = extended_metrics(frame)
    assert m['position'] == {'available': False}
    assert m['lesion'] == {'available': False}
    for group in ['image', 'patient_max', 'patient_median']:
        assert {'mae','accuracy_within_0_3','accuracy_within_0_5','grade_accuracy','tmae','severe_error_rate','clinical_risk'} <= set(m[group])
    assert m['r_final'] == pytest.approx(.4*m['image']['cor']+.4*m['patient_max']['cor']+.2*m['center_balanced_patient_max']['cor'])
    assert np.asarray(m['grade_confusion_matrix']).sum() == 4


def test_duplicate_evaluation_images_fail_closed():
    frame = pd.DataFrame(dict(image_uid=['same','same'], patient_uid=['p','p'], center_id=['c','c'], true_score=[1.,1.], pred_score=[1.,1.]))
    with pytest.raises(ValueError, match='unique'):
        extended_metrics(frame)
