from sfibai_b.tensorboard_schema import evaluation_tag, training_tag


def test_seven_layer_tensorboard_routes_priority_metrics_to_fixed_groups() -> None:
    assert evaluation_tag("image/grade_accuracy") == (
        "01_image_accuracy/val/grade_accuracy"
    )
    assert evaluation_tag("patient_max/grade_accuracy") == (
        "02_patient_accuracy/val/patient_max/grade_accuracy"
    )
    assert evaluation_tag("r_final") == "03_cor_selection/val/r_final"
    assert evaluation_tag("image/grade_macro_auroc") == (
        "04_auc_calibration/val/grade_macro_auroc"
    )
    assert evaluation_tag("image/mae") == "05_error_profile/val/image/mae"
    assert evaluation_tag("position/accuracy") == (
        "06_auxiliary/val/position/accuracy"
    )
    assert training_tag("total") == "07_training_health/train/total"

