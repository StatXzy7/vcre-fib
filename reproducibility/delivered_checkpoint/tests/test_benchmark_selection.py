from __future__ import annotations

from sfibai_b.benchmark import select_runtime


def results(compile_throughput: float, *, audit_ok: bool = True) -> dict:
    return {
        "workers": {
            "4": {"stable": True, "images_per_second": 170.0},
            "6": {"stable": True, "images_per_second": 190.0},
            "8": {"stable": True, "images_per_second": 185.0},
        },
        "backends": {
            "eager": {"stable": True, "images_per_second": 190.0},
            "compile": {
                "stable": True,
                "images_per_second": compile_throughput,
                "numerical_audit_ok": audit_ok,
                "graph_breaks": 0,
            },
        },
    }


def test_selects_fastest_stable_worker_and_compile_above_eight_percent() -> None:
    selected = select_runtime(results(207.0))

    assert selected == {"num_workers": 6, "backend": "compile"}


def test_keeps_eager_when_compile_gain_is_below_threshold() -> None:
    selected = select_runtime(results(204.0))

    assert selected == {"num_workers": 6, "backend": "eager"}


def test_keeps_eager_when_compile_fails_numerical_or_graph_audit() -> None:
    failed_numerics = select_runtime(results(230.0, audit_ok=False))
    graph_break = results(230.0)
    graph_break["backends"]["compile"]["graph_breaks"] = 1

    assert failed_numerics["backend"] == "eager"
    assert select_runtime(graph_break)["backend"] == "eager"
