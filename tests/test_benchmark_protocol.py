import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('benchmark_protocol',Path(__file__).resolve().parents[1]/'tools/benchmark_protocol.py')
protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)


def test_raw_pair_reporting_preserves_uncertainty_and_does_not_turn_speed_into_correctness():
    records=[dict(pair=pair,backend=backend,elapsed_seconds=(pair+1)*scale,status='ok',correctness_checked=True)
        for pair in range(7) for backend,scale in [('python',2.),('rust',1.)]]
    report=protocol.paired_summary(records,repeats=7)
    assert report['ratio_python_over_rust']==2.
    assert report['backends']['rust']['median_seconds']==4.
    assert report['backends']['rust']['iqr_seconds']==3.
    assert report['backends']['rust']['p95_seconds']==pytest.approx(6.7)
    assert report['peak_memory_comparison'] is None
    assert report['performance_target']=='report_only'
    records[-1]['correctness_checked']=False
    diagnostic=protocol.paired_summary(records,repeats=7)
    assert not diagnostic['valid_speed_comparison']
    assert 'ratio_python_over_rust' not in diagnostic
    with pytest.raises(ValueError,match='both backend'):
        protocol.paired_summary(records[:-1],repeats=7)
