import pytest
import pandas as pd
from pathlib import Path
from llm_bench.profiling.parser import NsightParser


@pytest.fixture
def gpu_kernel_csv(tmp_path: Path) -> Path:
    csv_data = '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"\n45.2,1200000000,100,12000000,11500000,10000000,15000000,1000000,"ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn"\n30.1,800000000,100,8000000,7800000,7000000,9000000,500000,"fmha_v2_flash_attention_fp16_64_128_S_128_sm80_kernel"\n15.5,410000000,200,2050000,2000000,1800000,2500000,200000,"void cudnn::winograd_nonfused::winogradForwardData"\n9.2,245000000,50,4900000,4800000,4500000,5500000,300000,"void at::native::vectorized_elementwise_kernel"\n'
    f = tmp_path / "trace_gpukernsum.csv"
    f.write_text(csv_data)
    return f


@pytest.fixture
def gpu_mem_csv(tmp_path: Path) -> Path:
    csv_data = '"Operation","Total Time (ns)","Count","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)"\n"[CUDA memcpy HtoD]",500000000,50,10000000,9500000,8000000,12000000,1000000\n"[CUDA memcpy DtoH]",200000000,30,6666667,6500000,6000000,8000000,500000\n"[CUDA memset]",50000000,100,500000,480000,450000,600000,50000\n'
    f = tmp_path / "trace_gpumemtimesum.csv"
    f.write_text(csv_data)
    return f


def test_parse_kernel_summary(gpu_kernel_csv: Path) -> None:
    parser = NsightParser()
    df = parser.parse_kernel_summary(gpu_kernel_csv)
    assert len(df) == 4
    assert "name" in df.columns
    assert "time_pct" in df.columns
    assert "total_time_ns" in df.columns
    assert "avg_ns" in df.columns
    assert df.iloc[0]["time_pct"] == pytest.approx(45.2)


def test_parse_memory_summary(gpu_mem_csv: Path) -> None:
    parser = NsightParser()
    df = parser.parse_memory_summary(gpu_mem_csv)
    assert len(df) == 3
    assert "operation" in df.columns
    assert "total_time_ns" in df.columns


def test_top_kernels_by_time(gpu_kernel_csv: Path) -> None:
    parser = NsightParser()
    df = parser.parse_kernel_summary(gpu_kernel_csv)
    top = parser.top_kernels(df, n=2)
    assert len(top) == 2
    assert "gemm" in top.iloc[0]["name"].lower()


def test_classify_kernel_type(gpu_kernel_csv: Path) -> None:
    parser = NsightParser()
    df = parser.parse_kernel_summary(gpu_kernel_csv)
    classified = parser.classify_kernels(df)
    assert "kernel_type" in classified.columns
    types = set(classified["kernel_type"])
    assert "gemm" in types
    assert "attention" in types
