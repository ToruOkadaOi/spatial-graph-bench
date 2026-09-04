"""Unit tests for cryptographic hashing utilities."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from spatial_graph_bench.utils.hashing import (
    hash_array,
    hash_bytes,
    hash_config,
    hash_dict,
    hash_string,
)


class DummyModel(BaseModel):
    name: str
    val: int


def test_hash_string_and_bytes():
    h1 = hash_string("test_string")
    h2 = hash_bytes(b"test_string")
    assert h1 == h2
    assert len(h1) == 64


def test_hash_dict_invariance():
    d1 = {"a": 1, "b": "val", "c": [1, 2]}
    d2 = {"c": [1, 2], "a": 1, "b": "val"}
    assert hash_dict(d1) == hash_dict(d2)


def test_hash_config():
    m = DummyModel(name="test", val=42)
    h = hash_config(m)
    assert isinstance(h, str)
    assert len(h) == 64


def test_hash_array():
    arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    assert hash_array(arr1) == hash_array(arr2)
    # Different dtype must yield different hash
    assert hash_array(arr1) != hash_array(arr3)
