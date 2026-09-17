"""
uv run examples/basic.py
"""
from renikud_onnx import G2P

g2p = G2P()  # fetches model from HF on first run
print(g2p.phonemize("הוא רצה את זה גם, אבל היא רצה מהר והקדימה אותו"))
