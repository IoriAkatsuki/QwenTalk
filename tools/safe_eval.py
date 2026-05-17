"""AST 安全数学求值 — 替代内置 evaluator 避免任意代码执行。

允许:
  - 数值常量 (int/float)
  - 算术运算 (+, -, *, /, //, %, **, 一元 +/-)
  - math 子集函数白名单 (sqrt/log/sin/cos/factorial 等)
  - 常量 pi, e

拒绝:
  - 字符串、列表、dict、lambda
  - 名字 (除白名单)
  - 关键字参数
"""
from __future__ import annotations

import ast
import math
import operator

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

_FUNCS: dict = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan",
                 "asin", "acos", "atan", "floor", "ceil", "factorial", "pi", "e")
}
_FUNCS["abs"] = abs
_FUNCS["round"] = round


def _walk(node):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("仅支持数值常量")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in _FUNCS:
            raise ValueError(f"未知名字: {node.id}")
        return _FUNCS[node.id]
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        return op(_walk(node.left), _walk(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(_walk(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("仅支持顶层函数调用")
        fn = _FUNCS.get(node.func.id)
        if fn is None or not callable(fn):
            raise ValueError(f"未知函数: {node.func.id}")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        return fn(*[_walk(a) for a in node.args])
    raise ValueError(f"不支持的语法: {type(node).__name__}")


def compute(expression: str):
    """AST 数学求值，失败抛 ValueError。"""
    return _walk(ast.parse(expression, mode="eval").body)
