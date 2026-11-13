# -*- coding: utf-8 -*-
"""配置加载器：读取 config.yaml，提供路径解析与默认值。"""
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = None) -> dict:
    """加载并校验配置。返回 dict，路径字段解析为绝对路径的配置副本。"""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_root"] = str(PROJECT_ROOT)
    cfg["_config_path"] = str(path)

    # 相对路径 -> 绝对路径（仅 paths.* 段）
    paths = cfg.get("paths", {})
    cfg["paths"] = {k: _abs(p, paths.get("_base", PROJECT_ROOT)) for k, p in paths.items()}

    # 确保派生目录存在
    for key in ("partitions_dir", "annotations_dir", "meta_dir", "work_dir",
                "results_dir", "cache_dir", "log_dir"):
        if key in cfg["paths"]:
            Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    # 工具路径解析（相对项目根）
    tools = cfg.get("tools", {})
    for key in ("mmseqs", "blastp", "makeblastdb", "interproscan", "java"):
        v = tools.get(key, "")
        if v and not Path(v).is_absolute():
            tools[key] = str((PROJECT_ROOT / v))
    cfg["tools"] = tools
    return cfg


def _abs(p, base):
    if p is None:
        return None
    p = str(p)
    pp = Path(p)
    return p if pp.is_absolute() else str((base / pp).resolve())


def setup_logging(cfg: dict):
    """统一的日志配置：控制台 + 文件。"""
    import logging
    import sys
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg.get("runtime", {}).get("log_level", "INFO").upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)
    fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)
    return logging.getLogger("pipeline")
