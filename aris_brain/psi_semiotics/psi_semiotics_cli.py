"""
Ψ-Semiotics CLI — 量子符号学引擎命令行接口

使用:
  python psi_semiotics_cli.py concept <name> [--desc <text>]
  python psi_semiotics_cli.py path <source> <target>
  python psi_semiotics_cli.py analogy <a> <b> <c> <d>
  python psi_semiotics_cli.py compose <a> <b> <result>
  python psi_semiotics_cli.py evolve <concept> [--steps 10]
  python psi_semiotics_cli.py field <text>
  python psi_semiotics_cli.py verify
  python psi_semiotics_cli.py interactive
  python psi_semiotics_cli.py run <psi_file>
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.WARNING)


def load_engine():
    """延迟加载 Ψ-Semiotics 引擎"""
    from psi_semiotics.psi_semiotics_core import PsiSemioticsEngine, Rotor, _hash_to_vec
    from psi_semiotics.structured_encoder import StructuredSemanticEncoder
    eng = PsiSemioticsEngine(dim=1024)
    return eng, Rotor, _hash_to_vec, StructuredSemanticEncoder


def cmd_concept(args):
    """注册概念"""
    eng, Rotor, _, _ = load_engine()
    name = args[0]
    desc = args[1] if len(args) > 1 else ""
    eng.add_symbol(name, desc=desc)
    sym = eng.get_symbol(name)
    return {
        "name": name,
        "importance": sym.importance if sym else None,
        "symbols_total": len(eng.symbols),
    }


def cmd_path(args):
    """注册/学习路径"""
    eng, Rotor, _, _ = load_engine()
    source, target = args[0], args[1]
    
    eng.add_symbol(source, "")
    eng.add_symbol(target, "")
    
    s_vec = eng.symbols[source].center
    t_vec = eng.symbols[target].center
    
    start = time.time_ns()
    rotor = Rotor.learn(s_vec, t_vec)
    elapsed = (time.time_ns() - start) / 1000
    
    eng.rotors[f"rotor_{source}_{target}"] = rotor
    sim = float(s_vec @ t_vec)
    
    return {
        "source": source,
        "target": target,
        "similarity": round(sim, 4),
        "rotor_learn_us": round(elapsed, 1),
        "path": f"Path(Tensor, {source}, {target})",
    }


def cmd_analogy(args):
    """类比推理"""
    eng, Rotor, _, enc = load_engine()
    a, b, c, d = args[0], args[1], args[2], args[3]
    
    for name in [a, b, c, d]:
        eng.add_symbol(name, "")
    
    # 验证类比方向
    s_a, s_b, s_c = eng.symbols[a].center, eng.symbols[b].center, eng.symbols[c].center
    
    rotor = Rotor.learn(s_a, s_b)
    predicted = rotor.apply(s_c)
    
    actual_d = eng.symbols[d].center
    strength = float(predicted @ actual_d)
    
    # 向量代数验证
    diff_ab = s_a - s_b
    diff_cd = eng.symbols[c].center - actual_d
    dir_sim = float(diff_ab @ diff_cd) / (np.linalg.norm(diff_ab) * np.linalg.norm(diff_cd) + 1e-10)
    
    return {
        "analogy": f"{a}:{b} :: {c}:{d}",
        "predicted_to_actual": round(strength, 4),
        "direction_consistency": round(dir_sim, 4),
        "valid": strength > 0.3 or dir_sim > 0.3,
    }


def cmd_compose(args):
    """符号组合"""
    eng, Rotor, _, _ = load_engine()
    a, b, result = args[0], args[1], args[2]
    
    for name in [a, b]:
        eng.add_symbol(name, "")
    
    sym = eng.compose_add(a, b, result)
    
    sa = float(sym.center @ eng.symbols[a].center) if a in eng.symbols else None
    sb = float(sym.center @ eng.symbols[b].center) if b in eng.symbols else None
    
    return {
        "operation": f"{a} ⊕ {b} → {result}",
        f"sim_to_{a}": round(sa, 4) if sa else None,
        f"sim_to_{b}": round(sb, 4) if sb else None,
    }


def cmd_evolve(args):
    """薛定谔演化"""
    from math_physics_lib import SchrodingerEvolution
    eng, Rotor, _, _ = load_engine()
    
    concept_name = args[0]
    steps = int(args[1]) if len(args) > 1 else 10
    
    eng.add_symbol(concept_name, "")
    initial = eng.symbols[concept_name].center
    
    sch = SchrodingerEvolution(dim=1024)
    sch.set_hamiltonian([1.0, 0.5, 0.3, 0.2, 0.1])
    
    start = time.time()
    states = sch.evolve(initial, dt=0.1, steps=steps)
    elapsed = time.time() - start
    
    return {
        "concept": concept_name,
        "steps": steps,
        "final_norm": round(float(np.linalg.norm(states[-1])), 4),
        "overlap_with_initial": round(float(states[-1] @ initial), 4),
        "latency_ms": round(elapsed * 1000, 1),
    }


def cmd_field(args):
    """语义场查询"""
    eng, Rotor, _, enc = load_engine()
    text = " ".join(args)
    
    v = enc().encode(text) if text else np.zeros(1024)
    field = eng.semantic_field_map(v, top_k=5)
    
    return {
        "query": text,
        "field": [{"symbol": n, "strength": round(s, 4)} for n, s in field],
    }


def cmd_verify(args):
    """完整性验证"""
    eng, Rotor, _, enc = load_engine()
    
    # 注册测试集
    tests = [
        ("king", "queen"),
        ("man", "woman"),
        ("consciousness", "awareness"),
        ("hot", "cold"),
        ("cat", "dog"),
    ]
    
    results = []
    for a, b in tests:
        eng.add_symbol(a, "")
        eng.add_symbol(b, "")
        sim = float(eng.symbols[a].center @ eng.symbols[b].center)
        results.append({
            "pair": f"{a}~{b}",
            "similarity": round(sim, 4),
        })
    
    # 验证类比
    eng.add_symbol("king", ""); eng.add_symbol("queen", "")
    eng.add_symbol("man", ""); eng.add_symbol("woman", "")
    k, q, m, w = [eng.symbols[n].center for n in ["king", "queen", "man", "woman"]]
    rotor = Rotor.learn(k, q)
    pred = rotor.apply(m)
    analogy_strength = float(pred @ w)
    
    return {
        "concepts": len(eng.symbols),
        "rotors": len(eng.rotors),
        "ops": eng.semantic_ops,
        "semantic_checks": results,
        "king_queen_man_woman": round(analogy_strength, 4),
    }


def cmd_interactive(args):
    """交互模式"""
    eng, Rotor, _, enc = load_engine()
    print("Ψ-Semiotics Interactive")
    print("Commands: concept <name>, path <a> <b>, analogy <a> <b> <c> <d>,")
    print("          compose <a> <b> <r>, field <text>, evolve <c>, verify, quit")
    print()
    
    while True:
        try:
            line = input("Ψ> ").strip()
            if not line:
                continue
            if line == "quit":
                break
            
            parts = line.split()
            cmd = parts[0]
            cmd_args = parts[1:]
            
            handlers = {
                "concept": cmd_concept,
                "path": cmd_path,
                "analogy": cmd_analogy,
                "compose": cmd_compose,
                "field": cmd_field,
                "evolve": cmd_evolve,
                "verify": cmd_verify,
            }
            
            handler = handlers.get(cmd)
            if handler:
                result = handler(cmd_args)
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"Unknown: {cmd}")
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Error: {e}")


def cmd_run(args):
    """运行 PsiLang .psi 文件（占位）"""
    psi_file = args[0]
    path = Path(psi_file)
    if not path.exists():
        return {"error": f"File not found: {psi_file}"}
    
    code = path.read_text(encoding="utf-8")
    return {
        "file": psi_file,
        "size": len(code),
        "status": "parsed (PsiLang v3 runtime pending)",
        "note": "PsiLang v3 compiler integration in progress",
    }


# ── 命令行路由 ──

COMMANDS = {
    "concept": cmd_concept,
    "path": cmd_path,
    "analogy": cmd_analogy,
    "compose": cmd_compose,
    "evolve": cmd_evolve,
    "field": cmd_field,
    "verify": cmd_verify,
    "interactive": cmd_interactive,
    "run": cmd_run,
}


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  python {sys.argv[0]} <command> [args...]")
        print(f"  Commands: {', '.join(COMMANDS.keys())}")
        return
    
    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    if cmd in COMMANDS:
        result = COMMANDS[cmd](args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Unknown: {cmd}")


if __name__ == "__main__":
    main()
