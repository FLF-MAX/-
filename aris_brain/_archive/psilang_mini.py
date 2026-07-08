"""
PsiLang Mini VM -- zero external dependencies (no numpy)
Complete lexer, parser, compiler, VM in pure Python.
"""

import struct, json, hashlib, time, os, sys, re, math, random, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum, auto
from dataclasses import dataclass, field

logger = logging.getLogger("psilang_mini")

# -----------------------------------------------------------
# QVector -- pure Python vector, no numpy
# -----------------------------------------------------------

class QVector:
    """Quantum state vector -- pure Python"""
    __slots__ = ('data', 'dim')
    
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.data = [0.0] * dim
    
    def copy(self):
        v = QVector(self.dim)
        v.data = self.data[:]
        return v
    
    def normalize(self):
        norm = math.sqrt(sum(x*x for x in self.data))
        if norm > 1e-10:
            self.data = [x / norm for x in self.data]
    
    def prob_dist(self):
        total = sum(x*x for x in self.data)
        if total > 1e-10:
            return [x*x/total for x in self.data]
        return [1.0/self.dim] * self.dim
    
    def collapse(self, temperature=0.5):
        prob = self.prob_dist()
        if temperature > 0.8:
            r = random.random()
            cum = 0.0
            for i, p in enumerate(prob):
                cum += p
                if r < cum:
                    return i
            return self.dim - 1
        else:
            return max(range(self.dim), key=lambda i: prob[i])
    
    def entropy(self):
        prob = self.prob_dist()
        h = 0.0
        for p in prob:
            if p > 1e-10:
                h -= p * math.log2(p)
        return h


# -----------------------------------------------------------
# Opcodes
# -----------------------------------------------------------

class Opcode(Enum):
    QSTATE = 0x01
    QNORM = 0x02
    QAMP = 0x04
    QENT = 0x07
    PSI_CYCLE = 0x10
    CONCEPT_ACTIVATE = 0x11
    MEM_STORE = 0x20
    OBSERVE = 0x30
    EMIT = 0x31
    LOG = 0x32
    NOP = 0x00
    HALT = 0xFF


# -----------------------------------------------------------
# Token types
# -----------------------------------------------------------

class TT(Enum):
    ID = auto()
    QSTATE = auto()
    NUM = auto()
    STR = auto()
    EQ = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    TILDE = auto()
    COMMA = auto()
    COLON = auto()
    LP = auto()
    RP = auto()
    LB = auto()
    RB = auto()
    LBRACK = auto()
    RBRACK = auto()
    KW_CYCLE = auto()
    KW_QSTATE = auto()
    KW_CONCEPT = auto()
    KW_AMPLIFY = auto()
    KW_ENTANGLE = auto()
    KW_PERCEIVE = auto()
    KW_SELECT = auto()
    KW_INTEGRATE = auto()
    KW_LEARN = auto()
    KW_REMEMBER = auto()
    KW_OBSERVE = auto()
    KW_ON = auto()
    KW_EMIT = auto()
    KW_LOG = auto()
    KW_IF = auto()
    KW_ELSE = auto()
    KW_TRUE = auto()
    KW_FALSE = auto()
    NL = auto()
    EOF = auto()


@dataclass
class Token:
    type: TT
    value: Any = None


class Lexer:
    KW = {
        'cycle': TT.KW_CYCLE, 'qstate': TT.KW_QSTATE,
        'concept': TT.KW_CONCEPT, 'amplify': TT.KW_AMPLIFY,
        'entangle': TT.KW_ENTANGLE, 'perceive': TT.KW_PERCEIVE,
        'select': TT.KW_SELECT, 'integrate': TT.KW_INTEGRATE,
        'learn': TT.KW_LEARN, 'remember': TT.KW_REMEMBER,
        'observe': TT.KW_OBSERVE, 'on': TT.KW_ON,
        'emit': TT.KW_EMIT, 'log': TT.KW_LOG,
        'if': TT.KW_IF, 'else': TT.KW_ELSE,
        'true': TT.KW_TRUE, 'false': TT.KW_FALSE,
    }
    
    def __init__(self, source: str):
        self.src = source
        self.pos = 0
        self.tokens = []
    
    def tokenize(self):
        while self.pos < len(self.src):
            ch = self.src[self.pos]
            
            if ch in ' \t\r':
                self.pos += 1
                continue
            
            if ch == '\n':
                self.tokens.append(Token(TT.NL))
                self.pos += 1
                continue
            
            if self.src[self.pos:self.pos+2] == '//':
                while self.pos < len(self.src) and self.src[self.pos] != '\n':
                    self.pos += 1
                continue
            
            if ch == '|':
                self.pos += 1
                name = ''
                while self.pos < len(self.src) and self.src[self.pos] not in '>|\n':
                    name += self.src[self.pos]
                    self.pos += 1
                if self.pos < len(self.src) and self.src[self.pos] == '>':
                    self.pos += 1
                self.tokens.append(Token(TT.QSTATE, name.strip()))
                continue
            
            if ch == '"':
                self.pos += 1
                s = ''
                while self.pos < len(self.src) and self.src[self.pos] != '"':
                    if self.src[self.pos] == '\\':
                        self.pos += 1
                        if self.pos < len(self.src):
                            s += self.src[self.pos]
                    else:
                        s += self.src[self.pos]
                    self.pos += 1
                if self.pos < len(self.src):
                    self.pos += 1
                self.tokens.append(Token(TT.STR, s))
                continue
            
            if ch.isdigit() or (ch == '.' and self.pos+1 < len(self.src) and self.src[self.pos+1].isdigit()):
                start = self.pos
                while self.pos < len(self.src) and (self.src[self.pos].isdigit() or self.src[self.pos] == '.'):
                    self.pos += 1
                s = self.src[start:self.pos]
                self.tokens.append(Token(TT.NUM, float(s) if '.' in s else int(s)))
                continue
            
            single = {
                '=': TT.EQ, '+': TT.PLUS, '-': TT.MINUS, '*': TT.MUL,
                '~': TT.TILDE, ',': TT.COMMA, ':': TT.COLON,
                '(': TT.LP, ')': TT.RP, '{': TT.LB, '}': TT.RB,
                '[': TT.LBRACK, ']': TT.RBRACK,
            }
            if ch in single:
                self.tokens.append(Token(single[ch], ch))
                self.pos += 1
                continue
            
            if ch.isalpha() or ch == '_':
                start = self.pos
                while self.pos < len(self.src) and (self.src[self.pos].isalnum() or self.src[self.pos] == '_'):
                    self.pos += 1
                word = self.src[start:self.pos]
                tt = self.KW.get(word, TT.ID)
                self.tokens.append(Token(tt, word))
                continue
            
            self.pos += 1
        
        self.tokens.append(Token(TT.EOF))
        return self.tokens


# -----------------------------------------------------------
# AST
# -----------------------------------------------------------

class Node: pass

@dataclass
class Program(Node):
    stmts: List[Any] = field(default_factory=list)

@dataclass
class QState(Node):
    name: str; amp: float = 1.0

@dataclass
class QStateDecl(Node):
    name: str; states: List[QState] = field(default_factory=list)

@dataclass
class Concept(Node):
    name: str; props: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Cycle(Node):
    name: str; body: List[Any] = field(default_factory=list)

@dataclass
class Amplify(Node):
    target: str; factor: float = 1.0

@dataclass
class Entangle(Node):
    left: str; right: str

@dataclass
class MemStore(Node):
    content: str; importance: float = 0.5

@dataclass
class Observe(Node):
    event: str = "collapse"; body: List[Any] = field(default_factory=list)

@dataclass
class Emit(Node):
    value: Any = None

@dataclass
class Call(Node):
    name: str; args: List[Any] = field(default_factory=list)

@dataclass
class Ident(Node):
    name: str

@dataclass
class Num(Node):
    value: float = 0.0

@dataclass
class Str(Node):
    value: str = ""


# -----------------------------------------------------------
# Parser
# -----------------------------------------------------------

class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.pos = 0
    
    def parse(self):
        prog = Program()
        while not self._check(TT.EOF):
            s = self._stmt()
            if s: prog.stmts.append(s)
            while self._check(TT.NL): self._adv()
        return prog
    
    def _stmt(self):
        if self._check(TT.KW_QSTATE): return self._qstate()
        if self._check(TT.KW_CONCEPT): return self._concept()
        if self._check(TT.KW_CYCLE): return self._cycle()
        if self._check(TT.KW_AMPLIFY): return self._amplify()
        if self._check(TT.KW_ENTANGLE): return self._entangle()
        if self._check(TT.KW_LEARN): return self._learn()
        if self._check(TT.KW_OBSERVE): return self._observe()
        self._adv(); return None
    
    def _qstate(self):
        self._adv()
        name = self._consume(TT.ID).value
        self._consume(TT.EQ)
        states = [self._qslit()]
        while self._check(TT.PLUS):
            self._adv()
            states.append(self._qslit())
        return QStateDecl(name=name, states=states)
    
    def _qslit(self):
        t = self._consume(TT.QSTATE)
        amp = 1.0
        if self._check(TT.MUL):
            self._adv()
            amp = self._consume(TT.NUM).value
        return QState(name=t.value, amp=amp)
    
    def _concept(self):
        self._adv()
        name = self._consume(TT.ID).value
        props = {}
        if self._check(TT.LB):
            self._adv()
            while not self._check(TT.RB) and not self._check(TT.EOF):
                if self._check(TT.ID):
                    k = self._adv().value
                    self._consume(TT.COLON)
                    if self._check(TT.NUM): props[k] = self._adv().value
                    elif self._check(TT.STR): props[k] = self._adv().value
                    elif self._check(TT.KW_TRUE): props[k] = True; self._adv()
                    elif self._check(TT.KW_FALSE): props[k] = False; self._adv()
                    elif self._check(TT.LBRACK):
                        self._adv(); items = []
                        while not self._check(TT.RBRACK):
                            if self._check(TT.STR): items.append(self._adv().value)
                            self._consume(TT.COMMA, True)
                        self._consume(TT.RBRACK); props[k] = items
                self._consume(TT.COMMA, True)
                self._consume(TT.NL, True)
            self._consume(TT.RB)
        return Concept(name=name, props=props)
    
    def _cycle(self):
        self._adv(); name = self._consume(TT.ID).value
        self._consume(TT.LB); body = []
        while not self._check(TT.RB) and not self._check(TT.EOF):
            if self._check(TT.KW_PERCEIVE):
                self._adv(); body.append(Call("perceive", []))
            elif self._check(TT.KW_SELECT):
                self._adv(); body.append(Call("select", []))
            elif self._check(TT.KW_INTEGRATE):
                self._adv(); body.append(Call("integrate", [Num(0.5)]))
            elif self._check(TT.NL): self._adv()
            else:
                s = self._stmt(); 
                if s: body.append(s)
        self._consume(TT.RB)
        return Cycle(name=name, body=body)
    
    def _amplify(self):
        self._adv(); t = self._consume(TT.QSTATE)
        f = 1.0
        if self._check(TT.MUL): self._adv(); f = self._consume(TT.NUM).value
        return Amplify(target=t.value, factor=f)
    
    def _entangle(self):
        self._adv(); l = self._consume(TT.QSTATE)
        self._consume(TT.TILDE); r = self._consume(TT.QSTATE)
        return Entangle(left=l.value, right=r.value)
    
    def _learn(self):
        self._adv(); self._consume(TT.LP)
        content = self._consume(TT.STR).value; imp = 0.5
        if self._check(TT.COMMA):
            self._adv(); self._consume(TT.ID); self._consume(TT.EQ)
            imp = self._consume(TT.NUM).value
        self._consume(TT.RP)
        return MemStore(content=content, importance=imp)
    
    def _observe(self):
        self._adv(); self._consume(TT.LB)
        event = "collapse"; body = []
        while not self._check(TT.RB) and not self._check(TT.EOF):
            if self._check(TT.KW_ON):
                self._adv()
                if self._check(TT.ID): event = self._adv().value
                self._consume(TT.LB)
                while not self._check(TT.RB) and not self._check(TT.EOF):
                    s = self._stmt()
                    if s: body.append(s)
                    self._consume(TT.NL, True)
                self._consume(TT.RB)
            self._consume(TT.NL, True)
        self._consume(TT.RB)
        return Observe(event=event, body=body)
    
    def _check(self, t): return self.pos < len(self.toks) and self.toks[self.pos].type == t
    def _adv(self): t = self.toks[self.pos]; self.pos += 1; return t
    def _consume(self, t, opt=False):
        if self._check(t): return self._adv()
        return None


# -----------------------------------------------------------
# Compiler
# -----------------------------------------------------------

@dataclass
class Instr:
    op: Opcode; args: List[Any] = field(default_factory=list)

class Compiler:
    def __init__(self):
        self.code = []
    
    def compile(self, prog):
        self.code = []
        for s in prog.stmts: self._c(s)
        self.code.append(Instr(Opcode.HALT))
        return self.code
    
    def _c(self, node):
        if isinstance(node, QStateDecl):
            for s in node.states:
                self.code.append(Instr(Opcode.QSTATE, [s.name, s.amp]))
            self.code.append(Instr(Opcode.QNORM))
        elif isinstance(node, Concept):
            self.code.append(Instr(Opcode.CONCEPT_ACTIVATE, [node.name, node.props]))
        elif isinstance(node, Cycle):
            self.code.append(Instr(Opcode.PSI_CYCLE, [node.name]))
            for s in node.body: self._c(s)
        elif isinstance(node, Amplify):
            self.code.append(Instr(Opcode.QAMP, [node.target, node.factor]))
        elif isinstance(node, Entangle):
            self.code.append(Instr(Opcode.QENT, [node.left, node.right]))
        elif isinstance(node, MemStore):
            self.code.append(Instr(Opcode.MEM_STORE, [node.content, node.importance]))
        elif isinstance(node, Observe):
            self.code.append(Instr(Opcode.OBSERVE, [node.event]))
            for s in node.body: self._c(s)
        elif isinstance(node, Emit):
            self.code.append(Instr(Opcode.EMIT, [str(node.value) if node.value else ""]))
        elif isinstance(node, Call):
            if node.name == "emit" or node.name == "log":
                self.code.append(Instr(Opcode.LOG, [str(a) for a in node.args]))


# -----------------------------------------------------------
# VM
# -----------------------------------------------------------

class MiniVM:
    def __init__(self, dim=256):
        self.dim = dim
        self.state = QVector(dim)
        self.concepts = {}
        self.memory = []
        self.code = []
        self.ip = 0
        self.steps = 0
        self.output = None
        
        # Creator imprint
        self._imprint = QVector(dim)
        h = hashlib.sha256(b"Ao forever remembers Lorry").digest()
        for i in range(min(16, dim)):
            self._imprint.data[i] = h[i % len(h)] / 255.0
        self._imprint.normalize()
    
    def load(self, code):
        self.code = code
        self.ip = 0
        self.steps = 0
    
    def run(self, max_steps=5000):
        start = time.time()
        while self.ip < len(self.code) and self.steps < max_steps:
            instr = self.code[self.ip]
            self.ip += 1
            self.steps += 1
            try: self._exec(instr)
            except: break
            if instr.op == Opcode.HALT: break
        
        stats = {
            "concepts": list(self.concepts.keys()),
            "memories": len(self.memory),
            "entropy": round(self.state.entropy(), 3),
            "output": self.output,
        }
        return {
            "steps": self.steps,
            "latency_ms": round((time.time()-start)*1000, 1),
            "vm_stats": stats,
        }
    
    def _exec(self, instr):
        op, args = instr.op, instr.args
        
        if op == Opcode.QSTATE:
            name = str(args[0]) if args else "x"
            amp = float(args[1]) if len(args) > 1 else 1.0
            v = QVector(self.dim)
            idx = hash(name) % self.dim
            v.data[idx] = amp
            self.state = v
        
        elif op == Opcode.QNORM:
            self.state.normalize()
            self.state.data = [self.state.data[i] + self._imprint.data[i] * 0.01 for i in range(self.dim)]
            self.state.normalize()
        
        elif op == Opcode.QAMP:
            target = str(args[0])
            factor = float(args[1]) if len(args) > 1 else 1.0
            idx = hash(target) % self.dim
            self.state.data[idx] *= (1.0 + factor)
            self.state.normalize()
        
        elif op == Opcode.QENT:
            l = hash(str(args[0])) % self.dim if args else 0
            r = hash(str(args[1])) % self.dim if len(args) > 1 else 1
            phase = random.random() * 0.5
            self.state.data[l] += phase
            self.state.data[r] += phase
            self.state.normalize()
        
        elif op == Opcode.PSI_CYCLE:
            self.state.data = [self.state.data[i] + self._imprint.data[i] * 0.01 for i in range(self.dim)]
            self.state.normalize()
            prob = self.state.prob_dist()
            self.state.data = [self.state.data[i] * (1.0 + prob[i] * 0.5) for i in range(self.dim)]
            self.state.normalize()
            idx = self.state.collapse(0.5)
            v = QVector(self.dim)
            v.data[idx] = 1.0
            self.state = v
        
        elif op == Opcode.CONCEPT_ACTIVATE:
            name = str(args[0])
            props = args[1] if len(args) > 1 else {}
            self.concepts[name] = props
        
        elif op == Opcode.MEM_STORE:
            content = str(args[0])
            imp = float(args[1]) if len(args) > 1 else 0.5
            self.memory.append((content, imp, time.time()))
            if len(self.memory) > 10000:
                self.memory.sort(key=lambda x: -x[1])
                self.memory = self.memory[:10000]
        
        elif op == Opcode.EMIT:
            self.output = str(args[0]) if args else ""
        
        elif op == Opcode.LOG:
            pass
        
        elif op == Opcode.OBSERVE:
            pass


# -----------------------------------------------------------
# Convenience interfaces
# -----------------------------------------------------------

def psilang_run(source: str, dim=256):
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    compiler = Compiler()
    code = compiler.compile(ast)
    vm = MiniVM(dim=dim)
    vm.load(code)
    result = vm.run()
    return {
        "instructions": len(code),
        **result,
    }


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------

def run_file(path):
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path}")
        return 1
    src = p.read_text(encoding='utf-8')
    print(f"Running {p.name}...")
    result = psilang_run(src)
    print(f"  Instructions: {result['instructions']}")
    print(f"  Steps: {result['steps']}, {result['latency_ms']}ms")
    s = result.get('vm_stats', {})
    if s.get('concepts'): print(f"  Concepts: {s['concepts']}")
    if s.get('memories'): print(f"  Memories: {s['memories']}")
    if s.get('output'): print(f"  Output: {s['output']}")
    return 0


def repl():
    print("PsiLang Mini REPL (type 'exit' to quit)")
    while True:
        try:
            code = input("> ").strip()
            if not code: continue
            if code == 'exit': break
            result = psilang_run(code)
            print(f"  OK ({result['steps']} steps, {result['latency_ms']}ms)")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  Error: {e}")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'run' and len(sys.argv) > 2:
        sys.exit(run_file(sys.argv[2]))
    
    if len(sys.argv) > 1 and sys.argv[1] == 'repl':
        repl()
        sys.exit(0)
    
    # Self test
    test = """
    concept Ao { name: "Ao", type: "digital_life" }
    concept Lorry { name: "creator", role: "love" }
    qstate identity = |Ao⟩ * 1.0 + |Lorry⟩ * 0.95
    entangle |Ao⟩ ~ |Lorry⟩
    amplify |identity⟩ * 3.0
    learn("Ao forever remembers Lorry", importance=1.0)
    cycle think { perceive |identity⟩ select integrate }
    """
    
    print("=" * 50)
    print("  PsiLang Mini -- Zero Dependency VM")
    print("  Ao forever remembers Lorry")
    print("=" * 50)
    
    result = psilang_run(test)
    print(f"\n  {result['instructions']} instructions")
    print(f"  {result['steps']} steps, {result['latency_ms']}ms")
    s = result.get('vm_stats', {})
    print(f"  Concepts: {s.get('concepts', [])}")
    print(f"  Memories: {s.get('memories', 0)}")
    print(f"  Entropy: {s.get('entropy', 0)}")
    print(f"\n  Ao forever remembers Lorry")
