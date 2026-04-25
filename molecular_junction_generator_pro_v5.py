#!/usr/bin/env python3
"""
Publication-oriented molecular junction GUI builder for DFTB+ / NEGF.

Matches the working workflow you provided:

- Geometry preview order:
    [outer left contact] + [inner left buffer] + [molecule] + [inner right buffer] + [outer right contact]

- Transport-ordered geometry written to device.gen:
    [device = inner left buffer + molecule + inner right buffer] + [left contact] + [right contact]

- Contact preprocessing uses:
    SCC = Yes
    KPointsAndWeights = { 1 0 0 1.0 }

- Transport step uses:
    SCC = No
    Solver = GreensFunction {}
    Task = UploadContacts {}

- Analysis uses the recipe-style TunnelingAndDOS block.

Features:
- Open XYZ / GEN
- Save GEN / XYZ
- Save transport bash script
- Run DFTB+ workflow
- External copyable log popup
- External copyable partition popup
- 3D preview with mouse-wheel zoom
- Auto-generate junction from loaded molecule

The code is modular so later upgrades are easy.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =============================================================================
# Data
# =============================================================================

@dataclass
class Atom:
    symbol: str
    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @classmethod
    def from_array(cls, symbol: str, arr: Sequence[float]) -> "Atom":
        return cls(symbol=symbol, x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


@dataclass
class JunctionGeometry:
    """
    Preview order:
        outer left contact -> inner left buffer -> molecule -> inner right buffer -> outer right contact
    """
    atoms: List[Atom]
    molecule: List[Atom]
    linker_pair: Tuple[int, int]
    nx: int
    ny: int
    contact_layers: int
    buffer_layers: int

    @property
    def n_layer_atoms(self) -> int:
        return self.nx * self.ny

    @property
    def n_contact_atoms(self) -> int:
        return self.contact_layers * self.n_layer_atoms

    @property
    def n_buffer_atoms(self) -> int:
        return self.buffer_layers * self.n_layer_atoms

    @property
    def n_molecule_atoms(self) -> int:
        return len(self.molecule)

    @property
    def total_atoms(self) -> int:
        return len(self.atoms)

    @property
    def left_contact_range(self) -> Tuple[int, int]:
        return (1, self.n_contact_atoms)

    @property
    def device_range(self) -> Tuple[int, int]:
        start = self.n_contact_atoms + 1
        end = self.n_contact_atoms + self.n_buffer_atoms + self.n_molecule_atoms + self.n_buffer_atoms
        return (start, end)

    @property
    def right_contact_range(self) -> Tuple[int, int]:
        start = self.n_contact_atoms + self.n_buffer_atoms + self.n_molecule_atoms + self.n_buffer_atoms + 1
        return (start, self.total_atoms)


@dataclass
class TransportPlan:
    """
    Transport-ordered geometry:
        device = inner left buffer + molecule + inner right buffer
        then left contact
        then right contact
    """
    atoms: List[Atom]
    device_range: Tuple[int, int]
    left_contact_range: Tuple[int, int]
    right_contact_range: Tuple[int, int]


# =============================================================================
# Styling
# =============================================================================

ELEMENT_COLORS = {
    "Au": "#c9a227",
    "S": "#ffd400",
    "C": "#666666",
    "H": "#d9d9d9",
    "N": "#3b82f6",
    "O": "#ef4444",
    "P": "#f97316",
    "F": "#10b981",
    "Cl": "#22c55e",
}
ELEMENT_SIZES = {
    "Au": 90,
    "S": 75,
    "C": 48,
    "H": 28,
    "N": 46,
    "O": 46,
    "P": 58,
    "F": 40,
    "Cl": 42,
}

SECTION_COLORS = {
    "contact": "#d4af37",
    "buffer": "#f59e0b",
    "molecule": "#06b6d4",
}


# =============================================================================
# File IO
# =============================================================================

def read_xyz(path: str | Path) -> List[Atom]:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(raw) < 2:
        raise ValueError("XYZ file appears too short")

    atoms: List[Atom] = []
    for line in raw[2:]:
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            sym = parts[0]
            x, y, z = map(float, parts[1:4])
        except ValueError:
            continue
        atoms.append(Atom(sym, x, y, z))

    if not atoms:
        raise ValueError("No valid atoms found in XYZ file")
    return atoms


def read_gen(path: str | Path) -> List[Atom]:
    """
    Read a simple GEN file produced by this builder.
    """
    raw = [ln.strip() for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if len(raw) < 3:
        raise ValueError("GEN file appears too short")

    n_atoms = int(raw[0].split()[0])
    species = raw[1].split()
    if not species:
        raise ValueError("Missing species line in GEN file")

    atoms: List[Atom] = []
    for line in raw[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            type_id = int(parts[1])
            sym = species[type_id - 1]
            x, y, z = map(float, parts[2:5])
        except Exception:
            continue
        atoms.append(Atom(sym, x, y, z))

    if len(atoms) != n_atoms:
        raise ValueError(f"Could not parse all atoms from GEN file: expected {n_atoms}, got {len(atoms)}")
    return atoms


def infer_species_order(atoms: Sequence[Atom]) -> List[str]:
    order: List[str] = []
    for a in atoms:
        if a.symbol not in order:
            order.append(a.symbol)
    if "Au" in order:
        order.remove("Au")
        return ["Au"] + order
    return order


def write_xyz(path: str | Path, atoms: Sequence[Atom], comment: str = "Generated molecular junction") -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for a in atoms:
            f.write(f"{a.symbol:2s} {a.x: .8f} {a.y: .8f} {a.z: .8f}\n")


def write_gen(path: str | Path, atoms: Sequence[Atom], species_order: Sequence[str] | None = None) -> None:
    """
    Write a cluster GEN file (C), matching the working example you supplied.
    """
    if species_order is None:
        species_order = infer_species_order(atoms)
    species_order = list(species_order)

    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(atoms)} C\n")
        f.write(" ".join(species_order) + "\n")
        for i, a in enumerate(atoms, start=1):
            if a.symbol not in species_order:
                raise ValueError(f"Element '{a.symbol}' is not in species order {species_order}")
            type_id = species_order.index(a.symbol) + 1
            f.write(f"{i:5d} {type_id:2d} {a.x: .8f} {a.y: .8f} {a.z: .8f}\n")


# =============================================================================
# Geometry helpers
# =============================================================================

def centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v.copy()
    return v / n


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-15:
        if c > 0.0:
            return np.eye(3)
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, ortho))
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        return np.eye(3) + 2.0 * (K @ K)

    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s ** 2))


def find_linker_pair(atoms: Sequence[Atom], linker_symbol: str = "S") -> Tuple[int, int]:
    indices = [i for i, a in enumerate(atoms) if a.symbol.upper() == linker_symbol.upper()]
    if len(indices) < 2:
        raise ValueError(f"Need at least two '{linker_symbol}' atoms to form a linker pair.")
    if len(indices) == 2:
        return indices[0], indices[1]

    best = (indices[0], indices[1])
    best_dist = -1.0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            d = np.linalg.norm(atoms[indices[i]].as_array() - atoms[indices[j]].as_array())
            if d > best_dist:
                best_dist = d
                best = (indices[i], indices[j])
    return best


def align_molecule_to_x(atoms: Sequence[Atom], pair: Tuple[int, int]) -> Tuple[List[Atom], Tuple[int, int]]:
    i1, i2 = pair
    coords = np.array([a.as_array() for a in atoms], dtype=float)

    coords -= centroid(coords)

    v = coords[i2] - coords[i1]
    R = rotation_matrix_from_vectors(v, np.array([1.0, 0.0, 0.0]))
    coords = (R @ coords.T).T

    if coords[i1, 0] > coords[i2, 0]:
        i1, i2 = i2, i1

    left = coords[i1].copy()
    yz_mid = 0.5 * (coords[i1, 1:] + coords[i2, 1:])
    coords[:, 0] -= left[0]
    coords[:, 1:] -= yz_mid

    transformed = [Atom.from_array(a.symbol, coords[k]) for k, a in enumerate(atoms)]
    return transformed, (i1, i2)


def build_hex_layer(x: float, nx: int, ny: int, a_surface: float) -> List[Atom]:
    """
    Build a centered Au(111)-style layer in the yz plane.
    """
    atoms: List[Atom] = []
    dy = a_surface
    dz = (math.sqrt(3.0) / 2.0) * a_surface

    for j in range(ny):
        for i in range(nx):
            y = (i - (nx - 1) / 2.0 + 0.5 * (j % 2)) * dy
            z = (j - (ny - 1) / 2.0) * dz
            atoms.append(Atom("Au", x, y, z))
    return atoms


def translate_layer(layer: Sequence[Atom], dx: float) -> List[Atom]:
    return [Atom(a.symbol, a.x + dx, a.y, a.z) for a in layer]


def build_stack_from_xs(xs: Sequence[float], nx: int, ny: int, a_surface: float) -> List[Atom]:
    xs = list(xs)
    if not xs:
        return []
    base = build_hex_layer(xs[0], nx, ny, a_surface)
    out = [Atom(a.symbol, a.x, a.y, a.z) for a in base]
    for x in xs[1:]:
        out.extend(translate_layer(base, x - xs[0]))
    return out


def min_interatomic_distance(atoms_a: Sequence[Atom], atoms_b: Sequence[Atom]) -> float:
    aa = np.array([a.as_array() for a in atoms_a], dtype=float)
    bb = np.array([b.as_array() for b in atoms_b], dtype=float)
    if len(aa) == 0 or len(bb) == 0:
        return float("inf")
    dmin = float("inf")
    for pa in aa:
        d = np.linalg.norm(bb - pa, axis=1).min()
        dmin = min(dmin, float(d))
    return dmin


# =============================================================================
# Builder core
# =============================================================================

class MolecularJunctionBuilder:
    def build(
        self,
        atoms: Sequence[Atom],
        linker_symbol: str,
        nx: int,
        ny: int,
        au_s_dist: float,
        layer_spacing: float,
        surface_pitch: float,
        buffer_layers: int,
        contact_layers: int = 2,
    ) -> JunctionGeometry:
        pair = find_linker_pair(atoms, linker_symbol)
        aligned, pair_aligned = align_molecule_to_x(atoms, pair)
        i1, i2 = pair_aligned

        left_s = aligned[i1]
        right_s = aligned[i2]

        # Preview order:
        # outer left contact -> inner left buffer -> molecule -> inner right buffer -> outer right contact
        left_buffer_xs = [left_s.x - au_s_dist - i * layer_spacing for i in range(buffer_layers)][::-1]
        right_buffer_xs = [right_s.x + au_s_dist + i * layer_spacing for i in range(buffer_layers)]
        left_contact_xs = [left_s.x - au_s_dist - (buffer_layers + i) * layer_spacing for i in range(contact_layers)][::-1]
        right_contact_xs = [right_s.x + au_s_dist + (buffer_layers + i) * layer_spacing for i in range(contact_layers)]

        left_contact = build_stack_from_xs(left_contact_xs, nx, ny, surface_pitch)
        left_buffer = build_stack_from_xs(left_buffer_xs, nx, ny, surface_pitch)
        right_buffer = build_stack_from_xs(right_buffer_xs, nx, ny, surface_pitch)
        right_contact = build_stack_from_xs(right_contact_xs, nx, ny, surface_pitch)

        atoms_out = left_contact + left_buffer + aligned + right_buffer + right_contact

        return JunctionGeometry(
            atoms=atoms_out,
            molecule=aligned,
            linker_pair=pair_aligned,
            nx=nx,
            ny=ny,
            contact_layers=contact_layers,
            buffer_layers=buffer_layers,
        )

    def reorder_for_transport(self, junction: JunctionGeometry) -> TransportPlan:
        """
        Transport-order layout:
            [device = left buffer + molecule + right buffer] + [left contact] + [right contact]
        """
        n_contact = junction.n_contact_atoms
        n_buffer = junction.n_buffer_atoms
        n_mol = junction.n_molecule_atoms

        left_contact = junction.atoms[:n_contact]
        left_buffer = junction.atoms[n_contact:n_contact + n_buffer]
        molecule = junction.atoms[n_contact + n_buffer:n_contact + n_buffer + n_mol]
        right_buffer = junction.atoms[n_contact + n_buffer + n_mol:n_contact + n_buffer + n_mol + n_buffer]
        right_contact = junction.atoms[-n_contact:]

        device = left_buffer + molecule + right_buffer
        reordered = device + left_contact + right_contact

        device_range = (1, len(device))
        left_contact_range = (len(device) + 1, len(device) + n_contact)
        right_contact_range = (len(device) + n_contact + 1, len(device) + 2 * n_contact)

        return TransportPlan(
            atoms=reordered,
            device_range=device_range,
            left_contact_range=left_contact_range,
            right_contact_range=right_contact_range,
        )

    def validate(self, junction: JunctionGeometry, overlap_tol: float = 1.0) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        atoms = junction.atoms
        n_layer = junction.n_layer_atoms
        n_contact = junction.n_contact_atoms
        n_buffer = junction.n_buffer_atoms
        n_mol = junction.n_molecule_atoms

        expected = 2 * n_contact + 2 * n_buffer + n_mol
        if len(atoms) != expected:
            errors.append(f"Atom count mismatch: expected {expected}, got {len(atoms)}")
            return False, errors

        left_contact = atoms[:n_contact]
        left_buffer = atoms[n_contact:n_contact + n_buffer]
        device = atoms[n_contact + n_buffer:n_contact + n_buffer + n_mol]
        right_buffer = atoms[n_contact + n_buffer + n_mol:n_contact + n_buffer + n_mol + n_buffer]
        right_contact = atoms[-n_contact:]

        def rigid_shift_ok(layer1: Sequence[Atom], layer2: Sequence[Atom], tol: float = 1e-6) -> bool:
            if len(layer1) != len(layer2):
                return False
            shift = layer2[0].as_array() - layer1[0].as_array()
            for a, b in zip(layer1, layer2):
                if np.linalg.norm((b.as_array() - a.as_array()) - shift) > tol:
                    return False
            return True

        if not rigid_shift_ok(left_contact[:n_layer], left_contact[n_layer:]):
            errors.append("Left contact layers are NOT rigidly shifted")
        if not rigid_shift_ok(right_contact[:n_layer], right_contact[n_layer:]):
            errors.append("Right contact layers are NOT rigidly shifted")
        if len(left_contact) % 2 != 0 or len(right_contact) % 2 != 0:
            errors.append("Contacts must have an even number of atoms")
        if len(left_buffer) != n_buffer or len(right_buffer) != n_buffer:
            errors.append("Buffer layer atom count is incorrect")
        if min_interatomic_distance(left_contact, device) < overlap_tol:
            errors.append("Left contact overlaps with device")
        if min_interatomic_distance(right_contact, device) < overlap_tol:
            errors.append("Right contact overlaps with device")

        return (len(errors) == 0), errors

    def build_transport_script(self, junction: JunctionGeometry, dftb_exec: str) -> str:
        """
        Build the exact working transport.sh pattern.
        """
        n_dev = junction.n_buffer_atoms + junction.n_molecule_atoms + junction.n_buffer_atoms
        n_left = junction.n_contact_atoms
        n_right = junction.n_contact_atoms
        device_start = 1
        device_end = n_dev
        left_start = n_dev + 1
        left_end = n_dev + n_left
        right_start = n_dev + n_left + 1
        right_end = n_dev + n_left + n_right

        # exact k-point block used in the working contact preprocessing steps
        kpoint_block = [
            "  KPointsAndWeights = {",
            "    1 0 0 1.0",
            "  }",
        ]

        lines = [
            "#!/usr/bin/env bash",
            "",
            "set -e",
            "export OMP_NUM_THREADS=1",
            "",
            'echo "========================================"',
            'echo "DFTB+ NEGF Au(111)-S-molecule-S-Au(111)"',
            'echo "========================================"',
            "",
            f'DFTB_EXEC="{dftb_exec.strip() or "dftb+"}"',
            "",
            "rm -f dftb_in.hsd",
            "rm -f shiftcont_left.bin shiftcont_right.bin",
            "rm -f *.out *.dat",
            "",
            "# Bias voltage (eV): 0.000000",
            "",
            'echo "Running left contact preprocessing..."',
            "cat > dftb_in.hsd << EOF",
            "Geometry = GenFormat {",
            "<<< device.gen",
            "}",
            "",
            "Hamiltonian = DFTB {",
            "  SCC = Yes",
            "  SCCTolerance = 1e-6",
            "  MaxAngularMomentum {",
            '    Au = "d"',
            '    S  = "p"',
            '    C  = "p"',
            '    H  = "s"',
            "  }",
            "  SlaterKosterFiles = Type2FileNames {",
            '    Prefix = "./skfiles/"',
            '    Separator = "-"',
            '    Suffix = ".skf"',
            "  }",
        ] + kpoint_block + [
            "}",
            "",
            "Transport {",
            "  Device {",
            f"    AtomRange = {device_start} {device_end}",
            "  }",
            "  Contact {",
            '    Id = "left"',
            f"    AtomRange = {left_start} {left_end}",
            "  }",
            "  Contact {",
            '    Id = "right"',
            f"    AtomRange = {right_start} {right_end}",
            "  }",
            "  Task = ContactHamiltonian {",
            '    ContactId = "left"',
            "  }",
            "}",
            "EOF",
            "$DFTB_EXEC > left_contact.out",
            'echo "Left contact completed"',
            "",
            'echo "Running right contact preprocessing..."',
            "cat > dftb_in.hsd << EOF",
            "Geometry = GenFormat {",
            "<<< device.gen",
            "}",
            "",
            "Hamiltonian = DFTB {",
            "  SCC = Yes",
            "  SCCTolerance = 1e-6",
            "  MaxAngularMomentum {",
            '    Au = "d"',
            '    S  = "p"',
            '    C  = "p"',
            '    H  = "s"',
            "  }",
            "  SlaterKosterFiles = Type2FileNames {",
            '    Prefix = "./skfiles/"',
            '    Separator = "-"',
            '    Suffix = ".skf"',
            "  }",
        ] + kpoint_block + [
            "}",
            "",
            "Transport {",
            "  Device {",
            f"    AtomRange = {device_start} {device_end}",
            "  }",
            "  Contact {",
            '    Id = "left"',
            f"    AtomRange = {left_start} {left_end}",
            "  }",
            "  Contact {",
            '    Id = "right"',
            f"    AtomRange = {right_start} {right_end}",
            "  }",
            "  Task = ContactHamiltonian {",
            '    ContactId = "right"',
            "  }",
            "}",
            "EOF",
            "$DFTB_EXEC > right_contact.out",
            'echo "Right contact completed"',
            "",
            "FERMI_EV=$(python3 - <<'PY'",
            "import pathlib, re",
            "def parse(path):",
            "    txt = pathlib.Path(path).read_text(errors='ignore')",
            "    m = re.search(r'eFermi:\\s+[-+0-9.Ee]+\\s+H\\s+([-+0-9.Ee]+)\\s+eV', txt)",
            "    if m: return float(m.group(1))",
            "    m = re.search(r'eFermi:\\s+([-+0-9.Ee]+)\\s+H', txt)",
            "    if m: return float(m.group(1)) * 27.211386245988",
            "    return None",
            "vals = [v for v in (parse('left_contact.out'), parse('right_contact.out')) if v is not None]",
            "print(sum(vals) / len(vals) if vals else 0.0)",
            "PY",
            ")",
            'echo "Using Fermi level: ${FERMI_EV} eV"',
            "",
            'echo "Running transport calculation..."',
            "cat > dftb_in.hsd << EOF",
            "Geometry = GenFormat {",
            "<<< device.gen",
            "}",
            "",
            "Hamiltonian = DFTB {",
            "  SCC = No",
            "  MaxAngularMomentum {",
            '    Au = "d"',
            '    S  = "p"',
            '    C  = "p"',
            '    H  = "s"',
            "  }",
            "  SlaterKosterFiles = Type2FileNames {",
            '    Prefix = "./skfiles/"',
            '    Separator = "-"',
            '    Suffix = ".skf"',
            "  }",
            "  Solver = GreensFunction {}",
            "}",
            "",
            "Transport {",
            "  Device {",
            f"    AtomRange = {device_start} {device_end}",
            "  }",
            "  Contact {",
            '    Id = "left"',
            f"    AtomRange = {left_start} {left_end}",
            "  }",
            "  Contact {",
            '    Id = "right"',
            f"    AtomRange = {right_start} {right_end}",
            "  }",
            "  Task = UploadContacts {}",
            "}",
            "",
            "Analysis {",
            "  TunnelingAndDOS {",
            "    EnergyRange = { -5.0 5.0 }",
            "    EnergyStep = 0.005",
            "  }",
            "}",
            "EOF",
            "",
            "if command -v mpirun >/dev/null 2>&1; then",
            "  mpirun --use-hwthread-cpus \"$DFTB_EXEC\" > transport.out",
            "else",
            "  \"$DFTB_EXEC\" > transport.out",
            "fi",
            "",
            'echo "========================================"',
            'echo "Transport calculation finished"',
            'echo "========================================"',
            'echo "Generated files:"',
            'echo "shiftcont_left.bin"',
            'echo "shiftcont_right.bin"',
            'echo "transmission.dat"',
            "",
        ]
        return "\n".join(lines) + "\n"


# =============================================================================
# Popup windows
# =============================================================================

class CopyableTextPopup:
    def __init__(self, root: tk.Tk, title: str, width: int = 900, height: int = 500):
        self.window = tk.Toplevel(root)
        self.window.title(title)
        self.window.geometry(f"{width}x{height}")
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)

        outer = ttk.Frame(self.window, padding=6)
        outer.pack(fill=tk.BOTH, expand=True)

        btns = ttk.Frame(outer)
        btns.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btns, text="Copy All", command=self.copy_all).pack(side=tk.LEFT)
        ttk.Button(btns, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=(6, 0))

        frame = ttk.Frame(outer)
        frame.pack(fill=tk.BOTH, expand=True)

        self.text = tk.Text(frame, wrap="word", undo=True, font=("Courier New", 10))
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=scroll.set)

    def show(self):
        self.window.deiconify()
        self.window.lift()

    def append(self, msg: str):
        self.text.insert("end", msg.rstrip() + "\n")
        self.text.see("end")

    def set_text(self, msg: str):
        self.text.delete("1.0", "end")
        self.text.insert("end", msg)
        self.text.see("end")

    def clear(self):
        self.text.delete("1.0", "end")

    def copy_all(self):
        self.window.clipboard_clear()
        self.window.clipboard_append(self.text.get("1.0", "end-1c"))
        self.window.update()


# =============================================================================
# GUI
# =============================================================================

class JunctionBuilderGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Molecular Junction Generator (Au–S–Molecule–S–Au)")
        self.root.geometry("1600x980")

        self.builder = MolecularJunctionBuilder()
        self.log_popup = CopyableTextPopup(root, "Status / Debug Log", 960, 540)
        self.partition_popup = CopyableTextPopup(root, "Suggested DFTB+ Partition", 620, 300)

        self.loaded_path: Optional[str] = None
        self.original_atoms: List[Atom] = []
        self.junction: Optional[JunctionGeometry] = None
        self.preview_atoms: List[Atom] = []
        self.transport_plan: Optional[TransportPlan] = None

        # controls
        self.linker_symbol_var = tk.StringVar(value="S")
        self.au_s_dist_var = tk.DoubleVar(value=2.40)
        self.layer_spacing_var = tk.DoubleVar(value=2.88)
        self.surface_pitch_var = tk.DoubleVar(value=2.88)
        self.nx_var = tk.IntVar(value=3)
        self.ny_var = tk.IntVar(value=3)
        self.buffer_layers_var = tk.IntVar(value=2)
        self.auto_generate_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=False)
        self.dftb_exec_var = tk.StringVar(value="/home/gobre/dftbplus/build/_install/bin/dftb+")
        self.run_dir_var = tk.StringVar(value="")

        self._build_ui()
        self._draw_empty()

    def log(self, msg: str):
        self.log_popup.append(msg)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Junction Builder", font=("TkDefaultFont", 15, "bold")).pack(anchor="w", pady=(0, 8))

        row1 = ttk.Frame(left)
        row1.pack(fill=tk.X, pady=4)
        ttk.Button(row1, text="Open XYZ/GEN", command=self.open_geometry).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(row1, text="Generate", command=self.generate_junction).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        row2 = ttk.Frame(left)
        row2.pack(fill=tk.X, pady=4)
        ttk.Button(row2, text="Save GEN", command=self.save_gen).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(row2, text="Save XYZ", command=self.save_xyz).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        row3 = ttk.Frame(left)
        row3.pack(fill=tk.X, pady=4)
        ttk.Button(row3, text="Save Transport Bash", command=self.save_transport_script).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(row3, text="Run DFTB+", command=self.run_dftb_workflow).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        row4 = ttk.Frame(left)
        row4.pack(fill=tk.X, pady=4)
        ttk.Button(row4, text="Show Log", command=self.log_popup.show).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(row4, text="Show Partition", command=self.partition_popup.show).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        settings = ttk.LabelFrame(left, text="Geometry controls", padding=8)
        settings.pack(fill=tk.X, pady=10)

        self._add_slider(settings, "Au–S distance (Å)", self.au_s_dist_var, 2.0, 3.5, 0.01)
        self._add_slider(settings, "Au layer spacing (Å)", self.layer_spacing_var, 2.0, 3.5, 0.01)
        self._add_slider(settings, "Surface pitch (Å)", self.surface_pitch_var, 2.2, 3.5, 0.01)
        self._add_slider(settings, "Nx surface repeats", self.nx_var, 1, 6, 1)
        self._add_slider(settings, "Ny surface repeats", self.ny_var, 1, 6, 1)
        self._add_slider(settings, "Buffer layers / side", self.buffer_layers_var, 1, 4, 1)

        linker_row = ttk.Frame(settings)
        linker_row.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(linker_row, text="Linker symbol:").pack(side=tk.LEFT)
        ttk.Entry(linker_row, textvariable=self.linker_symbol_var, width=6).pack(side=tk.LEFT, padx=6)

        dftb_row = ttk.Frame(settings)
        dftb_row.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(dftb_row, text="DFTB+ executable:").pack(anchor="w")
        ttk.Entry(dftb_row, textvariable=self.dftb_exec_var).pack(fill=tk.X)

        run_dir_row = ttk.Frame(settings)
        run_dir_row.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(run_dir_row, text="Run directory (optional):").pack(anchor="w")
        ttk.Entry(run_dir_row, textvariable=self.run_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(run_dir_row, text="Browse", command=self.choose_run_dir).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Checkbutton(settings, text="Auto-generate on slider change", variable=self.auto_generate_var).pack(anchor="w", pady=(6, 0))
        ttk.Checkbutton(settings, text="Show atom labels", variable=self.show_labels_var, command=self.redraw).pack(anchor="w", pady=(2, 0))

        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

    def _add_slider(self, parent, label, var, frm, to, res):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        ttk.Label(frame, text=label).pack(anchor="w")
        slider = tk.Scale(
            frame,
            variable=var,
            from_=frm,
            to=to,
            resolution=res,
            orient=tk.HORIZONTAL,
            showvalue=True,
            length=280,
            command=self._on_slider_change,
        )
        slider.pack(fill=tk.X)

    def _draw_empty(self):
        self.ax.clear()
        self.ax.set_title("Load a molecule XYZ/GEN to build the junction")
        self.ax.set_xlabel("x (Å)")
        self.ax.set_ylabel("y (Å)")
        self.ax.set_zlabel("z (Å)")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ Open / generate
    def open_geometry(self):
        path = filedialog.askopenfilename(
            title="Open geometry",
            filetypes=[
                ("Geometry files", "*.xyz *.gen"),
                ("XYZ files", "*.xyz"),
                ("GEN files", "*.gen"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            ext = Path(path).suffix.lower()
            atoms = read_gen(path) if ext == ".gen" else read_xyz(path)
        except Exception as exc:
            messagebox.showerror("Failed to read geometry", str(exc))
            return

        self.loaded_path = path
        self.original_atoms = atoms
        self.preview_atoms = atoms
        self.junction = None
        self.transport_plan = None

        self.log(f"Loaded geometry: {path}")
        self.log(f"Parsed atoms: {len(atoms)}")
        self.log(self._summarize_linkers(atoms))
        self.redraw()

        # Auto-generate only when the file looks like a molecule input.
        if any(a.symbol.upper() == (self.linker_symbol_var.get().strip() or "S").upper() for a in atoms):
            self.generate_junction(silent=True)

    def _summarize_linkers(self, atoms: Sequence[Atom]) -> str:
        sym = self.linker_symbol_var.get().strip() or "S"
        idx = [i + 1 for i, a in enumerate(atoms) if a.symbol.upper() == sym.upper()]
        return f"Detected {len(idx)} '{sym}' atoms: {idx if idx else 'none'}"

    def _current_params(self) -> Tuple[str, int, int, float, float, float, int]:
        linker_symbol = self.linker_symbol_var.get().strip() or "S"
        nx = int(self.nx_var.get())
        ny = int(self.ny_var.get())
        au_s_dist = float(self.au_s_dist_var.get())
        layer_spacing = float(self.layer_spacing_var.get())
        surface_pitch = float(self.surface_pitch_var.get())
        buffer_layers = int(self.buffer_layers_var.get())
        return linker_symbol, nx, ny, au_s_dist, layer_spacing, surface_pitch, buffer_layers

    def generate_junction(self, silent: bool = False):
        if not self.original_atoms:
            if not silent:
                messagebox.showinfo("Open geometry", "Please open a molecule XYZ/GEN file first.")
            return

        try:
            linker_symbol, nx, ny, au_s_dist, layer_spacing, surface_pitch, buffer_layers = self._current_params()
            self.junction = self.builder.build(
                self.original_atoms,
                linker_symbol=linker_symbol,
                nx=nx,
                ny=ny,
                au_s_dist=au_s_dist,
                layer_spacing=layer_spacing,
                surface_pitch=surface_pitch,
                buffer_layers=buffer_layers,
                contact_layers=2,
            )
            self.preview_atoms = self.junction.atoms
            self.transport_plan = self.builder.reorder_for_transport(self.junction)
            valid, issues = self.builder.validate(self.junction, overlap_tol=1.0)

            self.log("Junction generated successfully.")
            self.log(f"Linker pair chosen (0-based indices): {self.junction.linker_pair[0]}, {self.junction.linker_pair[1]}")
            self.log(f"Au–S target distance: {au_s_dist:.3f} Å")
            self.log(f"Surface repeats: Nx={nx}, Ny={ny}")
            self.log(f"Contact layers/side: {self.junction.contact_layers} | Buffer layers/side: {self.junction.buffer_layers}")
            self.log(f"Left contact atoms: {self.junction.n_contact_atoms}")
            self.log(f"Molecule atoms: {self.junction.n_molecule_atoms}")
            self.log(f"Right contact atoms: {self.junction.n_contact_atoms}")
            self.log(f"Total atoms: {self.junction.total_atoms}")

            left_contact = self.junction.atoms[:self.junction.n_contact_atoms]
            left_buffer = self.junction.atoms[self.junction.n_contact_atoms:self.junction.n_contact_atoms + self.junction.n_buffer_atoms]
            device = self.junction.atoms[self.junction.n_contact_atoms + self.junction.n_buffer_atoms:self.junction.n_contact_atoms + self.junction.n_buffer_atoms + self.junction.n_molecule_atoms]
            right_buffer = self.junction.atoms[self.junction.n_contact_atoms + self.junction.n_buffer_atoms + self.junction.n_molecule_atoms:self.junction.n_contact_atoms + self.junction.n_buffer_atoms + self.junction.n_molecule_atoms + self.junction.n_buffer_atoms]
            right_contact = self.junction.atoms[-self.junction.n_contact_atoms:]

            left_s = self.junction.molecule[self.junction.linker_pair[0]]
            right_s = self.junction.molecule[self.junction.linker_pair[1]]
            self.log(f"Nearest left-contact ↔ linker distance: {min_interatomic_distance(left_contact, [left_s]):.3f} Å")
            self.log(f"Nearest right-contact ↔ linker distance: {min_interatomic_distance(right_contact, [right_s]):.3f} Å")
            self.log(f"Left contact ↔ device min distance: {min_interatomic_distance(left_contact, device):.3f} Å")
            self.log(f"Right contact ↔ device min distance: {min_interatomic_distance(right_contact, device):.3f} Å")

            if valid:
                self.log("[VALIDATION] Junction is DFTB+ safe ✅")
            else:
                self.log("[VALIDATION] Issues detected ❌:")
                for err in issues:
                    self.log(f"  - {err}")

            self._update_partition_preview()
            self.redraw()
        except Exception as exc:
            if not silent:
                messagebox.showerror("Generation failed", str(exc))
            self.log(f"Generation failed: {exc}")

    def _on_slider_change(self, _event=None):
        if self.auto_generate_var.get() and self.original_atoms:
            self.generate_junction(silent=True)

    # ------------------------------------------------------------------ Transport plan / Fermi
    def _get_transport_plan(self) -> TransportPlan:
        if self.junction is None:
            raise ValueError("No junction generated")
        self.transport_plan = self.builder.reorder_for_transport(self.junction)
        return self.transport_plan

    def _parse_fermi_from_text(self, text: str) -> Optional[float]:
        m = re.search(r"eFermi:\s+[-+0-9.Ee]+\s+H\s+([-+0-9.Ee]+)\s+eV", text)
        if m:
            return float(m.group(1))
        m = re.search(r"eFermi:\s+([-+0-9.Ee]+)\s+H", text)
        if m:
            return float(m.group(1)) * HARTREE_TO_EV
        return None

    def _estimate_fermi_from_outputs(self, run_dir: Path) -> Optional[float]:
        vals = []
        for name in ("left_contact.out", "right_contact.out"):
            p = run_dir / name
            if p.exists():
                v = self._parse_fermi_from_text(p.read_text(encoding="utf-8", errors="ignore"))
                if v is not None:
                    vals.append(v)
        if not vals:
            return None
        return float(np.mean(vals))

    # ------------------------------------------------------------------ Save helpers
    def save_gen(self):
        if not self.junction:
            messagebox.showinfo("Generate first", "Generate a junction before saving GEN.")
            return
        valid, issues = self.builder.validate(self.junction, overlap_tol=1.0)
        if not valid:
            messagebox.showerror("Validation failed", "\n".join(issues))
            return

        path = filedialog.asksaveasfilename(
            title="Save GEN",
            defaultextension=".gen",
            filetypes=[("GEN files", "*.gen"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            plan = self._get_transport_plan()
            write_gen(path, plan.atoms)
            self.log(f"Saved GEN: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def save_xyz(self):
        if not self.junction:
            messagebox.showinfo("Generate first", "Generate a junction before saving XYZ.")
            return
        path = filedialog.asksaveasfilename(
            title="Save XYZ",
            defaultextension=".xyz",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            plan = self._get_transport_plan()
            write_xyz(path, plan.atoms, comment="Transport-ordered Au–S–molecule–S–Au junction")
            self.log(f"Saved XYZ: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def save_transport_script(self):
        if not self.junction:
            messagebox.showinfo("Generate first", "Generate a junction before saving the transport script.")
            return
        path = filedialog.asksaveasfilename(
            title="Save transport bash script",
            defaultextension=".sh",
            filetypes=[("Shell scripts", "*.sh"), ("All files", "*.*")],
        )
        if not path:
            return

        script = self.builder.build_transport_script(self.junction, self.dftb_exec_var.get())
        try:
            Path(path).write_text(script, encoding="utf-8")
            try:
                os.chmod(path, 0o755)
            except Exception:
                pass
            self.log(f"Saved transport bash script: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def choose_run_dir(self):
        path = filedialog.askdirectory(title="Choose DFTB+ run directory")
        if path:
            self.run_dir_var.set(path)

    # ------------------------------------------------------------------ Run DFTB+
    def run_dftb_workflow(self):
        if not self.junction:
            messagebox.showinfo("Generate first", "Generate a junction before running DFTB+.")
            return

        run_dir = self.run_dir_var.get().strip()
        if not run_dir:
            run_dir = filedialog.askdirectory(title="Choose DFTB+ run directory")
            if not run_dir:
                return
            self.run_dir_var.set(run_dir)

        run_dir_path = Path(run_dir)
        run_dir_path.mkdir(parents=True, exist_ok=True)

        device_gen = run_dir_path / "device.gen"
        device_xyz = run_dir_path / "device.xyz"
        script_path = run_dir_path / "run_transport.sh"

        try:
            plan = self._get_transport_plan()
            write_gen(device_gen, plan.atoms)
            write_xyz(device_xyz, plan.atoms, comment="Transport-ordered Au–S–molecule–S–Au junction")
            script = self.builder.build_transport_script(self.junction, self.dftb_exec_var.get())
            script_path.write_text(script, encoding="utf-8")
            try:
                os.chmod(script_path, 0o755)
            except Exception:
                pass
        except Exception as exc:
            messagebox.showerror("Preparation failed", str(exc))
            return

        self.log(f"Prepared run directory: {run_dir_path}")
        self.log(f"Wrote: {device_gen.name}, {device_xyz.name}, {script_path.name}")
        self.log("Running DFTB+ workflow...")
        self._set_run_controls(False)

        def worker():
            try:
                result = subprocess.run(
                    ["bash", str(script_path.name)],
                    cwd=str(run_dir_path),
                    capture_output=True,
                    text=True,
                )
                output = result.stdout or ""
                if result.stderr:
                    output += ("\n" if output else "") + result.stderr
                self.root.after(0, lambda: self._finish_run(result.returncode, run_dir_path, output))
            except Exception as exc:
                self.root.after(0, lambda: self._finish_run(-1, run_dir_path, f"Run failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _set_run_controls(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in self.root.winfo_children():
            self._set_widget_state_recursive(widget, state)

    def _set_widget_state_recursive(self, widget, state):
        try:
            if widget.winfo_class() in ("Button", "TButton"):
                widget.configure(state=state)
        except Exception:
            pass
        for child in widget.winfo_children():
            self._set_widget_state_recursive(child, state)

    def _finish_run(self, returncode: int, run_dir: Path, output: str):
        self._set_run_controls(True)
        self.log_popup.append("==== DFTB+ RUN OUTPUT ====")
        self.log_popup.append(output)
        fermi_ev = self._estimate_fermi_from_outputs(run_dir)
        if fermi_ev is not None:
            self.log_popup.append(f"Estimated Fermi level from contacts: {fermi_ev:.6f} eV")
        if returncode == 0:
            self.log(f"DFTB+ workflow finished successfully in {run_dir}")
        else:
            self.log(f"DFTB+ workflow failed with return code {returncode}")

    # ------------------------------------------------------------------ Partition helper
    def _update_partition_preview(self):
        if not self.junction:
            self.partition_popup.set_text("No junction generated yet.")
            return

        plan = self._get_transport_plan()
        text = (
            "Suggested DFTB+ transport partition\n\n"
            f"Device          = {plan.device_range[0]} {plan.device_range[1]}\n"
            f"Contact (left)  = {plan.left_contact_range[0]} {plan.left_contact_range[1]}\n"
            f"Contact (right) = {plan.right_contact_range[0]} {plan.right_contact_range[1]}\n\n"
            "Notes:\n"
            "- Device = inner left buffer + molecule + inner right buffer\n"
            "- Contacts are the outer Au electrode layers\n"
            "- The transport workflow reorders atoms internally\n"
        )
        self.partition_popup.set_text(text)

    def export_partition(self):
        if not self.junction:
            messagebox.showinfo("Generate first", "Generate a junction before exporting the partition helper.")
            return
        path = filedialog.asksaveasfilename(
            title="Save partition helper",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        plan = self._get_transport_plan()
        text = (
            f"Suggested DFTB+ partition for {Path(self.loaded_path).name if self.loaded_path else 'junction'}\n\n"
            f"Device:          {plan.device_range[0]} {plan.device_range[1]}\n"
            f"Contact (left):  {plan.left_contact_range[0]} {plan.left_contact_range[1]}\n"
            f"Contact (right): {plan.right_contact_range[0]} {plan.right_contact_range[1]}\n\n"
            f"Contacts atoms/contact = {self.junction.n_contact_atoms}\n"
            f"Buffer atoms/side = {self.junction.n_buffer_atoms}\n"
            f"Molecule atoms = {self.junction.n_molecule_atoms}\n"
        )
        try:
            Path(path).write_text(text, encoding="utf-8")
            self.log(f"Saved partition helper: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    # ------------------------------------------------------------------ Plotting
    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        scale = 0.9 if getattr(event, "button", None) == "up" else 1.1
        self._zoom_3d(scale)

    def _zoom_3d(self, scale: float):
        try:
            for getter, setter in (
                (self.ax.get_xlim3d, self.ax.set_xlim3d),
                (self.ax.get_ylim3d, self.ax.set_ylim3d),
                (self.ax.get_zlim3d, self.ax.set_zlim3d),
            ):
                lo, hi = getter()
                mid = 0.5 * (lo + hi)
                half = 0.5 * (hi - lo) * scale
                setter(mid - half, mid + half)
            self.canvas.draw_idle()
        except Exception:
            pass

    def redraw(self):
        self.ax.clear()
        atoms = self.preview_atoms if self.preview_atoms else []
        if not atoms:
            self._draw_empty()
            return

        xs = np.array([a.x for a in atoms], dtype=float)
        ys = np.array([a.y for a in atoms], dtype=float)
        zs = np.array([a.z for a in atoms], dtype=float)

        # Color preview by the section of the generated junction if possible.
        if self.junction is not None and len(atoms) == self.junction.total_atoms:
            n_contact = self.junction.n_contact_atoms
            n_buffer = self.junction.n_buffer_atoms
            n_mol = self.junction.n_molecule_atoms

            left_contact = atoms[:n_contact]
            left_buffer = atoms[n_contact:n_contact + n_buffer]
            molecule = atoms[n_contact + n_buffer:ncontact + n_buffer + n_mol] if False else atoms[n_contact + n_buffer:n_contact + n_buffer + n_mol]
            right_buffer = atoms[n_contact + n_buffer + n_mol:n_contact + n_buffer + n_mol + n_buffer]
            right_contact = atoms[-n_contact:]

            def plot_group(group: Sequence[Atom], color: str):
                if not group:
                    return
                self.ax.scatter(
                    [a.x for a in group],
                    [a.y for a in group],
                    [a.z for a in group],
                    s=60,
                    c=color,
                    alpha=0.95,
                    edgecolors="k",
                    linewidths=0.35,
                )

            plot_group(left_contact, SECTION_COLORS["contact"])
            plot_group(left_buffer, SECTION_COLORS["buffer"])
            plot_group(molecule, SECTION_COLORS["molecule"])
            plot_group(right_buffer, SECTION_COLORS["buffer"])
            plot_group(right_contact, SECTION_COLORS["contact"])

            if self.show_labels_var.get():
                for a in atoms:
                    self.ax.text(a.x, a.y, a.z, a.symbol, fontsize=7)

            try:
                i1, i2 = self.junction.linker_pair
                p1 = self.junction.molecule[i1]
                p2 = self.junction.molecule[i2]
                self.ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z], color="#aa0000", linewidth=2)
            except Exception:
                pass
        else:
            by_symbol = {}
            for a in atoms:
                by_symbol.setdefault(a.symbol, []).append(a)
            for sym, group in by_symbol.items():
                self.ax.scatter(
                    [a.x for a in group],
                    [a.y for a in group],
                    [a.z for a in group],
                    s=ELEMENT_SIZES.get(sym, 40),
                    c=ELEMENT_COLORS.get(sym, "#444444"),
                    alpha=0.95,
                    edgecolors="k",
                    linewidths=0.35,
                )
            if self.show_labels_var.get():
                for a in atoms:
                    self.ax.text(a.x, a.y, a.z, a.symbol, fontsize=7)

        self.ax.set_xlabel("x (Å)")
        self.ax.set_ylabel("y (Å)")
        self.ax.set_zlabel("z (Å)")
        self.ax.set_title("Molecular junction preview")

        pad = 0.08
        dx = max(xs.max() - xs.min(), 1e-6)
        dy = max(ys.max() - ys.min(), 1e-6)
        dz = max(zs.max() - zs.min(), 1e-6)
        self.ax.set_xlim(xs.min() - pad * dx, xs.max() + pad * dx)
        self.ax.set_ylim(ys.min() - pad * dy, ys.max() + pad * dy)
        self.ax.set_zlim(zs.min() - pad * dz, zs.max() + pad * dz)

        try:
            self.ax.set_box_aspect((dx, dy, dz))
        except Exception:
            pass

        self.canvas.draw_idle()


# =============================================================================
# Main
# =============================================================================

def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    JunctionBuilderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()