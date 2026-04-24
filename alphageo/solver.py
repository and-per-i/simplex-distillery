from __future__ import annotations
from pathlib import Path
import logging
from typing import TYPE_CHECKING, Optional, Any
import copy

from newclid.api import GeometricSolver as NCGeometricSolver
from newclid.jgex.formulation import JGEXFormulation
from newclid.jgex.clause import JGEXClause
from newclid.jgex.problem_builder import JGEXProblemBuilder
from newclid.problem import ProblemSetup
from newclid.proof_state import ProofState

if TYPE_CHECKING:
    from newclid.jgex.definition import JGEXDefinition

LOGGER = logging.getLogger(__name__)

class LegacyGeometricSolver:
    def __init__(self, nc_solver: NCGeometricSolver, jgex_problem: JGEXFormulation, builder: JGEXProblemBuilder):
        self.nc_solver = nc_solver
        self.jgex_problem = jgex_problem
        self.builder = builder
        self.run_infos = {}

    def run(self) -> bool:
        success = self.nc_solver.run()
        if self.nc_solver.run_infos:
            # Convert RunInfos to dict if needed, or just keep it
            self.run_infos = self.nc_solver.run_infos.model_dump()
        return success

    def proof(self) -> str:
        return self.nc_solver.proof()

    def write_solution(self, out_file: Path):
        if out_file is None:
            print(self.proof())
            return
        out_file.write_text(self.proof())

    def draw_figure(self, out_file: Path):
        self.nc_solver.draw_figure(out_file=out_file, jgex_problem=self.jgex_problem)

    def get_setup_string(self) -> str:
        # The LM expects a specific string format. 
        # JGEXFormulation.__str__ is: setup | aux ? goals
        # Usually we only want the setup part for the LM prompt.
        setup_str = "; ".join(str(c) for c in self.jgex_problem.setup_clauses)
        if self.jgex_problem.auxiliary_clauses:
            setup_str += " | " + "; ".join(str(c) for c in self.jgex_problem.auxiliary_clauses)
        return setup_str

    def get_problem_string(self) -> str:
        return str(self.jgex_problem)

    def get_proof_state(self) -> Any:
        # For beam search, we need to be able to save and load state.
        # Since ProofState might be complex, we might need to store the jgex_problem state.
        return copy.deepcopy(self.jgex_problem)

    def load_state(self, state: JGEXFormulation):
        self.jgex_problem = copy.deepcopy(state)
        # We need to rebuild the nc_solver for this problem
        self._rebuild_solver()

    def load_problem_string(self, pstring: str):
        self.jgex_problem = JGEXFormulation.from_text(pstring)
        self._rebuild_solver()

    def _rebuild_solver(self):
        self.builder.with_problem(self.jgex_problem)
        setup = self.builder.build()
        
        from newclid.api import GeometricSolverBuilder
        gs_builder = GeometricSolverBuilder(rng=self.builder.rng)
        # Preserve rules from the previous solver
        gs_builder.with_rules(self.nc_solver.rules)
        self.nc_solver = gs_builder.build(setup)

    def get_existing_points(self) -> list[str]:
        return [str(p) for p in self.jgex_problem.points]

    def get_defs(self) -> dict[str, JGEXDefinition]:
        return self.builder.jgex_defs

    def validate_clause_txt(self, aux_string: str) -> str:
        # Check if the aux_string is a valid JGEX clause
        try:
            clauses = JGEXClause.from_str(aux_string)
            if not clauses:
                return "ERROR: empty clause"
            return aux_string
        except Exception as e:
            return f"ERROR: {str(e)}"

    def add_auxiliary_construction(self, aux_string: str):
        # aux_string is like "e = on_line e a b"
        new_clauses = JGEXClause.from_str(aux_string)
        self.jgex_problem.auxiliary_clauses += new_clauses
        self._rebuild_solver()
