"""Forward model: the repressilator vector field.

The model is the symmetric, dimensionless one of Box 1 of Elowitz & Leibler
(*Nature* **403**, 335-338, 2000), extended with the GFP reporter that the
experiment actually observes.  Writing ``i`` for a gene and ``j`` for the
repressor acting on it::

    dm_i / dt = -m_i + alpha / (1 + p_j ** n) + alpha_0
    dp_i / dt = -beta * (p_i - m_i)

The wiring of the loop is

    LacI --| tetR -> TetR --| cI -> CI --| lacI -> LacI

and the reporter plasmid puts *gfp* under P_Ltet01, so GFP is repressed by
TetR alongside *cI*.

The equations below are written out one species at a time rather than
vectorised: the point of this package is to differentiate through them and
to be able to perturb any single reaction, so being able to read each
species' balance on its own line is worth more than the speed of a stacked
form.
"""

import torch
from torch import nn

from differentiable_cell.data import RepressilatorParams, RepressilatorState


class Repressilator(nn.Module):
    """Right-hand side of the repressilator ODE system.

    The module is stateless: it maps a state (and a set of kinetic
    parameters) to the time derivative of that state.  Integration is left to
    the caller, so the same module can be used with a fixed-step scheme, an
    adaptive solver, or an adjoint method.

    Both entry points return derivatives, not an updated state:

    * :meth:`forward` takes and returns :class:`RepressilatorState`, which is
      the convenient form for downstream code;
    * :meth:`forward_tensor` takes and returns bare tensors, which is the
      form a solver or a ``torch.func`` transform wants.

    Every operation is elementwise, so all inputs broadcast against each
    other: a single cell, a batch of cells, or a batch of cells each with its
    own parameters all work without changing anything here.
    """

    def forward(
        self, state: RepressilatorState, params: RepressilatorParams
    ) -> RepressilatorState:
        """Compute the time derivative of ``state``.

        Args:
            state: current concentrations of the eight species.
            params: kinetic parameters; broadcastable against ``state``.

        Returns:
            A :class:`RepressilatorState` whose fields are the time
            derivatives ``d/dt`` of the corresponding fields of ``state``,
            with the same batch shape as ``state``.
        """
        (
            dm_lacI,
            dm_tetR,
            dm_cI,
            dp_LacI,
            dp_TetR,
            dp_CI,
            dm_gfp,
            dp_GFP,
        ) = self.forward_tensor(
            m_lacI=state.m_lacI,
            m_tetR=state.m_tetR,
            m_cI=state.m_cI,
            p_LacI=state.p_LacI,
            p_TetR=state.p_TetR,
            p_CI=state.p_CI,
            m_gfp=state.m_gfp,
            p_GFP=state.p_GFP,
            alpha=params.alpha,
            alpha_0=params.alpha_0,
            n=params.n,
            beta=params.beta,
            alpha_GFP=params.alpha_GFP,
            beta_GFP=params.beta_GFP,
        )

        return RepressilatorState(
            m_lacI=dm_lacI,
            m_tetR=dm_tetR,
            m_cI=dm_cI,
            p_LacI=dp_LacI,
            p_TetR=dp_TetR,
            p_CI=dp_CI,
            m_gfp=dm_gfp,
            p_GFP=dp_GFP,
            batch_size=state.batch_size,
        )

    def forward_tensor(
        self,
        m_lacI: torch.Tensor,
        m_tetR: torch.Tensor,
        m_cI: torch.Tensor,
        p_LacI: torch.Tensor,
        p_TetR: torch.Tensor,
        p_CI: torch.Tensor,
        m_gfp: torch.Tensor,
        p_GFP: torch.Tensor,
        alpha: torch.Tensor,
        alpha_0: torch.Tensor,
        n: torch.Tensor,
        beta: torch.Tensor,
        alpha_GFP: torch.Tensor,
        beta_GFP: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Tensor-level right-hand side.

        Same equations as :meth:`forward`, on unwrapped tensors.

        Args:
            m_lacI, m_tetR, m_cI: mRNA concentrations of the three repressor
                genes.
            p_LacI, p_TetR, p_CI: the corresponding repressor proteins.
            m_gfp, p_GFP: reporter mRNA and protein.
            alpha: maximal (de-repressed) transcription rate.
            alpha_0: promoter leakiness.
            n: Hill coefficient of repression.
            beta: protein-to-mRNA decay-rate ratio for the repressors.
            alpha_GFP: maximal transcription rate of the reporter promoter.
            beta_GFP: the same decay ratio for the reporter protein.

        Returns:
            The eight derivatives, in the field order of
            :class:`RepressilatorState`: ``(dm_lacI, dm_tetR, dm_cI, dp_LacI,
            dp_TetR, dp_CI, dm_gfp, dp_GFP)``.
        """
        # --- transcription: each gene is repressed by the protein upstream
        # of it in the cycle, with a leaky floor alpha_0.
        # CI --| lacI
        dm_lacI = -m_lacI + alpha / (1.0 + p_CI**n) + alpha_0
        # LacI --| tetR
        dm_tetR = -m_tetR + alpha / (1.0 + p_LacI**n) + alpha_0
        # TetR --| cI
        dm_cI = -m_cI + alpha / (1.0 + p_TetR**n) + alpha_0

        # --- translation and protein decay: each protein is produced from
        # its own mRNA and degraded (ssrA-tagged, hence beta ~ 1).
        dp_LacI = -beta * (p_LacI - m_lacI)
        dp_TetR = -beta * (p_TetR - m_tetR)
        dp_CI = -beta * (p_CI - m_cI)

        # --- reporter: gfp sits on P_Ltet01, so TetR represses it exactly as
        # it represses cI.  It gets its own transcription rate (higher-copy
        # plasmid than the repressilator) and its own, smaller decay ratio
        # (GFP-AAV is far more stable than the ssrA-tagged repressors).
        dm_gfp = -m_gfp + alpha_GFP / (1.0 + p_TetR**n) + alpha_0
        dp_GFP = -beta_GFP * (p_GFP - m_gfp)

        return (
            dm_lacI,
            dm_tetR,
            dm_cI,
            dp_LacI,
            dp_TetR,
            dp_CI,
            dm_gfp,
            dp_GFP,
        )
