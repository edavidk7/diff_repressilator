"""State and parameter containers for the repressilator.

The repressilator (Elowitz & Leibler, *Nature* **403**, 335-338, 2000) is a
cyclic negative-feedback loop of three transcriptional repressors carried on a
plasmid in *E. coli*:

    LacI  --|  tetR  ->  TetR  --|  cI  ->  CI  --|  lacI  ->  LacI

Gene names are written lower case and italic in the literature (*lacI*,
*tetR*, *cI*); their protein products are capitalised (LacI, TetR, CI).  We
keep that convention here: ``m_*`` fields hold mRNA of gene ``*``, ``p_*``
fields hold the corresponding protein.
"""

import torch
from tensordict import TensorClass


class RepressilatorState(TensorClass):
    """Dynamical variables of a (batch of) repressilator cell(s).

    All fields share the same leading batch shape, so a single instance can
    hold one cell, a population of cells, or a whole trajectory
    (``batch_size=(n_cells, n_steps)``).

    Units follow the non-dimensionalisation of Box 1 of the paper:

    * time is measured in units of the mRNA lifetime (~2 min in *E. coli*);
    * protein levels are in units of ``K_M``, the number of repressor
      molecules needed to half-maximally repress a promoter (~40 monomers per
      cell);
    * mRNA levels are rescaled by the translation efficiency, i.e. by the
      average number of proteins produced per transcript (~20).

    Attributes:
        m_lacI: *lacI* mRNA concentration.  Transcribed from P_Ltet01, so it
            is repressed by TetR.
        m_tetR: *tetR* mRNA concentration.  Transcribed from P_Llac01, so it
            is repressed by LacI.
        m_cI: *cI* mRNA concentration.  Transcribed from P_Ltet01 as well in
            the original construct's ordering; in the loop below it is the
            species repressed by TetR's downstream partner.  See
            :class:`RepressilatorParams` for the wiring convention used.
        p_LacI: LacI repressor protein (from *E. coli*).  Represses *tetR*.
        p_TetR: TetR repressor protein (from transposon Tn10).  Represses
            *cI*.
        p_CI: CI repressor protein (from phage lambda).  Represses *lacI*,
            closing the loop.
        m_gfp: mRNA of the *gfp* reporter gene (``gfp-aav``) carried on the
            second, compatible reporter plasmid.  Its promoter is P_Ltet01,
            so it is repressed by TetR and therefore reports the phase of the
            oscillator.
        p_GFP: green fluorescent protein - the experimentally observable
            readout.  Matures and decays slowly (half-life ~90 min), which
            low-pass filters the underlying oscillation.
    """

    m_lacI: torch.Tensor
    m_tetR: torch.Tensor
    m_cI: torch.Tensor

    p_LacI: torch.Tensor
    p_TetR: torch.Tensor
    p_CI: torch.Tensor

    m_gfp: torch.Tensor
    p_GFP: torch.Tensor


class RepressilatorParams(TensorClass):
    """Kinetic parameters of the repressilator model.

    These are the parameters of the symmetric, dimensionless model of Box 1::

        dm_i / dt = -m_i + alpha / (1 + p_j ** n) + alpha_0
        dp_i / dt = -beta * (p_i - m_i)

    where ``i`` runs over (*lacI*, *tetR*, *cI*) and ``j`` over the repressor
    acting on gene ``i``, i.e. (CI, LacI, TetR) respectively.

    Being a TensorClass, every parameter is a tensor and can therefore be
    batched (a population of cells with different parameters) and
    differentiated through (``requires_grad=True``) - which is the point of
    this package.

    Attributes:
        alpha: maximal transcription rate of a fully de-repressed promoter,
            expressed as protein copies per cell per mRNA lifetime.  Set by
            promoter strength and ribosome-binding-site efficiency.  Large
            ``alpha`` favours oscillation.
        alpha_0: "leakiness" - the residual transcription rate of a fully
            repressed promoter, in the same units.  In the paper
            ``alpha_0 / alpha ~ 1e-3``; leakiness comparable to ``K_M``
            shrinks the oscillatory region.
        n: Hill coefficient of repression, i.e. the cooperativity with which
            a repressor binds its operator(s).  ``n = 2`` in the paper (two
            operator sites per promoter); the unstable - hence oscillatory -
            domain grows steeply with ``n``.
        beta: ratio of the protein decay rate to the mRNA decay rate.  The
            ssrA degradation tags ("lite") bring the repressor half-life down
            to ~10 min against ~2 min for mRNA, so ``beta`` is of order one -
            a condition for sustained oscillation.
        alpha_GFP: maximal transcription rate of the reporter's promoter.
            The reporter is a separate P_Ltet01 copy on a high-copy ColE1
            plasmid, whereas the repressilator sits on the low-copy pSC101
            origin, so its effective rate differs from ``alpha`` even though
            the promoter sequence is the same.
        beta_GFP: same ratio for the GFP reporter.  GFP-AAV is markedly more
            stable than the tagged repressors (half-life ~90 min), so this is
            much smaller than ``beta``.
    """

    alpha: torch.Tensor
    alpha_0: torch.Tensor
    n: torch.Tensor
    beta: torch.Tensor
    alpha_GFP: torch.Tensor
    beta_GFP: torch.Tensor
