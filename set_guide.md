# Set construction guide

Fossil proves properties  of dynamical models based on sets of states. This guide provdies an explanation of the implicit assumptions and conventions used in the construction of these sets, and tips on how to construct sets that are amenable to analysis.

## Set construction

Fossil distinguishes between *data* sets and *symbolic* sets. The exact domains which are passed to Fossil for data and for symbolic are not identical. In this guide we will refer to symbolic sets as *sets* and data sets as *data*, and we will use the term *domain* to refer the the region of the state space which is passed to Fossil as a set. Also, $XD$ refers to the symbolic domain, and $SD$ refers to the data domain.

Semantically, we refer to the following domains:
XD: The region of interest.
XI: The initial conditions.
XU: The unsafe region.
XS: The safe region (equivalent to XD \ XU).
XG: The goal region.
XF: The final region

Fossil ensures that the exact sets are passed to it for the corresponding certificate. If a redundant set is passed, Fossil will  throw an error. This might seem like a nuisance, but it  helps prevent incorrect certificates from being generated.

Generally, SD is used to train the lie derivative condition for any certificate, so often should encompass the entire region of interest.

### Lyaupunov sets

#### Lyapunov Required Symbolic Sets

XD

#### Lyapunov Required Data Sets

SD

The simplest case. Only pass XD as a set and corresponding SD as data. Ideally, SD only contains points which are in fact stable, but of course this is not always possible.

### ROA sets

#### ROAD Required Symbolic Sets

XI

#### ROA Required Data Sets

SD, SI

Here, XD is unused by the verifier, and should not be passed. Only XI is used by the verifier, and only to prove that it lies within the level set that is being verified.
SD is used to train the lie derivative condition for the certificate, and again should ideally only contain points which are in fact stable.
SI is used to estimate the smallest level set that contains XI, which is then the level set in which the Lyapunov conditions are checked. Samples over the border of SI is more efficient but not required.

### Barrier sets

#### Barrier Required Symbolic Sets

XD, XU, XI

#### Barrier Required Data Sets

SD, SU, SI

Nothing unusual here. Just a warning that a barrier certificate implicitly assumes that trajectories remain within XD, so safety is only proved within XD. In other words, trajectories are not guaranteed to remain within XD. If they leave XD, no guarantee of safety is proven.

### RWA sets

#### RWA Required Symbolic Sets

XD, XI, XS, XS_BORDER, XG

#### RWA Required Data Sets

SD, SI, SU

Crucially, XS is assumed to be closed. In practice this generally means that XS, for example a rectangle surrounding the origin. However, an open region may be excluded from XS. Fossil frames this as a safe set, rather than an unsafe set, which is in practice more convenient and easier to handle for this certificate. However, data should be generated for the unsafe region, SU, which is the complement of XS. XG, XI are also compact.

### RSWA sets

#### RSWA Required Symbolic Sets

XD, XI, XS, XS_BORDER, XG, XG_BORDER

#### RSWA Required Data Sets

SD, SI, SU, SG, SS, SG_BORDER

This certificate makes the same assumptions as the RWA certificate, in that XS is closed. It also requires the border of the goal set to be passed symbolically, and requires data from the safe set (used to search for beta) and the border of the goal set.

### Stable Safe Sets

#### Stable Safe Required Symbolic Sets

XD, XI, XU

#### Stable Safe Required Data Sets

SD, SI, SU, SR (optional)

The only certificate with an optional set. SD is used to train the lie derivative condition for the Barrier and the ROA certificate, but this might not be sensible in practice. As mentioned, the ROA certificate might struggle is lots of unstable points are included, and ideally should be trained on points between the origin and XI. If the barrier certificate requires and XD that does not align with this, a separate data set SR can be passed which is instead the data used to train the ROA certitificate. Ideally, this contains data from XI and the region between XI and the origin (even more ideally, only regions that trajectories starting in XI will pass through).

### RAR sets

#### RAR Required Symbolic Sets

XD, XI, XS, XS_BORDER, XG, XF

#### RAR Required Data Sets

SD, SI, SU, SG, SF, SNF

This certificate relies on the RWA certificate, so the safe set should be closed. The set XNF refers to the set of points which are not in the final set, XF, and is used to help learn the final invariant. Therefore SNF should contain points which are not in the final set, but are in the overall domain. The closer to XF and closer together the points are, the better - sampling the border of XF can be effective.
