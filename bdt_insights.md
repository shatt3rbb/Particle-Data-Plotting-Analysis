# BDT Performance Insights

This document captures key insights comparing the BDT score distributions in the current analysis against typical high-energy physics publications (e.g., ATLAS).

## 1. The Cosmetic Difference: Log-Odds vs. Bounded Probabilities

In typical ATLAS papers (which often use ROOT's TMVA package), the BDT score is tightly bounded between **-1.0 and 1.0**. 
In the current Scikit-Learn implementation, the `clf.decision_function()` is used to generate the histogram values, which outputs the raw **log-odds** of the trees. Log-odds can conceptually range from $-\infty$ to $+\infty$ (often seen ranging from -5 to +5).

**How to adjust (visual only):**
To make the plots look exactly like the bounded versions in the papers, you can switch from `decision_function()` to **`predict_proba()`** in the code, which forces the BDT to output a probability between `0.0` and `1.0`. 
*(Note: This is purely visual and doesn't change the actual significance optimization).*

## 2. The Physical Difference: Separation Power and Feature Selection

In literature plots, there is typically a strong physical separation where Background heavily clusters on one side (e.g., `-0.8`) and Signal clusters on the other (e.g., `+0.7`). This means the BDT has successfully learned how to distinguish them.

If the current Signal and Background distributions are stacked closely on top of each other, this overlap indicates the BDT is struggling to tell them apart.

### Why is the BDT struggling?
A BDT is only as smart as the kinematic features provided to it. Currently, the BDT might be starved for information if it's only using a limited set of variables like `['met_tst', 'MetOHT', 'dLepR', 'dPhill', 'dMetZPhi']`.

While angular variables (like `dPhill`) and Missing Transverse Energy (`met_tst`) are good, they often aren't enough on their own to separate complex physics backgrounds (like $WZ$ or $Z$+jets) from the Signal.

### Recommended Kinematic Variables to Add
To achieve the beautiful separation seen in publication-quality BDTs, it is highly recommended to feed the BDT "heavy hitter" kinematic variables, such as:

1. **Invariant Masses:** The invariant mass of the two leptons (`M2Lep` or $m_{\ell\ell}$) is arguably the most powerful variable in $Z$-boson topologies because backgrounds (like top quarks or $W$+jets) will not reconstruct nicely at the 91 GeV $Z$-mass peak.
2. **Transverse Masses:** Variables like $m_T^{ZZ}$ or the transverse mass of the $W$ boson (if a lepton is present) tightly bound kinematics where neutrinos escape.
3. **Jet Kinematics:** If the analysis phase-space allows (e.g., Vector Boson Scattering - VBS), including variables like `detajj` (delta-eta between jets) and `mjj` (invariant mass of the dijet system) will massively empower the BDT. VBS signals have a very distinct 2-jet forward signature that backgrounds lack.
4. **$p_T$ of the Lepton System:** The transverse momentum of the $Z$ candidate ($p_T^{\ell\ell}$).

### Next Steps Strategy
1. **Extend Features:** Try adding `M2Lep` and `mT_ZZ` (or equivalent invariant/transverse mass variables from `config.yaml`) to the `features` array in the BDT script. Run the analysis again to see if the separation improves.
2. **Feature Importances:** Extract and print the Feature Importances from the trained BDT (`print(clf.feature_importances_)`). This will reveal which variables are driving the decisions. Once kinematic masses are added, they typically dwarf angular variables in importance.
