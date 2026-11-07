# Tree-model support

Implement the existing `read`, `fix`, `write`, and `improve` workflow for fitted XGBoost gbtree models, CatBoost symmetric trees and scikit-learn random forests. Reuse the session engine, probes, budgets, sandbox, checkpoints, diffs, transcripts and results.

## Scope

- Scalar regression, binary classification and multiclass classification.
- Native XGBoost categorical partitions/recoding and CatBoost one-hot/CTR models, including missing routes. Categorical CatBoost requires a private reference pool.
- Fixed features, topology, node and tree counts, output dimensions and categorical rules. No fitting, new trees or structural rewrites.
- Editable leaf arrays, numeric XGBoost/forest thresholds and safe shared CatBoost borders. Forest classification arrays contain valid probabilities.
- Immutable model descriptions and bounded tree inspection, without implicit prediction.

## Delivery

1. Shared tree staging/validation and forest adapter under `src/weightspace/trees`.
2. XGBoost adapter using reloadable native JSON rather than dumps, preserving wrapper/booster identities and categorical recoding.
3. CatBoost JSON-to-CBM synchronization, preserving hashes, CTR tables, shared border references and scale/bias. Temporary files are private and cleaned up.
4. Session tools/prompts, text-mode structure, original-object probe synchronization and error finalization. Invalid edits remain staged while native state stays valid.
5. Backend-specific write initialization before committing its diff baseline; zero-init bypass. Multi-border CatBoost groups remain intact and are reported.
6. Native export, explicit native-file selection, trusted forest joblib loading, atomic versioned tree patches and CLI flags. Legacy tensor array exports and patches retain their behavior.
7. Optional dependency extras, offline examples, documentation and NumPy/tree-only CI with minimum/current backend versions.

## Verification

Use deterministic small fitted models and `ScriptedLLM`; no provider credentials. Verify untouched prediction parity, known leaf/threshold edits, categorical native round trips, raw margins/probabilities, rollback after validation/native-load failures, patch compatibility, tracking, all four tasks, text mode, initialization, budgets and probes closing over original models. Run the existing suite, Ruff, formatting, mypy and package builds.

Required backend ranges: scikit-learn `>=1.6,<2`, XGBoost `>=3.1,<4`, CatBoost `>=1.2.8,<1.3`. NumPy-only use must not require these packages or Torch.

## Constraints

Parameter-only exports do not reconstruct predictive tree models. Forest unpickling is trusted-input-only. External mutation/refitting during a session is unsupported. Training statistics, feature importance and OOB scores remain descriptions of the original fit. Native categorical support does not guarantee readable category labels or acceptance of unseen labels.
