def display_fairness_table_lr(X_test, y_test, tuned_model, scaled_X_test_features, threshold):
    def group_fairness_table(df, group_col, threshold, y_true='y_true', y_proba='y_proba', reference=None):
        """Return a per-group fairness table and the chosen reference group."""
        # pick reference = largest group if not provided
        ref = reference or df[group_col].value_counts().idxmax()

        def recall_pos(g):
            mask = g[y_true] == 1
            if mask.sum() == 0:
                return np.nan
            y_hat = (g[y_proba] >= threshold).astype(int)
            return recall_score(g.loc[mask, y_true], y_hat[mask])

        agg = (
            df.groupby(group_col)
              .apply(lambda g: pd.Series({
                  'n': len(g),
                  'observed_churn_rate': g[y_true].mean(),
                  'mean_pred_proba': g[y_proba].mean(),
                  f'prediction_rate@{threshold:.3f}': (g[y_proba] >= threshold).mean(),
                  'recall_pos': recall_pos(g),
              }))
              .reset_index()
        )

        ref_mean = agg.loc[agg[group_col] == ref, 'mean_pred_proba'].values[0]
        agg[f'Δ_vs_{ref}'] = (agg['mean_pred_proba'] - ref_mean).abs()
        agg['stat_parity_ok(≤0.05)'] = agg[f'Δ_vs_{ref}'] <= 0.05
        return agg.sort_values('n', ascending=False), ref

    group_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "PaymentMethod", "InternetService", "PaperlessBilling"]  
  
    X_test_original_groups = X_test[group_cols].copy()
    if hasattr(y_test, "dtype") and y_test.dtype == object:
        y_true = y_test.map({"No": 0, "Yes": 1}).astype(int)
    elif isinstance(y_test, pd.DataFrame) and "Churn" in y_test.columns:
        y_true = y_test["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    else:
        y_true = y_test.astype(int)  # already 0/1

    if hasattr(tuned_model, "predict_proba"):           # single estimator
        models = {"logreg_final": tuned_model}
    elif isinstance(tuned_model, (list, tuple)):        # list/tuple of estimators
        models = {f"model_{i}": m for i, m in enumerate(tuned_model)}
    elif isinstance(tuned_model, dict):                  # already a dict
        models = tuned_model
    else:
        raise TypeError("tuned_model must be an estimator, list/tuple of estimators, or dict.")

    X_for_pred = scaled_X_test_features

    results = {}
    for name, model in models.items():
        # get probabilities for the positive class
        y_proba = model.predict_proba(X_for_pred)[:, 1]

        # build evaluation frame aligned by index
        df_eval = X_test_original_groups.copy()
        df_eval["y_true"] = pd.Series(y_true.values, index=df_eval.index)
        df_eval["y_proba"] = pd.Series(y_proba, index=df_eval.index)

    # per-group tables for this model
        model_tables = {}
        for gcol in group_cols:
            sub = df_eval[[gcol, "y_true", "y_proba"]].dropna()
            table, ref = group_fairness_table(
                sub, gcol, threshold, y_true="y_true", y_proba="y_proba"
            )
            model_tables[gcol] = (table, ref)
        results[name] = model_tables

    st.title("Fairness Evaluation — Group-Based Metrics")
    st.write("Global threshold for recall comparisons (equal opportunity): ", threshold)

    for model_name, tables in results.items():
        with st.expander(f"Model: {model_name}", expanded=False):
            for gcol, (table, ref) in tables.items():
                st.subheader(f"Group column: {gcol} (ref: {ref})")
                st.dataframe(table, use_container_width=True)

                # CSV download
                csv_buf = StringIO()
                table.to_csv(csv_buf, index=False)
                st.download_button(
                    label=f"Download CSV — {model_name} · {gcol}",
                    data=csv_buf.getvalue(),
                    file_name=f"fairness_{model_name}_{gcol}.csv",
                    mime="text/csv"
                )
