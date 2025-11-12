def get_customers_for_explanation(y_test):
    idx0s = y_test.index[y_test == 0]
    idx1s = y_test.index[y_test == 1]
    if len(idx0s) == 0 or len(idx1s) == 0:
        st.error("Need at least one 0 and one 1 in y_test to do this analysis.")
        return

    i_non   = idx0s[0]
    i_churn = idx1s[0]

    return [i_non, i_churn]
