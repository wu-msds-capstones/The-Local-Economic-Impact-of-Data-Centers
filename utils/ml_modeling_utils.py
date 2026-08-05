
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import statsmodels.formula.api as smf
from scipy.stats import probplot
import statsmodels.api as sm 
import linearmodels as lm
from diff_diff import CallawaySantAnna

# slice df to specific year
def return_specific_year(df, year):
    assert"year" in df.columns, "data frame does not have a year column"
    year = int(year)
    sliced_df = df[df["year"]==year]
    sliced_df = sliced_df.drop(columns = ["year"])
    return sliced_df



# return updated version of dataframe with specified columns made to be per-capita
def make_per_capita(df, per_capita_cols):

    for col in per_capita_cols:
        df[f"{col}_per_capita"] = df[col] / df["population"]

    df = df.drop(columns=per_capita_cols)
    return df



# Check for high correlation between x variables indicating a violation of the independence assumption
def show_correlation_heat_map(df):
    corr_matrix = df.corr()
    # Plot heatmap correlations
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, vmin=-1, vmax=1)
    plt.title("Correlation Matrix Heatmap")
    plt.show()



# function to return model accuracy report
def model_accuracy_report(model, X_train, y_train, X_test, y_test):
    # training calculations
    y_pred_train = model.predict(X_train)
    train_acc = sum(y_pred_train == y_train) / len(y_train)
    print(f"Overall Training Accuracy: {train_acc:.4f}")
    
    actual_pos_train = (y_train == 1)
    true_pos_train = (y_pred_train == 1) & actual_pos_train
    tpr_train = sum(true_pos_train) / sum(actual_pos_train)
    print(f"True Positive Training Accuracy: {tpr_train:.4f}")
    
    print("-" * 20) 
    
    # test calculations
    y_pred_test = model.predict(X_test)
    test_acc = sum(y_pred_test == y_test) / len(y_test)
    print(f"Overall Test Accuracy: {test_acc:.4f}")
    
    actual_pos_test = (y_test == 1)
    true_pos_test = (y_pred_test == 1) & actual_pos_test
    tpr_test = sum(true_pos_test) / sum(actual_pos_test)
    print(f"True Positive Test Accuracy: {tpr_test:.4f}")

    ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred_test), 
    display_labels=model.classes_
    ).plot()

    plt.show()



# Help select numeric columns which meet different model assumptions
def column_selector(
    df,
    no_na = False,
    vif_cut_off = 5.0,
    outlier_sd_cut_off = 3.0,
    only_normal = False,
    return_report = False,
    cols_to_preserve = None
    ):
    """Filters DataFrame columns based on missingness, VIF, outliers, and normality.

    Examines all numeric columns in a DataFrame and filters them sequentially
    through missing value checks, Variance Inflation Factor (VIF) limits,
    standard deviation outlier cutoffs, and D'Agostino-Pearson normality tests.
    Non-numeric columns and user-specified preserved columns bypass these checks
    and are always retained.

    Args:
        df: The pandas DataFrame containing features to evaluate.
        no_na: If True, drops numeric columns that contain any NA/NaN values.
            Defaults to False.
        vif_cut_off: Maximum allowed Variance Inflation Factor threshold.
            Numeric columns with a VIF equal to or exceeding this threshold are
            dropped to reduce multicollinearity. Defaults to 5.0. To skip check 
            entirly enter a negative value.
        outlier_sd_cut_off: Number of standard deviations from the mean used to
            define outlier boundaries. Columns containing values beyond this
            range are dropped. Defaults to 3.0. To skip check 
            entirly enter a negative value.
        only_normal: If True, retains only numeric columns that satisfy
            normality criteria (p-value > 0.05 from D'Agostino-Pearson test or
            absolute skewness < 1.0). Defaults to False.
        return_report: If True, returns both the list of retained column names
            and a pandas DataFrame detailing reasons for dropped columns.
            Defaults to False.
        cols_to_preserve: A list of column names to bypass filtering and retain
            unconditionally in the output. Defaults to None.

    Returns:
        If return_report is False:
            A list of column names retained after filtering.
        If return_report is True:
            A tuple containing:
                - List[str]: Column names retained after filtering.
                - pd.DataFrame: Audit trail with columns ['dropped_col', 'reason'].
    """
    if cols_to_preserve is None:
        cols_to_preserve = []
    df = df.drop(columns=cols_to_preserve)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()
    cols_to_keep = numeric_cols.copy()
    df_num = df[numeric_cols]
    drop_reasons = []
    
    #na filter
    if no_na:
        filter_results = []
        for col in cols_to_keep:
            if not df_num[col].hasnans:
                filter_results.append(col)
            else: 
                drop_reasons.append({"dropped_col": col, "reason": "contained NA values"})
                
        cols_to_keep = filter_results
    
    df_num = df[cols_to_keep].dropna()

    #vif filter
    if vif_cut_off > 0:
        filter_results = []
        for idx, col in enumerate(cols_to_keep):
            vif = variance_inflation_factor(df_num.values, idx)
            if vif < vif_cut_off:
                filter_results.append(col)
            else:
                drop_reasons.append({"dropped_col": col, "reason": f"high VIF ({vif:.2f})"})
        cols_to_keep = filter_results
    
    #outlier filter
    if outlier_sd_cut_off > 0:
        filter_results = []
        for col in cols_to_keep:
            mean = df_num[col].mean()
            std = df_num[col].std()
            lower_bound = mean - (outlier_sd_cut_off * std)
            upper_bound = mean + (outlier_sd_cut_off * std)
            if ((df_num[col] > lower_bound) & (df_num[col] < upper_bound)).all():
                filter_results.append(col)
            else:
                drop_reasons.append({"dropped_col": col, "reason": f"exceeded {outlier_sd_cut_off} SD outliers"})
        cols_to_keep = filter_results

    #normality filter
    if only_normal:
        filter_results = []
        for col in cols_to_keep:
            skew = abs(stats.skew(df_num[col]))
            p_val = stats.normaltest(df_num[col]).pvalue
            if p_val > 0.05 or skew < 1:
                filter_results.append(col)
            else:
                drop_reasons.append({"dropped_col": col, "reason": f"non-normal (skew={skew:.2f}, p={p_val:.4f})"})
        cols_to_keep = filter_results

    drop_report_df = pd.DataFrame(drop_reasons)
    cols_to_keep = non_numeric_cols + cols_to_preserve + cols_to_keep 
    if return_report:
        return cols_to_keep, drop_report_df
    return cols_to_keep

def pivot_naics(bus_df, code_df, cols_to_pivot = ["tot_employee_count", "annual_payroll", "tot_establishment_count"]):
    """
    Groups similar NAICS codes via shared sector names and pivots into wide format.
    
    Parameters:
        bus_df = df with naics_industry_code key column
        code_df = code lookup df
        cols_to_pivot = list of columns to pivot 

    Returns:
        Wide DataFrame with one row per (county_id, year)
    """

    # Clean sector names
    code_df = code_df.copy()
    bucket_map = {
        "Agriculture, Forestry, Fishing and Hunting": "primary_industries",
        "Mining, Quarrying, and Oil and Gas Extraction": "primary_industries",
        "Utilities": "primary_industries",

        "Construction": "industrial",
        "Manufacturing": "industrial",

        "Wholesale Trade": "trade_transport",
        "Retail Trade": "trade_transport",
        "Transportation and Warehousing": "trade_transport",

        "Information": "information",

        "Finance and Insurance": "professional",
        "Real Estate and Rental and Leasing": "professional",
        "Professional, Scientific, and Technical Services": "professional",
        "Management of Companies and Enterprises": "professional",

        "Administrative and Support and Waste Management and Remediation Services": "public_services",
        "Educational Services": "public_services",
        "Health Care and Social Assistance": "public_services",
        "Arts, Entertainment, and Recreation": "public_services",
        "Accommodation and Food Services": "public_services",
        "Other Services (except Public Administration)": "public_services",
        "Public Administration": "public_services",
        
        "Unknown": "unknown",
    }
    
    code_df["bucket"] = code_df["definition"].map(bucket_map)

    # Merge codes and bus_df
    output_df = bus_df.merge(code_df, left_on = "naics_industry_code", right_on = "sector", how = "left")

    grouped_df = (
        output_df.groupby(["county_id", "year", "bucket"], as_index=False)
        [cols_to_pivot]
        .sum()
    )

    # Pivot
    pivot_df = grouped_df.pivot_table(
        index=["county_id", "year"],
        columns="bucket",
        values=cols_to_pivot,
        aggfunc="sum"
    )

    pivot_df.columns = [
        f"{metric}_{sector}" for metric, sector in pivot_df.columns
    ]

    pivot_df = pivot_df.reset_index()

    return pivot_df

def eda_summary(df, acceptable_skew = 3.0):
    """
    Returns missingness, number of unique observations, standard deviation,
    skew, zero information, min, max, and suggested transformation
    for all numeric columns in a dataframe

    Parameters:
        - df to run eda on
        - acceptable_skew: how much skew is considered "acceptable" for determining whether a transform is needed
    """
    num_df = df.select_dtypes(include="number").copy()

    summary = pd.DataFrame({
        "missing_share": num_df.isna().mean(),
        "n_unique": num_df.nunique(),
        "std": num_df.std(numeric_only=True).round(3),
        "skew": num_df.skew(numeric_only=True).round(3),
        "n_zeros": (num_df == 0).sum(),
        "zero_share": (num_df == 0).mean().round(3),
        "min": num_df.min(numeric_only = True).round(3),
        "max": num_df.max(numeric_only = True).round(3)
    }).sort_values("skew", ascending=False).reset_index().rename(columns={"index": "column"})

    summary["can_log"] = summary["min"] > 0
    summary["suggested_transform"] = np.where(
        summary["can_log"] & (summary["skew"] > acceptable_skew),
        "log",
        np.where(
            (~summary["can_log"]) & (summary["zero_share"] > 0) & (summary["skew"] > acceptable_skew),
            "log1p",
            "none"
        )
    )

    return summary

def apply_suggested_transforms(df, eda_df, cols_to_exclude = None):
    """
    Apply the suggested transformations from the `eda_summary` output
    """
    if cols_to_exclude is None:
        cols_to_exclude = []

    df = df.copy()
    transform_map = eda_df.set_index("column")["suggested_transform"].to_dict()

    rename_map = {}

    for col, transform in transform_map.items():
        if col in cols_to_exclude:
            continue

        if transform == "none":
            continue

        new_col = f"{col}_{transform}"

        if transform == "log":
            df[col] = np.where(df[col] > 0, np.log(df[col]), np.nan)
        elif transform == "log1p":
            df[col] = np.log1p(df[col])
        else:
            continue

        rename_map[col] = new_col

    df = df.rename(columns = rename_map)

    return df

def _pick_best_from_group(lasso_coef_df, candidates):
    temp = lasso_coef_df[lasso_coef_df["feature"].isin(candidates)].copy()
    if temp.empty:
        return []
    else:
        return [temp.sort_values("abs_coef", ascending=False)["feature"].iloc[0]]

def run_lasso(df, y_col, exclude_cols = None, verbose = True):
    """
    Runs lasso in a dataframe against a specified response variable,
    which helps narrow down variables that may be predictive of that response variable.
    Variables with a higher coefficient should be included,
    whereas variables at or near zero should be considered for exclusion from the final model.

    Parameters:
        - df: input dataframe
        - y_col: the response variable to run lasso against
        - exclude_cols: columns to explicitly exclude from lasso
        - verbose: whether results should be printed to console or not
    """
    target_col = y_col

    if exclude_cols is None:
        exclude_cols = []

    drop_cols = ["county_id", "year", target_col] + exclude_cols

    lasso_df = df.dropna().copy()
    lasso_df.columns = lasso_df.columns.map(str)

    x_cols = [c for c in lasso_df.columns if c not in drop_cols]
    
    X = lasso_df[x_cols]
    y = lasso_df[target_col]

    lasso_pipe = make_pipeline(
        StandardScaler(),
        LassoCV(cv = 5, random_state = 15215, max_iter = 10000)
    )

    lasso_pipe.fit(X, y)

    lasso_model = lasso_pipe.named_steps["lassocv"]
    coef_df = pd.DataFrame({
        "feature": x_cols,
        "coef": lasso_model.coef_
    })

    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending = False).reset_index(drop=True)

    if verbose:
        print(coef_df.to_string(index=False))

    coef_df = coef_df[coef_df["coef"] != 0].reset_index(drop = True)

    return coef_df

def select_grouped_features(
    lasso_coef_df,
    reg_dc_cols,
    reg_dc_density_cols,
    reg_size_cols,
    reg_sector_cols,
    reg_pop_col,
    reg_econ_cols,
    reg_env_cols,
    reg_permit_cols,
    include_information=False):      
    """
    Apply pre-defined grouping rules to lasso output
    """
    selected = set(lasso_coef_df["feature"].tolist())

    final_x = []

    # data center logic
    if not reg_dc_cols:
        final_x.extend([c for c in reg_dc_density_cols])
    else:
        has_dc_count = any(c in selected for c in reg_dc_cols)
        has_dc_density = any(c in selected for c in reg_dc_density_cols)
        has_size_col = any(c in selected for c in reg_size_cols)

        use_density = not (has_dc_count or has_size_col) and has_dc_density

        if use_density:
            final_x.extend([c for c in reg_dc_density_cols])
        else:
            final_x.extend([c for c in reg_dc_cols])
            final_x.extend(_pick_best_from_group(lasso_coef_df, reg_size_cols))

    # Population only if selected
    final_x.extend([c for c in reg_pop_col if c in selected])

    # Econ: any number
    final_x.extend([c for c in reg_econ_cols if c in selected])

    # Environmental: only one
    final_x.extend(_pick_best_from_group(lasso_coef_df, reg_env_cols))

    # Permits: only one
    final_x.extend(_pick_best_from_group(lasso_coef_df, reg_permit_cols))

    # Sectors: only if selected, but exclude information unless explicitly allowed
    sector_keep = reg_sector_cols.copy()
    if not include_information:
        sector_keep = [c for c in sector_keep if "information" not in c]

    final_x.extend([c for c in sector_keep if c in selected])

    final_x = list(dict.fromkeys(final_x))
    return final_x

def plot_regression_coeffs(model, y_col, reg_df, upper_p = 0.90, lower_p = 0.10):
    ci = model.conf_int()
    ci.columns = ["ci_low", "ci_high"]

    round_lower_p = int(lower_p * 100)
    round_upper_p = int(upper_p * 100)

    plot_df = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "pvalue": model.pvalues.values,
        "ci_low": ci["ci_low"].values,
        "ci_high": ci["ci_high"].values
    })

    plot_df = plot_df[(plot_df["term"] != "Intercept") & (~plot_df["term"].str.startswith("C("))].copy()

    plot_df[f"p{round_lower_p}"] = plot_df["term"].apply(lambda t: reg_df[t].quantile(lower_p))
    plot_df[f"p{round_upper_p}"] = plot_df["term"].apply(lambda t: reg_df[t].quantile(upper_p))
    plot_df["x_shift"] = plot_df[f"p{round_upper_p}"] - plot_df[f"p{round_lower_p}"]

    plot_df["effect"] = plot_df["coef"] * plot_df["x_shift"]
    plot_df["effect_low"] = plot_df["ci_low"] * plot_df["x_shift"]
    plot_df["effect_high"] = plot_df["ci_high"] * plot_df["x_shift"]

    plot_df["abs_effect"] = plot_df["effect"].abs()
    plot_df = plot_df.sort_values("abs_effect", ascending=False).sort_values("effect").reset_index(drop=True)

    plot_df["is_datacenter"] = plot_df["term"].str.contains("datacenter", case=False, na=False)

    plt.figure(figsize=(11, max(5, 0.55 * len(plot_df) + 1.5)))

    colors = np.where(
    (plot_df["effect_low"] <= 0) & (plot_df["effect_high"] >= 0),
    "#A0A830",  # blue if not significant
    np.where(plot_df["coef"] >= 0, "#006494", "#a12c7b")
)
    fontweights = np.where(plot_df["is_datacenter"], "bold", "normal")

    x_view_min = -0.2
    x_view_max = 0.2

    plot_df["plot_low"] = plot_df["effect_low"].clip(lower=x_view_min)
    plot_df["plot_high"] = plot_df["effect_high"].clip(upper=x_view_max)

    plt.hlines(
        y=plot_df["term"], 
        xmin = plot_df["plot_low"], 
        xmax = plot_df["plot_high"], 
        color = "#797562", 
        linewidth = 2)
    
    plt.scatter(
        plot_df["effect"].clip(lower=x_view_min, upper=x_view_max), 
        plot_df["term"], 
        c=colors,
        zorder = 3)

    plt.xlim(x_view_min, x_view_max)

    plt.axvline(0, color =  "#28251d", linestyle = "--", linewidth = 1)
    plt.xlabel(f"Predicted change in {y_col} from P{round_lower_p} to P{round_upper_p} of predictor")
    plt.ylabel("")
    plt.title(f"Regression effects for {y_col}")

    ax = plt.gca()

    for tick_label in ax.get_yticklabels():
        if "datacenter" in tick_label.get_text():
            tick_label.set_fontweight("bold")

    x_min = plot_df["effect_low"].min()
    x_max = plot_df["effect_high"].max()
    x_pad = (x_max - x_min) * 0.02 if x_max > x_min else 0.1

    for _, row in plot_df[plot_df["is_datacenter"]].iterrows():
        label_x = row["effect_high"] + x_pad
        label_txt = f"P{round_lower_p}={row[f'p{round_lower_p}']:.2f}, P{round_upper_p}={row[f'p{round_upper_p}']:.2f}"
        ax.text(
            label_x,
            row["term"],
            label_txt,
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#28251d"
        )

    for _, row in plot_df.iterrows():
        if row["effect_low"] < x_view_min:
            plt.scatter(x_view_min, row["term"], marker="<", color="#797562", s=35, zorder=4)
        if row["effect_high"] > x_view_max:
            plt.scatter(x_view_max, row["term"], marker=">", color="#797562", s=35, zorder=4)

    for _, row in plot_df[plot_df["is_datacenter"]].iterrows():
        x_text = np.clip(row["effect"], x_view_min, x_view_max)
        ax.annotate(
            f"{row['effect']:.4f}",
            xy=(x_text, row["term"]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#28251d"
        )

    plt.tight_layout()
    plt.show()

def run_linear_regression(df, y_col, x_cols, county_fe = True, year_fe = True):
    """
    Runs a linear regression and prints the regression output,
    residuals plot (Linearity and Equal Variance assumptions),
    QQ plot (Normally-distributed errors/residuals assumption),
    and VIF results (Independent observations (given X) assumption)

    Parameters:
        - df = the dataframe you want to do regression on
        - y_col = the response variable
        - x_cols = the explanatory variables
        - county_fe = whether you want to use a county fixed effect
        - year_fe = whether you want to use a year fixed effect

    Assumption(s):
        - Always clusters by county
    """

    reg_df = df.dropna(subset=[y_col] + x_cols).copy()

    reg_df_panel  = reg_df.set_index(["county_id", "year"])

    y = reg_df_panel[y_col]
    X = reg_df_panel[x_cols]

    entity_effects = county_fe
    time_effects = year_fe

    model = lm.PanelOLS(
        y,
        X,
        entity_effects = entity_effects,
        time_effects = time_effects
    ).fit(cov_type = "clustered", cluster_entity=True)

    fitted = model.fitted_values
    resid = model.resids

    ci = model.conf_int()
    ci.columns = ["ci_low", "ci_high"]

    summary_df = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "pvalue": model.pvalues.values,
        "ci_low": ci["ci_low"].values,
        "ci_high": ci["ci_high"].values
    })

    summary_df["pvalue"] = summary_df["pvalue"].apply(lambda x: f"{x:.6f}" if x >= 0.000001 else "<0.000001")
    summary_df["coef"] = summary_df["coef"].round(6)
    summary_df["ci_low"] = summary_df["ci_low"].round(6)
    summary_df["ci_high"] = summary_df["ci_high"].round(6)

    print("\nModel:")
    print(model.summary)

    plot_regression_coeffs(model, y_col, reg_df)

    # Examine Residuals
    plt.figure(figsize=(7, 5))
    plt.scatter(fitted, resid, alpha=0.6)
    plt.axhline(0, color="red", linestyle = "--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted Values")
    plt.show()

    # Check For Normal Distribution
    plt.figure(figsize=(7, 5))
    probplot(resid, dist = "norm", plot = plt)
    plt.title("Q-Q Plot of Residuals")
    plt.show()

    # Check VIF
    vif_X = reg_df[x_cols].copy()
    vif_X = sm.add_constant(vif_X, has_constant="add")

    vif_df = pd.DataFrame({
        "variable": vif_X.columns,
        "vif": [
            np.nan if col == "const" else variance_inflation_factor(vif_X.values, i)
            for i, col in enumerate(vif_X.columns)
        ]
    })

    print(vif_df)

    return {
        "model": model,
        "reg_df": reg_df
    }

def print_regression_analysis(result, y_col, reg_dc_cols, reg_dc_density_cols_lag):
    all_dc_cols = reg_dc_cols + reg_dc_density_cols_lag

    model = result["model"]

    rows = []
    for var in all_dc_cols:
        if var in model.params.index:
            rows.append({
                "variable": var,
                "coef": model.params[var],
                "pvalue": model.pvalues[var]
            })

    dc_table = pd.DataFrame(rows)

    print("\n" + "-" * 100)
    print(f"RESULTS FOR REGRESSION ON: {y_col}")
    print(f"R2 (Within): {model.rsquared_within:.4f}")
    print(f"R2 (Overall): {model.rsquared:.4f}")
    print("\nData center coefficients:")
    print(dc_table.to_string(index=False))

def run_lasso_plus_regression(
    df,
    y_col,
    reg_dc_cols,
    reg_dc_density_cols,
    reg_size_cols,
    reg_sector_cols,
    reg_pop_col,
    reg_econ_cols,
    reg_env_cols,
    reg_permit_cols,
    exclude_from_lasso=None,
    dc_lag=2,
    county_fe=True,
    year_fe=True,
    include_information=False):
    """
    Wrapper function for linear regression:
    1. runs lasso
    2. applies grouping rules
    3. adds lagged datacenter terms
    4. runs linear regression
    5. prints analysis
    """

    if exclude_from_lasso is None:
        exclude_from_lasso = []

    if reg_dc_cols is None:
        reg_dc_cols = []

    if reg_size_cols is None:
        reg_size_cols = []

    if dc_lag is not None:
        all_dc_cols = reg_dc_cols + reg_dc_density_cols

        reg_dc_cols = []
        reg_dc_density_cols = []

        for col in all_dc_cols:
            lag_col = f"{col}_lag{dc_lag}"
            if "per" in col:
                reg_dc_density_cols.append(lag_col)
            else:
                reg_dc_cols.append(lag_col)

        all_dc_cols = reg_dc_cols + reg_dc_density_cols

    df = df[[c for c in df.columns if "datacenter" not in c or c in all_dc_cols]]

    lasso_coef_df = run_lasso(
        df=df,
        y_col=y_col,
        exclude_cols=exclude_from_lasso, 
        verbose = True
    )

    dc_lasso_dfs = []
    for dc_col in all_dc_cols:
        dc_exclude = exclude_from_lasso + [y_col, dc_col]
        dc_lasso_df = run_lasso(
            df=df,
            y_col=dc_col,
            exclude_cols=dc_exclude,
            verbose = False
        )
        dc_lasso_dfs.append(dc_lasso_df)

    if dc_lasso_dfs:
        combined_lasso_df = pd.concat([lasso_coef_df] + dc_lasso_dfs)
        combined_lasso_df = combined_lasso_df.drop_duplicates(subset="feature").reset_index(drop=True)
        lasso_coef_df = combined_lasso_df

    selected_x = select_grouped_features(
        lasso_coef_df=lasso_coef_df,
        reg_dc_cols=reg_dc_cols,
        reg_dc_density_cols=reg_dc_density_cols,
        reg_size_cols=reg_size_cols,
        reg_sector_cols=reg_sector_cols,
        reg_pop_col=reg_pop_col,
        reg_econ_cols=reg_econ_cols,
        reg_env_cols=reg_env_cols,
        reg_permit_cols=reg_permit_cols,
        include_information=include_information
    )

    result = run_linear_regression(
        df=df,
        y_col=y_col,
        x_cols=selected_x,
        county_fe=county_fe,
        year_fe=year_fe,
    )

    print_regression_analysis(result, y_col, reg_dc_cols, reg_dc_density_cols)

def add_lags(df, col, lags=(1, 2, 3), group_col="county_id", time_col="year"):
    df = df.copy()
    df = df.sort_values([group_col, time_col])
    for lag in lags:
        df[f"{col}_lag{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df

def compare_models(models, names):
    rows = []
    for name, model in zip(names, models):
        rows.append({
            "model": name,
            "r2": model.rsquared,
            "adj_r2": model.rsquared_adj,
            "aic": model.aic,
            "bic": model.bic,
            "nobs": model.nobs
        })
    return pd.DataFrame(rows).sort_values("aic")

def combine_dfs(base_df, dfs_to_join):
    out_df = base_df.copy()

    for i, df in enumerate(dfs_to_join):
        if "county_id" not in df.columns:
            raise ValueError(f"No `county_id` column found in dfs_to_join[{i}]")

        # Make sure all tables only have one row per county
        dupes = df["county_id"][df[["county_id","year"]].duplicated()].unique()
        if len(dupes) > 0:
            raise ValueError(f"dfs_to_join[{i}] is not unique on `county_id and year`. Problematic Counties: {list(dupes)}")

        out_df = pd.merge(out_df, df, how = "left", on = ["county_id", "year"], validate = "one_to_one")

    missing_summary_table = (
        out_df.isna()
        .mean()
        .sort_values(ascending = False)
        .rename("missing_share")
        .reset_index()
        .rename(columns={"index":"column"})
    )

    print("Missing Summary: \n", missing_summary_table)

    return out_df

def plot_model_residuals(residuals, fitted_values):
    '''
    residuals: numpy array of model's residuals values
    fitted_values: numpy array of model's fitted values
    '''
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_values, residuals, alpha=0.3, color="navy", edgecolors="none")
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Fitted Values")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="navy", bins=50)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Model Residuals")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()


def create_DiD_model(df, y_var, first_treat="first_dc_year"):
    cs = CallawaySantAnna(
    control_group="never_treated",  
    estimation_method="reg",
    allow_unbalanced_panel=True
    )

    results = cs.fit(
        data=df,
        outcome=y_var,
        unit="county_id",
        time="year",
        first_treat=first_treat,
        covariates=[],
        aggregate="all"
    )

    es_df = (
        pd.DataFrame.from_dict(results.event_study_effects, orient="index")
        .sort_index()
        .loc[-5:5]
    )
    es_df["ci_lower"] = es_df["conf_int"].apply(lambda x: x[0])
    es_df["ci_upper"] = es_df["conf_int"].apply(lambda x: x[1])

    return results, es_df



def DiD_coeff_plot(es_df, y_var, treatment_type = "Data Center", save = False, clean_name = None, cluster_num = None):
    if clean_name is None:
        clean_name = y_var

    plt.figure(figsize=(7, 4))

    plt.errorbar(
        x=es_df.index,
        y=es_df["effect"],
        yerr=[es_df["effect"] - es_df["ci_lower"], es_df["ci_upper"] - es_df["effect"]],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        label="ATT Estimate (95% CI)",
    )

    plt.plot(es_df.index, es_df["effect"], color="#1f77b4", linestyle="--", alpha=0.7)
    plt.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.7)

    if cluster_num is None:
        plt.title(f"Event Study: {treatment_type} Impact on {clean_name}", fontsize=13, fontweight="bold", pad=12)
    else:
        plt.title(f"{treatment_type} Impact on {clean_name} in Cluster {cluster_num}", fontsize=13, fontweight="bold", pad=12)
        
    plt.xlabel(f"Relative Year to {treatment_type} Addition", fontsize=11)
    plt.ylabel(f"Difference in {clean_name}", fontsize=11)
    plt.xticks(es_df.index)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, loc="best")

    plt.tight_layout()
    plt.grid(False)
    if save:
        plt.savefig(f"../poster_plots/DiD_{clean_name}_{treatment_type}_{cluster_num}.png", dpi=70, bbox_inches="tight")
    plt.show()