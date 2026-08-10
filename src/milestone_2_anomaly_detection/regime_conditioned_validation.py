from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scipy.special import logsumexp

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .prepare_anomaly_features import (
    M1_FEATURES,
    VOLATILITY_FLOOR_QUANTILE,
    add_anomaly_features,
    estimate_volatility_floor,
    validate_features,
)


# ============================================================
# Import the already validated M1 HMM fitting routine
# ============================================================

M2_DIR = Path(__file__).resolve().parent
SRC_ROOT = M2_DIR.parent
PROJECT_ROOT = M2_DIR.parents[1]

sys.path.insert(
    0,
    str(SRC_ROOT),
)

from milestone_1_regime_detection.chronological_validation import fit_best_hmm


# ============================================================
# Paths
# ============================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "usdtry_features.csv"
)

OUTPUT_DIR = (
    M2_DIR
    / "outputs"
    / "regime_conditioned_validation"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "regime_conditioned_summary.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "regime_conditioned_predictions.csv"
)


# ============================================================
# Chronological split
# ============================================================

TRAIN_END = pd.Timestamp(
    "2022-12-30"
)


# ============================================================
# M1 HMM features
# ============================================================

HMM_FEATURES = [
    "Return_1D",
    "Return_5D",
    "Volatility_5D",
    "Volatility_20D",
    "Volatility_60D",
    "MA_Distance_20D",
    "MA_Slope_20D",
    "Drawdown_60D",
]


# ============================================================
# M2 anomaly features
# ============================================================

ANOMALY_FEATURES = [
    "Return_1D",
    "Return_5D",
    "Volatility_5D",
    "Volatility_20D",
    "Volatility_60D",
    "MA_Distance_20D",
    "MA_Slope_20D",
    "Drawdown_60D",
    "Return_ZScore_20D",
    "Volatility_Ratio_5D_60D",
    "Drawdown_Change_5D",
]


# ============================================================
# Regime names
# ============================================================

REGIME_NAMES = {
    0: "Low-Volatility Trend",
    1: "Elevated-Volatility Transition",
    2: "High-Volatility Stress",
}


# ============================================================
# Statistical baseline
# ============================================================

RETURN_Z_THRESHOLD = 4.0
VOL_RATIO_THRESHOLD = 2.0
DRAWDOWN_CHANGE_THRESHOLD = 0.05
ABS_RETURN_THRESHOLD = 0.04


# ============================================================
# ML configuration
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 500
MAX_SAMPLES = 512

MIN_REGIME_OBSERVATIONS = 50


# ============================================================
# Load
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    required = {
        "Date",
        "USDTRY",
        *M1_FEATURES,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    return (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )


# ============================================================
# Baseline flags
# ============================================================

def baseline_flag(df):

    return (
        (
            df["Return_ZScore_20D"].abs()
            >= RETURN_Z_THRESHOLD
        )
        |
        (
            df["Volatility_Ratio_5D_60D"]
            >= VOL_RATIO_THRESHOLD
        )
        |
        (
            df["Drawdown_Change_5D"].abs()
            >= DRAWDOWN_CHANGE_THRESHOLD
        )
        |
        (
            df["Absolute_Return_1D"]
            >= ABS_RETURN_THRESHOLD
        )
    ).to_numpy(dtype=bool)


# ============================================================
# Gaussian HMM causal filtering
# ============================================================

def get_diag_covariances(model):

    covars = np.asarray(
        model.covars_
    )

    if covars.ndim == 2:

        return covars

    if covars.ndim == 3:

        return np.stack(
            [
                np.diag(c)
                for c in covars
            ]
        )

    raise ValueError(
        f"Unexpected covariance shape: "
        f"{covars.shape}"
    )


def emission_log_probability(
    model,
    X,
):

    means = np.asarray(
        model.means_
    )

    variances = (
        get_diag_covariances(
            model
        )
    )

    variances = np.maximum(
        variances,
        1e-12,
    )

    n_obs = len(X)

    n_states = (
        model.n_components
    )

    n_features = (
        X.shape[1]
    )

    log_prob = np.empty(
        (
            n_obs,
            n_states,
        )
    )

    constant = (
        n_features
        * np.log(
            2.0 * np.pi
        )
    )

    for state in range(
        n_states
    ):

        diff = (
            X
            - means[state]
        )

        log_prob[:, state] = (
            -0.5
            * (
                constant
                + np.sum(
                    np.log(
                        variances[state]
                    )
                )
                + np.sum(
                    (
                        diff ** 2
                    )
                    / variances[state],
                    axis=1,
                )
            )
        )

    return log_prob


def causal_filter(
    model,
    X,
    previous_posterior=None,
):

    emissions = (
        emission_log_probability(
            model,
            X,
        )
    )

    transition = np.asarray(
        model.transmat_
    )

    filtered = np.zeros(
        (
            len(X),
            model.n_components,
        )
    )

    log_likelihood = 0.0

    for t in range(
        len(X)
    ):

        if t == 0:

            if previous_posterior is None:

                prior = np.asarray(
                    model.startprob_
                )

            else:

                prior = (
                    previous_posterior
                    @ transition
                )

        else:

            prior = (
                filtered[t - 1]
                @ transition
            )

        prior = np.maximum(
            prior,
            1e-300,
        )

        log_alpha = (
            np.log(prior)
            + emissions[t]
        )

        norm = logsumexp(
            log_alpha
        )

        filtered[t] = np.exp(
            log_alpha - norm
        )

        log_likelihood += norm

    return (
        filtered,
        log_likelihood,
    )


# ============================================================
# Fit causal HMM
# ============================================================

def fit_regime_model(
    train,
    test,
):

    scaler = StandardScaler()

    X_train = (
        scaler.fit_transform(
            train[
                HMM_FEATURES
            ]
        )
    )

    X_test = (
        scaler.transform(
            test[
                HMM_FEATURES
            ]
        )
    )

    print(
        "\n--- TRAINING M1 HMM ---"
    )

    model, log_likelihood, seed = (
        fit_best_hmm(
            X_train
        )
    )

    # --------------------------------------------------------
    # Causal train filtering
    # --------------------------------------------------------

    train_prob, causal_log_likelihood = (
        causal_filter(
            model,
            X_train,
        )
    )

    difference = abs(
        causal_log_likelihood
        - model.score(
            X_train
        )
    )

    print(
        "\nCausal likelihood check:"
    )

    print(
        f"hmmlearn: "
        f"{model.score(X_train):.8f}"
    )

    print(
        f"causal:   "
        f"{causal_log_likelihood:.8f}"
    )

    print(
        f"difference: "
        f"{difference:.12f}"
    )

    if difference > 1e-4:

        raise RuntimeError(
            "HMM causal filtering check failed."
        )

    # --------------------------------------------------------
    # Causal test filtering
    # --------------------------------------------------------

    test_prob, _ = causal_filter(
        model,
        X_test,
        previous_posterior=
            train_prob[-1],
    )

    # --------------------------------------------------------
    # Canonical state mapping
    #
    # Sort by learned Volatility_20D mean.
    # Standardization preserves ordering.
    # --------------------------------------------------------

    vol_index = (
        HMM_FEATURES.index(
            "Volatility_20D"
        )
    )

    ordering = np.argsort(
        model.means_[
            :,
            vol_index,
        ]
    )

    native_to_canonical = {
        int(ordering[0]): 0,
        int(ordering[1]): 1,
        int(ordering[2]): 2,
    }

    train_native = (
        train_prob.argmax(
            axis=1
        )
    )

    test_native = (
        test_prob.argmax(
            axis=1
        )
    )

    train_regime = np.array(
        [
            native_to_canonical[
                int(state)
            ]
            for state in train_native
        ]
    )

    test_regime = np.array(
        [
            native_to_canonical[
                int(state)
            ]
            for state in test_native
        ]
    )

    return (
        train_regime,
        test_regime,
        train_prob.max(axis=1),
        test_prob.max(axis=1),
    )


# ============================================================
# OCSVM preprocessing
# ============================================================

def fit_ocsvm_preprocessor(
    X,
):

    median = np.median(
        X,
        axis=0,
    )

    q25 = np.quantile(
        X,
        0.25,
        axis=0,
    )

    q75 = np.quantile(
        X,
        0.75,
        axis=0,
    )

    iqr = (
        q75 - q25
    )

    iqr = np.where(
        iqr > 1e-12,
        iqr,
        1.0,
    )

    robust = (
        (X - median)
        / iqr
    )

    compressed = np.arcsinh(
        robust
    )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        compressed
    )

    return {
        "median": median,
        "iqr": iqr,
        "scaler": scaler,
        "X_train": scaled,
    }


def transform_ocsvm(
    X,
    prep,
):

    robust = (
        (
            X
            - prep["median"]
        )
        / prep["iqr"]
    )

    compressed = np.arcsinh(
        robust
    )

    return (
        prep["scaler"]
        .transform(
            compressed
        )
    )


# ============================================================
# Fit one IF detector
# ============================================================

def fit_if(
    X,
    anomaly_rate,
):

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=min(
            MAX_SAMPLES,
            len(X),
        ),
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X)

    scores = (
        -model.score_samples(X)
    )

    threshold = float(
        np.quantile(
            scores,
            1.0 - anomaly_rate,
        )
    )

    return {
        "model": model,
        "threshold": threshold,
    }


# ============================================================
# Fit one OCSVM detector
# ============================================================

def fit_ocsvm(
    X,
    anomaly_rate,
):

    prep = (
        fit_ocsvm_preprocessor(
            X
        )
    )

    nu = float(
        np.clip(
            anomaly_rate,
            0.01,
            0.25,
        )
    )

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=nu,
    )

    model.fit(
        prep["X_train"]
    )

    scores = (
        -model
        .decision_function(
            prep["X_train"]
        )
        .ravel()
    )

    threshold = float(
        np.quantile(
            scores,
            1.0 - anomaly_rate,
        )
    )

    return {
        "model": model,
        "prep": prep,
        "threshold": threshold,
    }


# ============================================================
# Score helpers
# ============================================================

def score_if(
    X,
    fitted,
):

    score = (
        -fitted["model"]
        .score_samples(X)
    )

    flag = (
        score
        >= fitted["threshold"]
    )

    return score, flag


def score_ocsvm(
    X,
    fitted,
):

    X_scaled = (
        transform_ocsvm(
            X,
            fitted["prep"],
        )
    )

    score = (
        -fitted["model"]
        .decision_function(
            X_scaled
        )
        .ravel()
    )

    flag = (
        score
        >= fitted["threshold"]
    )

    return score, flag


# ============================================================
# Fit global and regime-conditioned models
# ============================================================

def fit_anomaly_models(
    train,
):

    X_all = (
        train[
            ANOMALY_FEATURES
        ]
        .to_numpy(dtype=float)
    )

    baseline = (
        baseline_flag(train)
    )

    global_rate = float(
        baseline.mean()
    )

    models = {
        "global_if":
            fit_if(
                X_all,
                global_rate,
            ),

        "global_ocsvm":
            fit_ocsvm(
                X_all,
                global_rate,
            ),

        "regimes": {},
    }

    print(
        "\n--- REGIME-SPECIFIC TRAINING ---"
    )

    for regime in [
        0,
        1,
        2,
    ]:

        mask = (
            train[
                "Regime"
            ].to_numpy()
            == regime
        )

        subset = (
            train.loc[
                mask
            ]
        )

        if len(subset) < (
            MIN_REGIME_OBSERVATIONS
        ):

            raise RuntimeError(
                f"Too few observations for "
                f"{REGIME_NAMES[regime]}: "
                f"{len(subset)}"
            )

        X = (
            subset[
                ANOMALY_FEATURES
            ]
            .to_numpy(dtype=float)
        )

        rate = float(
            baseline[
                mask
            ].mean()
        )

        # Prevent pathological zero rates.
        rate = float(
            np.clip(
                rate,
                0.01,
                0.25,
            )
        )

        print(
            f"{REGIME_NAMES[regime]:32s} "
            f"N={len(subset):4d} | "
            f"baseline anomaly rate="
            f"{rate:.2%}"
        )

        models[
            "regimes"
        ][regime] = {
            "training_rate":
                rate,

            "if":
                fit_if(
                    X,
                    rate,
                ),

            "ocsvm":
                fit_ocsvm(
                    X,
                    rate,
                ),
        }

    return models


# ============================================================
# Score global and conditioned detectors
# ============================================================

def score_test(
    test,
    models,
):

    test = test.copy()

    X = (
        test[
            ANOMALY_FEATURES
        ]
        .to_numpy(dtype=float)
    )

    # Global
    (
        test["Global_IF_Score"],
        test["Global_IF_Anomaly"],
    ) = score_if(
        X,
        models["global_if"],
    )

    (
        test["Global_OCSVM_Score"],
        test["Global_OCSVM_Anomaly"],
    ) = score_ocsvm(
        X,
        models[
            "global_ocsvm"
        ],
    )

    # Conditioned
    test[
        "Conditioned_IF_Score"
    ] = np.nan

    test[
        "Conditioned_IF_Anomaly"
    ] = False

    test[
        "Conditioned_OCSVM_Score"
    ] = np.nan

    test[
        "Conditioned_OCSVM_Anomaly"
    ] = False

    for regime in [
        0,
        1,
        2,
    ]:

        mask = (
            test["Regime"]
            == regime
        )

        if not mask.any():

            continue

        X_regime = (
            test.loc[
                mask,
                ANOMALY_FEATURES,
            ]
            .to_numpy(dtype=float)
        )

        regime_models = (
            models[
                "regimes"
            ][regime]
        )

        if_score, if_flag = (
            score_if(
                X_regime,
                regime_models["if"],
            )
        )

        oc_score, oc_flag = (
            score_ocsvm(
                X_regime,
                regime_models[
                    "ocsvm"
                ],
            )
        )

        test.loc[
            mask,
            "Conditioned_IF_Score",
        ] = if_score

        test.loc[
            mask,
            "Conditioned_IF_Anomaly",
        ] = if_flag

        test.loc[
            mask,
            "Conditioned_OCSVM_Score",
        ] = oc_score

        test.loc[
            mask,
            "Conditioned_OCSVM_Anomaly",
        ] = oc_flag

    test[
        "Baseline_Anomaly"
    ] = baseline_flag(
        test
    )

    return test


# ============================================================
# Summary
# ============================================================

def create_summary(
    train,
    test,
    models,
):

    rows = []

    for regime in [
        0,
        1,
        2,
    ]:

        train_subset = (
            train[
                train["Regime"]
                == regime
            ]
        )

        test_subset = (
            test[
                test["Regime"]
                == regime
            ]
        )

        if len(test_subset) == 0:

            continue

        train_baseline_rate = (
            baseline_flag(
                train_subset
            ).mean()
        )

        row = {
            "Regime":
                regime,

            "Regime_Name":
                REGIME_NAMES[
                    regime
                ],

            "Train_N":
                len(train_subset),

            "Test_N":
                len(test_subset),

            "Train_Baseline_Rate":
                train_baseline_rate,

            "Test_Baseline_Rate":
                test_subset[
                    "Baseline_Anomaly"
                ].mean(),

            "Global_IF_Test_Rate":
                test_subset[
                    "Global_IF_Anomaly"
                ].mean(),

            "Conditioned_IF_Test_Rate":
                test_subset[
                    "Conditioned_IF_Anomaly"
                ].mean(),

            "Global_OCSVM_Test_Rate":
                test_subset[
                    "Global_OCSVM_Anomaly"
                ].mean(),

            "Conditioned_OCSVM_Test_Rate":
                test_subset[
                    "Conditioned_OCSVM_Anomaly"
                ].mean(),
        }

        row[
            "Global_IF_Calibration_Gap"
        ] = abs(
            row[
                "Global_IF_Test_Rate"
            ]
            - train_baseline_rate
        )

        row[
            "Conditioned_IF_Calibration_Gap"
        ] = abs(
            row[
                "Conditioned_IF_Test_Rate"
            ]
            - train_baseline_rate
        )

        row[
            "Global_OCSVM_Calibration_Gap"
        ] = abs(
            row[
                "Global_OCSVM_Test_Rate"
            ]
            - train_baseline_rate
        )

        row[
            "Conditioned_OCSVM_Calibration_Gap"
        ] = abs(
            row[
                "Conditioned_OCSVM_Test_Rate"
            ]
            - train_baseline_rate
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Jaccard
# ============================================================

def jaccard(a, b):

    a = np.asarray(
        a,
        dtype=bool,
    )

    b = np.asarray(
        b,
        dtype=bool,
    )

    union = (
        a | b
    ).sum()

    if union == 0:

        return np.nan

    return float(
        (
            a & b
        ).sum()
        / union
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n--- MILESTONE 2: "
        "REGIME-CONDITIONED VALIDATION ---"
    )

    df = load_data()

    raw_train = (
        df[df["Date"] <= TRAIN_END]
        .copy()
        .reset_index(drop=True)
    )

    volatility_floor = estimate_volatility_floor(
        raw_train,
        quantile=VOLATILITY_FLOOR_QUANTILE,
    )

    featured = validate_features(
        add_anomaly_features(
            df,
            volatility_floor=volatility_floor,
        )
    )

    train = (
        featured[featured["Date"] <= TRAIN_END]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        featured[featured["Date"] > TRAIN_END]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "\n--- CHRONOLOGICAL SPLIT ---"
    )

    print(
        f"Train: "
        f"{train['Date'].min().date()} "
        f"to "
        f"{train['Date'].max().date()} "
        f"({len(train):,})"
    )

    print(
        f"Test:  "
        f"{test['Date'].min().date()} "
        f"to "
        f"{test['Date'].max().date()} "
        f"({len(test):,})"
    )

    (
        train_regime,
        test_regime,
        train_confidence,
        test_confidence,
    ) = fit_regime_model(
        train,
        test,
    )

    train[
        "Regime"
    ] = train_regime

    test[
        "Regime"
    ] = test_regime

    train[
        "Regime_Name"
    ] = (
        train[
            "Regime"
        ].map(
            REGIME_NAMES
        )
    )

    test[
        "Regime_Name"
    ] = (
        test[
            "Regime"
        ].map(
            REGIME_NAMES
        )
    )

    train[
        "Regime_Confidence"
    ] = train_confidence

    test[
        "Regime_Confidence"
    ] = test_confidence

    print(
        "\n--- CAUSAL REGIME DISTRIBUTION ---"
    )

    for regime in [
        0,
        1,
        2,
    ]:

        train_n = int(
            (
                train["Regime"]
                == regime
            ).sum()
        )

        test_n = int(
            (
                test["Regime"]
                == regime
            ).sum()
        )

        print(
            f"{REGIME_NAMES[regime]:32s} "
            f"train={train_n:4d} | "
            f"test={test_n:4d}"
        )

    models = fit_anomaly_models(
        train
    )

    test = score_test(
        test,
        models,
    )

    summary = create_summary(
        train,
        test,
        models,
    )

    print(
        "\n--- REGIME-CONDITIONED RESULTS ---"
    )

    printable = (
        summary.copy()
    )

    rate_columns = [
        column
        for column
        in printable.columns
        if (
            "Rate" in column
            or "Gap" in column
        )
    ]

    for column in rate_columns:

        printable[column] *= 100

    print(
        printable
        .round(3)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Overall test rates
    # --------------------------------------------------------

    print(
        "\n--- OVERALL TEST ANOMALY RATES ---"
    )

    for column in [
        "Baseline_Anomaly",
        "Global_IF_Anomaly",
        "Conditioned_IF_Anomaly",
        "Global_OCSVM_Anomaly",
        "Conditioned_OCSVM_Anomaly",
    ]:

        print(
            f"{column:30s}: "
            f"{test[column].mean():.2%}"
        )

    # --------------------------------------------------------
    # Mean regime calibration gaps
    # --------------------------------------------------------

    print(
        "\n--- MEAN REGIME CALIBRATION GAP ---"
    )

    gap_columns = [
        "Global_IF_Calibration_Gap",
        "Conditioned_IF_Calibration_Gap",
        "Global_OCSVM_Calibration_Gap",
        "Conditioned_OCSVM_Calibration_Gap",
    ]

    for column in gap_columns:

        print(
            f"{column:38s}: "
            f"{summary[column].mean():.2%}"
        )

    # --------------------------------------------------------
    # Global vs conditioned agreement
    # --------------------------------------------------------

    print(
        "\n--- GLOBAL VS CONDITIONED AGREEMENT ---"
    )

    print(
        "Isolation Forest Jaccard: "
        f"{jaccard(
            test['Global_IF_Anomaly'],
            test['Conditioned_IF_Anomaly']
        ):.4f}"
    )

    print(
        "One-Class SVM Jaccard: "
        f"{jaccard(
            test['Global_OCSVM_Anomaly'],
            test['Conditioned_OCSVM_Anomaly']
        ):.4f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    test.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(SUMMARY_FILE)
    print(PREDICTIONS_FILE)

    print(
        "\nRegime-conditioned validation "
        "complete."
    )


if __name__ == "__main__":
    main()

