import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize, Bounds, LinearConstraint
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ---------- PATHS ----------
REVIEWS_CSV = os.path.join("steam_dataset_2025_csv/reviews.csv")
APP_CATS_CSV = os.path.join("steam_dataset_2025_csv/application_categories.csv")
CATEGORIES_CSV = os.path.join("steam_dataset_2025_csv/categories.csv")

for p, label in [(REVIEWS_CSV, "reviews.csv"),
                 (APP_CATS_CSV, "application_categories.csv")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{label} not found at: {p}")
    else:
        print("✓ Found:", p)

if os.path.exists(CATEGORIES_CSV):
    print("✓ Found categories.csv (will use for names)")
else:
    print("⚠ categories.csv not found; using category_id only")


# ---------- HYPERPARAMETERS ----------
RANDOM_STATE = 42
TOP_N_CATEGORIES = 20
APPID_COL = "appid"
VOTED_UP_COL = "voted_up"


# ============================================
# DATA LOADING / PREP
# ============================================
def load_and_prepare_data():
    reviews = pd.read_csv(REVIEWS_CSV)
    app_cats = pd.read_csv(APP_CATS_CSV)

    if APPID_COL not in reviews.columns or APPID_COL not in app_cats.columns:
        raise KeyError("Both reviews and application_categories must have 'appid'")

    if VOTED_UP_COL not in reviews.columns:
        raise KeyError(f"reviews.csv must contain '{VOTED_UP_COL}' column")

    if "category_id" not in app_cats.columns:
        raise KeyError("application_categories.csv must contain 'category_id'")

    # Binary target
    reviews["positive"] = reviews[VOTED_UP_COL].astype(int)

    # Average positive score per app
    score_df = (
        reviews.groupby(APPID_COL)["positive"]
        .mean()
        .rename("review_score")
        .reset_index()
    )

    # Find top N categories
    top_cats = (
        app_cats["category_id"]
        .value_counts()
        .head(TOP_N_CATEGORIES)
        .index
        .tolist()
    )

    # One-hot encode categories
    cat_table = (
        app_cats[app_cats["category_id"].isin(top_cats)]
        .assign(value=1)
        .pivot_table(
            index=APPID_COL,
            columns="category_id",
            values="value",
            fill_value=0,
        )
    )

    # Rename columns
    cat_col_map = {cid: f"cat_{cid}" for cid in cat_table.columns}
    cat_table = cat_table.rename(columns=cat_col_map).reset_index()

    # Merge with target
    df = cat_table.merge(score_df, on=APPID_COL, how="inner")

    feature_cols = [c for c in df.columns if c.startswith("cat_")]
    X = df[feature_cols].astype(np.float32).values
    y = df["review_score"].astype(np.float32).values

    print(f"Total apps with categories & reviews: {len(df)}")
    print("Number of category features:", len(feature_cols))

    # Optional: map category id → name
    cat_name_map = {}
    if os.path.exists(CATEGORIES_CSV):
        cats_df = pd.read_csv(CATEGORIES_CSV)
        if "id" in cats_df.columns and "description" in cats_df.columns:
            id_to_name = dict(zip(cats_df["id"], cats_df["description"]))
            for cid, col_name in cat_col_map.items():
                cat_name_map[col_name] = id_to_name.get(cid, str(cid))

    return X, y, feature_cols, cat_name_map


# ============================================
# CONVEX LINEAR MODEL TRAINING
# ============================================
def train_convex_model(X_train, y_train, X_val, y_val):
    """
    Train Ridge regression (L2 regularization) for convex optimization.
    Ridge ensures positive semi-definite Hessian → convex objective.
    """
    print("\n=== Training Convex Linear Model (Ridge Regression) ===")
    
    # Ridge with positive=True ensures all coefficients are non-negative
    # This makes the maximization problem convex
    model = Ridge(alpha=0.1, positive=True, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Train MSE: {train_mse:.4f} | R²: {train_r2:.4f}")
    print(f"Val MSE: {val_mse:.4f} | R²: {val_r2:.4f}")
    
    return model


# ============================================
# CONVEX OPTIMIZATION (RELAXED + EXACT)
# ============================================
def optimize_tags(model, feature_cols, cat_name_map, config):
    """
    Two-stage optimization:
    1. Solve relaxed convex problem (continuous x ∈ [0,1])
    2. Round to binary solution respecting all constraints
    """
    w = model.coef_
    b = model.intercept_
    n = len(w)
    
    MAX_TAGS = config['max_tags']
    REQUIRED_TAGS = config['required_tags']
    EXCLUSIVE_PAIRS = config['exclusive_pairs']
    LAMBDA = config['lambda']
    
    # Map feature names to indices
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    
    required_idx = [col_to_idx[c] for c in REQUIRED_TAGS if c in col_to_idx]
    exclusive_idx = [(col_to_idx[a], col_to_idx[b])
                    for a, b in EXCLUSIVE_PAIRS
                    if a in col_to_idx and b in col_to_idx]
    
    print(f"\n=== Solving Convex Optimization ===")
    print(f"Max tags: {MAX_TAGS}")
    print(f"Required tags: {REQUIRED_TAGS}")
    print(f"L1 penalty (λ): {LAMBDA}")
    print(f"All coefficients non-negative: {np.all(w >= 0)}")
    
    # Show top coefficients for context
    top_k = 10
    top_idx = np.argsort(-w)[:top_k]
    print(f"\nTop {top_k} tag coefficients (impact on review score):")
    for rank, idx in enumerate(top_idx, 1):
        name = cat_name_map.get(feature_cols[idx], feature_cols[idx])
        print(f"  {rank}. {feature_cols[idx]}: {w[idx]:.4f} - {name}")
    
    # ===== STAGE 1: RELAXED CONVEX OPTIMIZATION =====
    def objective(x):
        predicted_score = np.dot(w, x) + b
        l1_penalty = LAMBDA * np.sum(x)
        return -predicted_score + l1_penalty
    
    def gradient(x):
        return -w + LAMBDA * np.ones(n)
    
    # CONSTRAINTS
    constraints = []
    
    # 1. Max tags: sum(x) ≤ MAX_TAGS
    constraints.append(LinearConstraint(np.ones(n), 0, MAX_TAGS))
    
    # 2. Required tags: x[i] = 1
    for i in required_idx:
        A = np.zeros(n)
        A[i] = 1
        constraints.append(LinearConstraint(A, 1, 1))
    
    # 3. Exclusive pairs: x[i] + x[j] ≤ 1
    for i, j in exclusive_idx:
        A = np.zeros(n)
        A[i] = 1
        A[j] = 1
        constraints.append(LinearConstraint(A, 0, 1))
    
    # Bounds: relaxed binary [0, 1]
    bounds = Bounds([0]*n, [1]*n)
    
    # Initial guess: prioritize high-weight tags
    x0 = np.zeros(n)
    for i in required_idx:
        x0[i] = 1.0
    # Add some randomness to help exploration
    x0 += np.random.rand(n) * 0.1
    x0 = np.clip(x0, 0, 1)
    
    # Solve relaxed problem
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        print(f"⚠ Relaxed optimization warning: {result.message}")
    
    x_relaxed = result.x
    
    print(f"\nRelaxed solution: {np.sum(x_relaxed > 0.01):.0f} tags with non-zero values")
    print(f"Relaxed objective value: {result.fun:.4f}")
    
    # ===== STAGE 2: GREEDY BINARY ROUNDING =====
    # Sort by relaxed solution values (descending)
    sorted_idx = np.argsort(-x_relaxed)
    
    # Binary solution
    x_binary = np.zeros(n)
    
    # First, add all required tags
    for i in required_idx:
        x_binary[i] = 1
    
    tags_added = len(required_idx)
    
    # Then greedily add highest-value tags respecting constraints
    for idx in sorted_idx:
        if tags_added >= MAX_TAGS:
            break
        if x_binary[idx] == 1:  # Already added (required tag)
            continue
        
        # Check exclusive constraints
        valid = True
        for i, j in exclusive_idx:
            if idx == i and x_binary[j] == 1:
                valid = False
                break
            if idx == j and x_binary[i] == 1:
                valid = False
                break
        
        if valid:
            x_binary[idx] = 1
            tags_added += 1
    
    # Extract selected tags
    selected_tags = [feature_cols[i] for i in range(n) if x_binary[i] > 0.5]
    
    # Predictions
    pred_relaxed = np.dot(w, x_relaxed) + b
    pred_binary = np.dot(w, x_binary) + b
    
    # Baseline: no tags
    pred_baseline = b
    
    return {
        'x_relaxed': x_relaxed,
        'x_binary': x_binary,
        'selected_tags': selected_tags,
        'pred_relaxed': pred_relaxed,
        'pred_binary': pred_binary,
        'pred_baseline': pred_baseline,
        'weights': w,
        'objective_relaxed': result.fun,
        'objective_binary': objective(x_binary),
        'success': result.success
    }


# ============================================
# VISUALIZATION
# ============================================
def visualize_results(result, feature_cols, cat_name_map):
    """Visualize optimization results with meaningful insights"""
    x_binary = result['x_binary']
    x_relaxed = result['x_relaxed']
    selected_tags = result['selected_tags']
    w = result['weights']
    
    print("\n" + "="*70)
    print("OPTIMAL TAG COMBINATION")
    print("="*70)
    for i, tag in enumerate(selected_tags, 1):
        name = cat_name_map.get(tag, tag)
        idx = [j for j, c in enumerate(feature_cols) if c == tag][0]
        contribution = w[idx]
        print(f"{i}. {tag}: {name}")
        print(f"   Relaxed value: {x_relaxed[idx]:.3f} | Weight: {contribution:.4f}")
    
    print("\n" + "="*70)
    print("PREDICTED REVIEW SCORES")
    print("="*70)
    print(f"Baseline (no tags):   {result['pred_baseline']:.4f}")
    print(f"Relaxed solution:     {result['pred_relaxed']:.4f} (improvement: {result['pred_relaxed']-result['pred_baseline']:+.4f})")
    print(f"Binary solution:      {result['pred_binary']:.4f} (improvement: {result['pred_binary']-result['pred_baseline']:+.4f})")
    print(f"Number of tags used:  {len(selected_tags)}")
    print(f"Optimization success: {result['success']}")
    
    # Create informative visualizations
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Tag weights sorted
    ax1 = fig.add_subplot(gs[0, :])
    sorted_w_idx = np.argsort(-w)
    ax1.bar(range(len(w)), w[sorted_w_idx], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Tag (sorted by weight)')
    ax1.set_ylabel('Weight (impact on review score)')
    ax1.set_title('Tag Weights from Ridge Regression (sorted)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Relaxed solution
    ax2 = fig.add_subplot(gs[1, 0])
    colors_relaxed = ['red' if x_binary[i] > 0.5 else 'skyblue' for i in range(len(x_relaxed))]
    ax2.bar(range(len(feature_cols)), x_relaxed, color=colors_relaxed, alpha=0.7)
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Rounding threshold')
    ax2.set_xlabel('Tag Index')
    ax2.set_ylabel('Relaxed Value [0,1]')
    ax2.set_title('Relaxed Convex Solution (red = selected in binary)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Binary solution
    ax3 = fig.add_subplot(gs[1, 1])
    colors_binary = ['green' if x > 0.5 else 'lightgray' for x in x_binary]
    ax3.bar(range(len(feature_cols)), x_binary, color=colors_binary, alpha=0.7)
    ax3.set_xlabel('Tag Index')
    ax3.set_ylabel('Binary Selection (0 or 1)')
    ax3.set_title('Rounded Binary Solution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Selected tags with their contributions
    ax4 = fig.add_subplot(gs[2, :])
    selected_idx = [i for i in range(len(x_binary)) if x_binary[i] > 0.5]
    selected_weights = [w[i] for i in selected_idx]
    selected_names = [cat_name_map.get(feature_cols[i], feature_cols[i])[:30] for i in selected_idx]
    
    bars = ax4.barh(range(len(selected_idx)), selected_weights, color='green', alpha=0.7)
    ax4.set_yticks(range(len(selected_idx)))
    ax4.set_yticklabels(selected_names)
    ax4.set_xlabel('Weight (contribution to review score)')
    ax4.set_title('Selected Tags and Their Impact')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, selected_weights)):
        ax4.text(val, i, f' {val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # Load data
    X, y, feature_cols, cat_name_map = load_and_prepare_data()
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Train convex model
    model = train_convex_model(X_train, y_train, X_val, y_val)
    
    # Configuration for optimization
    config = {
        'max_tags': 5,
        'required_tags': ['cat_1', 'cat_2'],  # Edit these based on your data
        'exclusive_pairs': [('cat_9', 'cat_10')],  # Edit these
        'lambda': 0.01  # REDUCED: Lower penalty = optimizer selects more useful tags
    }
    
    # Optimize
    result = optimize_tags(model, feature_cols, cat_name_map, config)
    
    # Visualize
    visualize_results(result, feature_cols, cat_name_map)