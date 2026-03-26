import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report
try:
    from . import helpers
except ImportError:
    import helpers
import os
import datetime

def run_bdt_analysis():
    # 1. Setup and Loading
    print("Starting BDT Analysis...")
    config_path = './src/config/config.yaml'
    
    # Load basic config
    region, control_type, event_type, scaling_option = helpers.load_run_config(config_path)
    # Force defaults if needed, or rely on config. Assuming config is good.
    
    base_path = helpers.get_SAMPLES_base_path()
    base_path = helpers.region_and_event_type_check(region, event_type, base_path)
    
    print(f"Loading data from {base_path} for region {region}...")
    
    sample_paths = helpers.get_SAMPLE_PATHS(base_path)
    
    # Define features
    features = ['met_tst', 'MetOHT', 'dLepR', 'dPhill', 'dMetZPhi']
    print(f"Training features: {features}")

    # Load and Label Data
    all_signals = []
    all_backgrounds = []
    
    # helpers.create_signal_and_background does this after loading all.
    
    samples = {}
    
    for sample_name, file_name in sample_paths.items():
        # Skip DATA for training
        if sample_name == 'DATA':
            continue
            
        print(f"Loading {sample_name}...")
        try:
             # Use reduced chunk size if needed from recent debugging
            helpers.chunk_size = 100000 
            df = helpers.load_data_filtering_event_type(base_path + file_name, event_type)
            df = helpers.calculate_derived_variables(df)
            df = helpers.apply_analysis_cuts(df, control_type)
            
            # Keep only necessary columns to save memory
            keep_cols = features + ['global_weight']
            # Ensure columns exist (some might be derived)
            available_cols = [c for c in keep_cols if c in df.columns]
            if len(available_cols) < len(keep_cols):
                print(f"Warning: Missing columns in {sample_name}. Found: {available_cols}")
            
            df = df[available_cols].copy() # Copy to defragment
            
            samples[sample_name] = df
        except Exception as e:
            print(f"Error loading {sample_name}: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
            
    # Apply scaling (getting weights right is important)
    scaling_factors = helpers.get_scaling_factors(scaling_option)
    samples = helpers.apply_scaling_factors(samples, scaling_factors)
    
    signal_dict, background_dict = helpers.create_signal_and_background(samples)
    
    # Concatenate
    for name, df in signal_dict.items():
        df['target'] = 1
        all_signals.append(df)
        
    for name, df in background_dict.items():
        df['target'] = 0
        all_backgrounds.append(df)
        
    if not all_signals or not all_backgrounds:
        print("Error: Missing signal or background data.")
        return

    df_sig = pd.concat(all_signals, ignore_index=True)
    df_bkg = pd.concat(all_backgrounds, ignore_index=True)
    
    # Combine
    full_df = pd.concat([df_sig, df_bkg], ignore_index=True)
    
    # Check for NaNs
    full_df = full_df.dropna(subset=features)
    
    X = full_df[features].values
    y = full_df['target'].values
    weights = full_df['global_weight'].abs().values # Trees prefer positive weights usually, or handle sign. 
    # For kinematic shape training, abs weight is often used to define "importance", but true separation should respect sign.
    # sklearn GBDT supports sample_weight. If we have neg weights (interference), it might be tricky.
    # Let's use abs() for training stability usually, or just raw if confident.
    # Given 'global_weight' can be negative in some NLO MC, using abs() is safer for simple BDTs unless handling interference explicitly.
    # Let's use raw weights and hope sklearn handles it (it typically does for gradient boosting, using them in loss).
    # But for now, let's stick to raw weights.
    weights = full_df['global_weight'].values

    # Handle negative weights caution: GBDT fits indices to minimize loss. 
    # If weights are negative, it maximizes loss? 
    # Standard practice: Take abs() for training presence, or ignore negative weights if small fraction.
    # Let's check yield.
    # For now, I will use abs(weights) for TRAINING to ensure stability.
    train_weights = np.abs(weights)

    # Split X, y, and BOTH the original weights and absolute training weights
    X_train, X_test, y_train, y_test, w_train_abs, w_test_abs, w_train_raw, w_test_raw = train_test_split(
        X, y, train_weights, weights, test_size=0.25, random_state=42, stratify=y
    )
    
    # 2. Training
    print("Training GradientBoostingClassifier...")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_train, y_train, sample_weight=w_train_abs)
    
    # 3. Evaluation
    print("Evaluating...")
    
    # Predictions
    y_score_test = clf.decision_function(X_test)
    y_score_train = clf.decision_function(X_train)
    
    # ROC Curve (Must use absolute weights for ROC geometry mathematically, since negative events invert True/False Positive Rates)
    fpr, tpr, _ = roc_curve(y_test, y_score_test, sample_weight=w_test_abs)
    roc_auc = auc(fpr, tpr)
    print(f"Test AUC: {roc_auc:.4f}")
    
    # Create Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/BDT_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()
    
    # Plot Score Distribution (Overtraining Check)
    plt.figure(figsize=(10, 6))
    
    # Signal Train/Test
    plt.hist(y_score_train[y_train==1], bins=40, range=(-5, 5), density=True, 
             weights=w_train_raw[y_train==1], alpha=0.5, color='blue', label='Signal (Train)')
    plt.hist(y_score_test[y_test==1], bins=40, range=(-5, 5), density=True, 
             weights=w_test_raw[y_test==1], histtype='step', linewidth=2, color='blue', linestyle='--', label='Signal (Test)')
             
    # Background Train/Test
    plt.hist(y_score_train[y_train==0], bins=40, range=(-5, 5), density=True, 
             weights=w_train_raw[y_train==0], alpha=0.5, color='red', label='Background (Train)')
    plt.hist(y_score_test[y_test==0], bins=40, range=(-5, 5), density=True, 
             weights=w_test_raw[y_test==0], histtype='step', linewidth=2, color='red', linestyle='--', label='Background (Test)')
             
    plt.xlabel('BDT Score')
    plt.ylabel('Normalized Density')
    plt.title('BDT Score Distribution (Train vs Test)')
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/score_distribution.png")
    plt.close()
    
    # Feature Importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    plt.xlim([-1, X.shape[1]])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()

    # NEW: Plot Score Distribution (Log Scale, Yields)
    plt.figure(figsize=(10, 6))
    
    # Calculate bins manually to ensure same binning
    bins = np.linspace(-5, 5, 41)
    
    # Background Train/Test (Yields)
    plt.hist(y_score_test[y_test==0], bins=bins, density=False, 
             weights=w_test_raw[y_test==0], histtype='stepfilled', alpha=0.3, color='red', label='Background (Test)')
             
    # Signal Train/Test (Yields)
    plt.hist(y_score_test[y_test==1], bins=bins, density=False, 
             weights=w_test_raw[y_test==1], histtype='step', linewidth=2, color='blue', label='Signal (Test)')
             
    plt.xlabel('BDT Score')
    plt.ylabel('Events (Weighted Yields)')
    plt.title('BDT Score Distribution (Test Set Yields)')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.savefig(f"{output_dir}/score_distribution_log_yield.png")
    plt.close()
    
    # 4. Significance Optimization
    print("Optimizing BDT Cut for Significance...")
    # Calculate Scale Factor for Projection (Test Size -> Full Data)
    # Test Size was 0.25. So Factor = 1 / 0.25 = 4 for Yields.
    # Sig scales with sqrt(Yields) -> sqrt(4) = 2.
    # Actually, we can just project yields to full luminosity first.
    test_fraction = 0.25
    scale_factor = 1.0 / test_fraction
    
    optimize_bdt_cut(clf, X_test, y_test, w_test_raw, output_dir, scale_factor)
    pass

    print(f"Analysis complete. Results saved to {output_dir}")

def optimize_bdt_cut(clf, X, y, weights, output_dir, scale_factor=1.0):
    """
    Scan BDT score thresholds to maximize significance.
    Sig = sqrt(2 * ((S+B) * ln(1 + S/B) - S))
    """
    # Get scores
    scores = clf.decision_function(X)
    
    # Define range of cuts to scan
    min_score = np.min(scores)
    max_score = np.max(scores)
    cuts = np.linspace(min_score, max_score, 100)
    
    significances = []
    valid_cuts = []
    
    max_significance = 0
    optimal_cut = min_score
    
    S_at_max = 0
    B_at_max = 0
    
    for cut in cuts:
        mask = scores > cut
        
        # Calculate Signal and Background yields passing the cut
        # We must use TRUE signed weights for significance to account for destructive interference
        # in NLO MC backgrounds, even if the BDT was trained on abs() weights.
        
        # Project Yields to Full Dataset Size
        S_test = np.sum(weights[(y==1) & mask])
        B_test = np.sum(weights[(y==0) & mask])
        
        S = S_test * scale_factor
        B = B_test * scale_factor
        
        # Calculate Significance
        if B <= 0 or S < 0:
            sig = 0
        else:
            try:
                term = 1 + (S / B)
                if term <= 0:
                    sig = 0
                else:
                    sig = np.sqrt(2 * ((S + B) * np.log(term) - S))
            except Exception:
                sig = 0
                
        # Store
        if sig > 0 and np.isfinite(sig):
            significances.append(sig)
            valid_cuts.append(cut)
            
            if sig > max_significance:
                max_significance = sig
                optimal_cut = cut
                S_at_max = S
                B_at_max = B
    
    print(f"Optimization Results (Projected to Full Dataset):")
    print(f"  Maximum Significance: {max_significance:.4f}")
    print(f"  Optimal BDT Cut: > {optimal_cut:.4f}")
    print(f"  Yields at Optimal Cut (Full): S = {S_at_max:.2f}, B = {B_at_max:.2f}")
    
    # Plot Significance vs Cut
    if valid_cuts:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_cuts, significances, marker='o', linestyle='-', markersize=4, color='green')
        plt.axvline(x=optimal_cut, color='red', linestyle='--', label=f'Optimal Cut: {optimal_cut:.2f}')
        plt.axhline(y=max_significance, color='red', linestyle=':', label=f'Max Sig: {max_significance:.2f}')
        
        plt.xlabel('BDT Score Threshold')
        plt.ylabel('Significance')
        plt.title('Significance vs BDT Cut')
        plt.legend()
        plt.grid()
        plt.savefig(f"{output_dir}/significance_scan.png")
        plt.close()
    else:
        print("Warning: No valid significance values found during scan.")

if __name__ == "__main__":
    run_bdt_analysis()
