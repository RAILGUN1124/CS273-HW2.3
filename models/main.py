import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
import warnings


SEED = 1935990857
USE_WORDNET_LEMMATIZATION = False
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def ensure_wordnet_resources():
    """Download WordNet resources only when lemmatization is enabled."""
    import nltk

    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def build_tfidf_vectorizer(use_wordnet=False):
    if not use_wordnet:
        # return TfidfVectorizer(stop_words='english', max_features=5000)
        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2), 
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            max_features=5000,
            stop_words=None  # or custom list that keeps "not", "no", "never"
        )

    try:
        from nltk.stem import WordNetLemmatizer
    except ImportError as exc:
        raise ImportError(
            "WordNet lemmatization requires nltk. Install it with: pip install nltk"
        ) from exc

    ensure_wordnet_resources()
    lemmatizer = WordNetLemmatizer()

    # Keep tokenization lightweight and deterministic without extra NLTK tokenizers.
    def wordnet_tokenizer(text):
        tokens = re.findall(r"\b\w+\b", str(text).lower())
        return [lemmatizer.lemmatize(token) for token in tokens]

    return TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        tokenizer=wordnet_tokenizer,
        token_pattern=None
    )


# def build_model_pipeline(clf, use_wordnet=False):
#     return Pipeline([
#         ('tfidf', build_tfidf_vectorizer(use_wordnet=use_wordnet)),
#         ('smote', SMOTE(random_state=SEED)),
#         ('model', clf)
#     ])
def build_model_pipeline(clf, use_wordnet=False):
    return Pipeline([
        ('tfidf', build_tfidf_vectorizer(use_wordnet=use_wordnet)),
        ('oversample', RandomOverSampler(random_state=SEED)),
        ('model', clf)
    ])

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        # Fallback for relative path if running from models/
        df = pd.read_excel(os.path.join('../Data', os.path.basename(filepath)))
        
    # Rename columns for clarity
    # Expected columns based on inspection: 'Review', 'positive=1/negative=0'
    # We map 'positive=1/negative=0' to 'label'
    
    # Check for correct columns
    if 'positive=1/negative=0' in df.columns:
        df = df.rename(columns={'Review': 'text', 'positive=1/negative=0': 'label'})
    else:
        # Fallback or error
        print("Required columns not found. Printing columns:")
        print(df.columns)
        raise ValueError("Column 'positive=1/negative=0' not found")

    # Drop NaNs
    df = df.dropna(subset=['text', 'label'])
    
    # Ensure label is int
    df['label'] = df['label'].astype(int)
    
    # Check distribution
    print("Label distribution before split:")
    print(df['label'].value_counts(normalize=True))
    
    return df

def objective(trial, model_name, X, y, use_wordnet=False):
    """
    Optuna objective function for hyperparameter tuning.
    Uses Pipeline(SMOTE, Classifier) to prevent data leakage.
    """
    if model_name == 'Logistic Regression':
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        # Using lbfgs solver (default) which supports l2
        clf = LogisticRegression(C=C, max_iter=1000, random_state=SEED, class_weight='balanced')
        
    elif model_name == 'Random Forest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            random_state=SEED, class_weight='balanced'
        )
        
    elif model_name == 'MLP / Neural Network':
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)])
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True)
        alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, 
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            max_iter=500, 
            random_state=SEED
        )
    
    # Pipeline: TF-IDF -> SMOTE -> Classifier
    pipeline = build_model_pipeline(clf, use_wordnet=use_wordnet)
    
    # 5-fold CV maximizing F1-score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
    return scores.mean()

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        print(f"Evaluating {name}...")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Training Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        
        # Test Metrics
        acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        
        results.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Train F1': train_f1,
            'Test Accuracy': acc,
            'Test F1-Score': f1,
            'Test Precision': precision,
            'Test Recall': recall
        })
        
        print(f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}")
        print(classification_report(y_test, y_test_pred))
        
        safe_name = name.replace(" ", "_").replace("/", "_")
        
        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        fig.suptitle(f'{name} Performance Analysis', fontsize=16)

        # 1. Train Confusion Matrix
        cm_train = confusion_matrix(y_train, y_train_pred)
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Train Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # 2. Test Confusion Matrix
        cm_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('Test Confusion Matrix')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        # 3. Traing Per-Class F1 Score Bar Chart
        clf_report_train = classification_report(y_train, y_train_pred, output_dict=True)
        f1_class_0_train = clf_report_train['0']['f1-score']
        f1_class_1_train = clf_report_train['1']['f1-score']
        
        metrics_train = ['Class 0 F1', 'Class 1 F1']
        values_train = [f1_class_0_train, f1_class_1_train]
        
        sns.barplot(x=metrics_train, y=values_train, hue=metrics_train, palette='Blues', ax=axes[2], legend=False)
        axes[2].set_ylim(0, 1.1)
        axes[2].set_title('Train F1-Score per Class')
        axes[2].set_ylabel('F1-Score')
        
        for i, v in enumerate(values_train):
            axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

        # 4. Test Per-Class F1 Score Bar Chart
        clf_report = classification_report(y_test, y_test_pred, output_dict=True)
        # Check keys - they are usually strings '0', '1' unless target_names is passed
        f1_class_0 = clf_report['0']['f1-score']
        f1_class_1 = clf_report['1']['f1-score']
        
        metrics = ['Class 0 F1', 'Class 1 F1']
        values = [f1_class_0, f1_class_1]
        
        sns.barplot(x=metrics, y=values, hue=metrics, palette='viridis', ax=axes[3], legend=False)
        axes[3].set_ylim(0, 1.1)
        axes[3].set_title('Test F1-Score per Class')
        axes[3].set_ylabel('F1-Score')
        
        # Add values on top of bars
        for i, v in enumerate(values):
            axes[3].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
            
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/{safe_name}_combined.png')
        plt.close()
        
    return pd.DataFrame(results)

def main():
    filepath = 'Data/restaurant_reviews-v2-1.xlsx'
    # Adjust path if script is run from project root
    if not os.path.exists(filepath):
         filepath = os.path.join('..', filepath)
         
    df = load_and_preprocess_data(filepath)
    
    # 1. Data Split (80% Train, 20% Test, Stratified)
    print("\nSplitting data...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['text'], df['label'], 
        test_size=0.2, 
        stratify=df['label'], 
        random_state=SEED,
        shuffle=True
    )
    
    print(f"Train size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")
    
    print("\nVerifying Stratified Split (Class Distribution):")
    print("Train Set Distribution:")
    print(y_train.value_counts(normalize=True))
    print("Test Set Distribution:")
    print(y_test.value_counts(normalize=True))
    
    # 2. Preprocessing & Vectorization
    # print("\nVectorizing text...")
    # vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    # # X_train_vec = vectorizer.fit_transform(X_train_raw)
    # # X_test_vec = vectorizer.transform(X_test_raw)
    
    # 4. Hyperparameter Tuning with Optuna
    print("\n=== Hyperparameter Tuning with Optuna ===")
    print(f"WordNet lemmatization enabled: {USE_WORDNET_LEMMATIZATION}")
    
    model_names = ['Logistic Regression', 'Random Forest', 'MLP / Neural Network']
    best_models = {}

    for model_name in model_names:
        print(f"\n--- Tuning {model_name} ---")
        
        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        study.optimize(
            lambda trial: objective(
                trial,
                model_name,
                X_train_raw,
                y_train,
                use_wordnet=USE_WORDNET_LEMMATIZATION
            ),
            n_trials=50   # reduce from 100 → less overfitting
        )
        
        print(f"Best params: {study.best_params}")
        print(f"Best CV F1: {study.best_value:.4f}")

        # Rebuild model with best params
        if model_name == 'Logistic Regression':
            clf = LogisticRegression(
                C=study.best_params['C'],
                max_iter=1000,
                random_state=SEED,
                class_weight='balanced'
            )
            
        elif model_name == 'Random Forest':
            clf = RandomForestClassifier(
                n_estimators=study.best_params['n_estimators'],
                max_depth=study.best_params['max_depth'],
                min_samples_split=study.best_params['min_samples_split'],
                random_state=SEED,
                class_weight='balanced'
            )
            
        elif model_name == 'MLP / Neural Network':
            clf = MLPClassifier(
                hidden_layer_sizes=study.best_params['hidden_layer_sizes'],
                learning_rate_init=study.best_params['learning_rate_init'],
                alpha=study.best_params['alpha'],
                max_iter=500,
                random_state=SEED,
            )

        # Final pipeline (same as CV)
        best_models[model_name] = build_model_pipeline(
            clf,
            use_wordnet=USE_WORDNET_LEMMATIZATION
        )
            
    # # 3. Apply SMOTE to training data (Separate step for Final Training)
    # print("\nApplying SMOTE to full training set for final model training...")
    # smote = SMOTE(random_state=SEED)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)
    
    # print("New Train Set Distribution after SMOTE:")
    # print(y_train_resampled.value_counts(normalize=True))

    # 5. Training & Evaluation
    print("\n=== Final Training & Evaluation ===")
    results_df = train_and_evaluate(best_models, X_train_raw, y_train, X_test_raw, y_test)
    
    print("\n=== Final Results Table ===")
    print(results_df)
    
    # Save results to CSV (optional but good practice)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/comparison_metrics.csv', index=False)
    print("\nResults and plots saved to results/")

if __name__ == "__main__":
    main()
