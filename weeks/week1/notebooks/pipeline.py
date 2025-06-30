# Import all necessary libraries
print("🚀 Setting up Complete ML Pipeline Environment")
print("=" * 50)

# Core ML and data libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Metaflow for MLOps
from metaflow import FlowSpec, step, Parameter, catch

# Utilities
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
print("📊 Ready to build production ML pipeline")
print("🎯 Target: Wine classification with comprehensive evaluation")

class CompleteMLPipeline(FlowSpec):
    """
    Production-ready ML pipeline for wine classification
    
    Features:
    - Comprehensive data preprocessing
    - Multiple algorithm comparison
    - Cross-validation and robust evaluation
    - Automated report generation
    - Production-ready model artifacts
    """
    
    # Configurable parameters
    test_size = Parameter('test_size',
                         help='Test set proportion (0.1-0.4)',
                         default=0.2,
                         type=float)
    
    random_state = Parameter('random_state',
                           help='Random seed for reproducibility',
                           default=42,
                           type=int)
    
    cv_folds = Parameter('cv_folds',
                        help='Number of cross-validation folds',
                        default=5,
                        type=int)
    
    models_to_test = Parameter('models',
                              help='Comma-separated list of models',
                              default='random_forest,logistic_regression,svm')
    
    @step
    def start(self):
        """
        Initialize pipeline with data loading and validation
        """
        print("🍷 Starting Complete Wine Classification Pipeline")
        print("=" * 50)
        print(f"📊 Configuration:")
        print(f"   Test size: {self.test_size}")
        print(f"   Random state: {self.random_state}")
        print(f"   CV folds: {self.cv_folds}")
        print(f"   Models: {self.models_to_test}")
        
        # Parameter validation
        if not (0.1 <= self.test_size <= 0.4):
            raise ValueError(f"test_size must be between 0.1 and 0.4, got {self.test_size}")
        
        # Load wine dataset
        wine_data = load_wine()
        
        # Store raw data and metadata
        self.X_raw = wine_data.data
        self.y_raw = wine_data.target
        self.feature_names = wine_data.feature_names
        self.target_names = wine_data.target_names
        
        # Create dataset info
        self.dataset_info = {
            'n_samples': self.X_raw.shape[0],
            'n_features': self.X_raw.shape[1],
            'n_classes': len(np.unique(self.y_raw)),
            'class_distribution': np.bincount(self.y_raw).tolist()
        }
        
        print(f"\n📈 Dataset Overview:")
        print(f"   Samples: {self.dataset_info['n_samples']}")
        print(f"   Features: {self.dataset_info['n_features']}")
        print(f"   Classes: {self.dataset_info['n_classes']}")
        
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        """
        Data preprocessing pipeline
        """
        print("\n🔧 Data Preprocessing...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_raw, self.y_raw,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y_raw
        )
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   📊 Train/Test split: {len(self.X_train)}/{len(self.X_test)}")
        print(f"   📏 Features scaled using StandardScaler")
        
        self.next(self.train_models)
    
    @catch(var='training_errors')
    @step
    def train_models(self):
        """
        Train and compare multiple ML algorithms
        """
        print("\n🤖 Training Multiple ML Models...")
        
        # Parse model list
        model_names = [name.strip() for name in self.models_to_test.split(',')]
        
        # Define model configurations
        model_configs = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True
            )
        }
        
        # Train models
        self.model_results = {}
        self.training_errors = {}
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name in model_names:
            if model_name in model_configs:
                try:
                    print(f"   🔨 Training {model_name}...")
                    
                    model = model_configs[model_name]
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, self.X_train_scaled, self.y_train,
                        cv=cv, scoring='accuracy'
                    )
                    
                    # Fit and evaluate
                    model.fit(self.X_train_scaled, self.y_train)
                    test_accuracy = model.score(self.X_test_scaled, self.y_test)
                    
                    self.model_results[model_name] = {
                        'model': model,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'test_accuracy': test_accuracy
                    }
                    
                    print(f"      CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                    print(f"      Test: {test_accuracy:.3f}")
                    
                except Exception as e:
                    print(f"      ❌ Training failed: {str(e)}")
                    self.training_errors[model_name] = str(e)
        
        print(f"\n   ✅ Successfully trained {len(self.model_results)} models")
        
        self.next(self.evaluate)
    
    @step
    def evaluate(self):
        """
        Evaluate models and select best performer
        """
        print("\n📊 Model Evaluation...")
        
        if not self.model_results:
            print("   ❌ No models to evaluate")
            self.best_model_name = None
            self.next(self.end)
            return
        
        # Find best model
        best_model_name = max(self.model_results.keys(),
                            key=lambda x: self.model_results[x]['cv_mean'])
        
        self.best_model_name = best_model_name
        best_results = self.model_results[best_model_name]
        
        print(f"   🏆 Best model: {best_model_name}")
        print(f"   📈 CV score: {best_results['cv_mean']:.3f} ± {best_results['cv_std']:.3f}")
        print(f"   🎯 Test accuracy: {best_results['test_accuracy']:.3f}")
        
        # Generate predictions for evaluation
        best_model = best_results['model']
        self.y_pred = best_model.predict(self.X_test_scaled)
        
        # Classification report
        self.classification_report = classification_report(
            self.y_test, self.y_pred,
            target_names=self.target_names,
            output_dict=True
        )
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Finalize pipeline
        """
        print("\n🎉 Complete ML Pipeline Finished!")
        print("=" * 40)
        
        if self.best_model_name:
            best_results = self.model_results[self.best_model_name]
            
            print("📊 Pipeline Summary:")
            print(f"   🏆 Best Model: {self.best_model_name}")
            print(f"   🎯 Accuracy: {best_results['test_accuracy']:.3f}")
            print(f"   📈 CV Score: {best_results['cv_mean']:.3f}")
            print(f"   🤖 Models Trained: {len(self.model_results)}")
            
            # Performance assessment
            accuracy = best_results['test_accuracy']
            if accuracy > 0.95:
                print("   ✅ Excellent performance - ready for production!")
            elif accuracy > 0.9:
                print("   ✅ Very good performance")
            elif accuracy > 0.8:
                print("   ⚠️ Good performance - consider improvements")
            else:
                print("   ❌ Performance needs improvement")
        else:
            print("❌ Pipeline execution failed")
        
        print("\n✨ All artifacts saved by Metaflow!")
if __name__ == '__main__':
    CompleteMLPipeline()

print("✅ CompleteMLPipeline class defined successfully!")
print("💡 To run: save as .py file and execute 'python pipeline.py run'")

