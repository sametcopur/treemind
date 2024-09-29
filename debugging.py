import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from tree_explainer.explainer import Tree
from tree_explainer.plots import plot_bar, plot_values_points, plot_dependecy, plot_points

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Ana özellikler
    feature_1 = np.random.normal(loc=0, scale=1, size=n_samples)
    feature_2 = np.random.normal(loc=0, scale=1, size=n_samples)
    feature_3 = np.random.uniform(low=-2, high=2, size=n_samples)
    feature_4 = np.random.normal(loc=0, scale=1, size=n_samples)
    feature_5 = np.random.normal(loc=0, scale=1, size=n_samples)
    
    # Hedef değişkeni oluştur (çeşitli gizli etkileşimlerle ve koşullu durumlarla)
    target = (
        0.5 * np.where(feature_1 > 0, feature_2, -feature_2) +  # Koşullu etkileşim: feature_1 > 0 olunca feature_2 pozitif etkiliyor, aksi halde negatif
        0.3 * np.maximum(feature_2, feature_3) +  # En büyük olan feature'ı seçip pozitif etki veriyoruz
        #-0.5 * np.where(feature_3 > 0, (feature_4)**2, feature_4) +  # feature_3 > 1 olunca feature_1'in karesi negatif etki yapıyor, aksi halde mutlak değer alınıyor
        -0.5 * (feature_4 * feature_5) + 
        np.random.normal(loc=0, scale=0.1, size=n_samples)  # Gürültü ekleyelim
    )
    
    # Hedef değişkeni ikili sınıflandırma için dönüştür
    target_binary = (target > target.mean()).astype(int)
    
    # DataFrame oluştur
    df = pd.DataFrame({
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3,
        'feature_4': feature_4,
        'feature_5': feature_5,
        'target': target_binary
    })
    
    return df

# Veri setini oluştur
df = generate_synthetic_data(n_samples=1000)

# Eğitim ve test setlerine ayır
X = df.drop('target', axis=1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters for LightGBM
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": 4,
    "feature_fraction": 0.9,
}

# Train the model
num_round = 100
model = lgb.train(params, train_data, num_round, valid_sets=[test_data])

tree = Tree(model, X_train)
tree.analyze_tree()

df = tree.analyze_feature_v2(2)

df = tree.analyze_dependency(0, 1)