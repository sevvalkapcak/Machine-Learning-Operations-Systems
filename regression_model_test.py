import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
# Veri setini yükle
df = pd.read_csv("/content/Expanded_data_with_more_features.csv")
df=pd.DataFrame(df)

# Bağımsız değişkenler
X = df[["Gender", "WklyStudyHours", "TestPrep"]]

# Bağımlı değişkenler
y = df[["MathScore", "ReadingScore", "WritingScore"]]

# Kategorik değişkenleri kodlama
X_encoded = pd.get_dummies(X, drop_first=True)

# Veri setini eğitim ve test veri setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Polinomal özellikler oluşturma
poly = PolynomialFeatures(degree=2,include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Polinomal regresyon modelini tanımlama
model = LinearRegression()

# Hyperparameter grid'i tanımlama
param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False],
    'copy_X': [True, False]
}

# Grid Search kullanarak hyperparameter tuning yapma
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_poly, y_train)

# En iyi parametreleri ve skoru yazdırma
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor (negatif ortalama kare hatası):", grid_search.best_score_)

# En iyi modeli seçme
best_model = grid_search.best_estimator_

# Test veri seti üzerinde tahmin yapma
y_pred = best_model.predict(X_test_poly)

# Performans metriklerini hesaplama
mse = mean_squared_error(y_test, y_pred)

print("Ortalama Kare Hatası (MSE):", mse)
# Testi geçme veya başarısız olma durumunu kontrol etme
if abs(mse) <= 300:
    print("Test başarılı: Ortalama Kare Hatası (MSE) hedef değere yakın.")
else:
    print("Test başarısız")