# 💰 Salary Prediction Model

Bu proje, profesyonel beyzbol oyuncularının sezonluk ve kariyer istatistiklerini kullanarak oyuncuların maaşlarını tahmin eden bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. Modelin oluşturulmasından önce detaylı bir veri analizi ve özellik mühendisliği süreci gerçekleştirilmiştir.

---

## 📁 Veri Seti Hakkında

Veri seti, Carnegie Mellon Üniversitesi'nin StatLib kütüphanesinden alınmıştır. 1986-1987 beyzbol sezonuna ait gerçek verilere dayanmaktadır ve Sports Illustrated (20 Nisan 1987) ve 1987 Beyzbol Ansiklopedisi Güncellemesi kaynak alınarak derlenmiştir.

- **Toplam Gözlem Sayısı**: 263  
- **Bağımsız Değişkenler**: 16  
- **Hedef Değişken**: `Salary` (Oyuncunun maaşı)

---

## 🔍 Değişkenler

| Değişken      | Açıklama                                                                 |
|---------------|--------------------------------------------------------------------------|
| `AtBat`       | 1986 sezonunda yapılan vuruş sayısı                                      |
| `Hits`        | 1986 sezonundaki isabetli vuruş sayısı                                   |
| `HmRun`       | 1986 sezonundaki home run (en değerli vuruş) sayısı                      |
| `Runs`        | 1986 sezonunda oyuncunun kazandırdığı skor sayısı                        |
| `RBI`         | Oyuncunun vuruşlarıyla koşu yaptırdığı oyuncu sayısı                     |
| `Walks`       | Rakip tarafından oyuncuya verilen serbest geçiş sayısı                   |
| `Years`       | Oyuncunun major ligde oynama süresi (yıl)                                |
| `CAtBat`      | Kariyeri boyunca yapılan toplam vuruş sayısı                             |
| `CHits`       | Kariyer boyunca yapılan isabetli vuruş sayısı                            |
| `CHmRun`      | Kariyer boyunca yapılan home run sayısı                                  |
| `CRuns`       | Kariyer boyunca kazandırılan toplam skor                                 |
| `CRBI`        | Kariyer boyunca koşu yaptırılan oyuncu sayısı                            |
| `CWalks`      | Kariyer boyunca kazanılan serbest geçiş sayısı                           |
| `League`      | 1986 sezonundaki ligin tipi (`A` veya `N`)                               |
| `Division`    | Oyuncunun bulunduğu bölge (`E`: East, `W`: West)                         |
| `PutOuts`     | 1986 sezonundaki savunma pozisyonunda topu dışarı çıkardığı sayı         |
| `Assists`     | 1986 sezonundaki savunma yardımı sayısı                                  |
| `Errors`      | 1986 sezonundaki hata sayısı                                             |
| `Salary`      | Oyuncunun maaşı (hedef değişken)                                         |

---

## 🧠 Kullanılan Yöntemler

- Veri temizleme ve eksik değer analizi  
- Kategorik değişkenlerin etiketlenmesi ve One-Hot Encoding  
- Özellik mühendisliği ve korelasyon analizi  
- Regresyon modelleri:
  - **Linear Regression**
  - **Ridge / Lasso**
  - **Random Forest Regressor**
  - **XGBoost Regressor**
- Model performans ölçütleri:
  - `R² Score`
  - `RMSE (Root Mean Squared Error)`
  - `MAE (Mean Absolute Error)`

---

## 📈 Proje Hedefi

Oyuncuların maaşlarını tahmin ederek:
- Performans-maliyet ilişkisini incelemek  
- Oyuncu verileriyle karar destek sistemleri oluşturmak  
- Gelecek maaş müzakerelerinde veri temelli tahminlerde bulunmak


