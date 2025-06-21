# Lojistik Regresyon Modellerinin İncelenmesi: Gradient Descent ve IRLS (YZ Yazımı)

Bu proje, "Framingham Kalp Hastalığı" veri setini kullanarak lojistik regresyon modelinin iki farklı optimizasyon algoritması ile sıfırdan (from-scratch) implementasyonunu içermektedir: **Gradient Descent** ve **Iterative Reweighted Least Squares (IRLS)**. Projenin amacı, bu iki algoritmanın teorik altyapısını açıklamak, Python ile implementasyonunu göstermek ve modellerin performansını karşılaştırmaktır.

Notebook çıktılarında da görüldüğü üzere, Gradient Descent algoritması, özellikle veri setindeki dengesizlik ve özelliklerin ölçeklendirilmemesi gibi nedenlerle başarısız bir sonuç verirken, Newton metodu tabanlı IRLS algoritması çok daha hızlı yakınsayarak anlamlı ve başarılı bir model ortaya koymuştur.

## İçindekiler
1.  [Projenin Amacı](#projenin-amacı)
2.  [Teorik Altyapı](#teorik-altyapı)
    -   [Lojistik Regresyon](#lojistik-regresyon)
    -   [Gradient Descent (Gradyan İniş)](#gradient-descent-gradyan-iniş)
    -   [Iterative Reweighted Least Squares (IRLS)](#iterative-reweighted-least-squares-irls)
3.  [Veri Seti](#veri-seti)
4.  [Uygulama Adımları](#uygulama-adımları)
    -   [Veri Ön İşleme](#veri-ön-i̇şleme)
    -   [Dengesiz Veri Seti Problemi ve Çözümü: Undersampling](#dengesiz-veri-seti-problemi-ve-çözümü-undersampling)
    -   [Model Implementasyonu ve Eğitim](#model-i̇mplementasyonu-ve-eğitim)
5.  [Sonuçlar ve Değerlendirme](#sonuçlar-ve-değerlendirme)
    -   [Karşılaşılan Sorunlar](#karşılaşılan-sorunlar)
    -   [Model Performanslarının Karşılaştırılması](#model-performanslarının-karşılaştırılması)
6.  [Proje Nasıl Çalıştırılır?](#proje-nasıl-çalıştırılır)

---

## 1. Projenin Amacı
Bu projenin temel amacı, ikili sınıflandırma (binary classification) problemlerinin çözümünde sıkça kullanılan Lojistik Regresyon modelini derinlemesine anlamak ve bu modelin parametrelerini optimize etmek için kullanılan iki temel algoritmayı karşılaştırmaktır. Özellikle, sıfırdan kodlama yapılarak algoritmaların matematiksel temellerinin pratiğe nasıl döküldüğü gösterilmektedir.

## 2. Teorik Altyapı

### Lojistik Regresyon
Lojistik regresyon, bir sonucun olasılığını tahmin etmek için kullanılan bir sınıflandırma algoritmasıdır. Bu olasılık değeri, **Sigmoid (Lojistik) Fonksiyon** aracılığıyla 0 ile 1 arasında bir değere dönüştürülür.

**Sigmoid Fonksiyonu:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
Burada $z = \theta^T X$, yani model parametreleri ($\theta$) ile girdi özelliklerinin ($X$) doğrusal birleşimidir.

**Maliyet Fonksiyonu (Binary Cross-Entropy):**
Modelin ne kadar iyi performans gösterdiğini ölçmek için maliyet fonksiyonu kullanılır. Lojistik regresyonda bu genellikle Negatif Log-Likelihood veya Binary Cross-Entropy Loss olarak adlandırılır.
$$J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} [y_i \log(\pi_i) + (1 - y_i) \log(1 - \pi_i)]$$
Burada $\pi_i = \sigma(\theta^T x_i)$ modelin tahmin olasılığı, $y_i$ ise gerçek etikettir. Amacımız bu $J(\theta)$ fonksiyonunu minimize etmektir.

### Gradient Descent (Gradyan İniş)
Gradient Descent, maliyet fonksiyonunu minimize etmek için kullanılan birinci dereceden bir optimizasyon algoritmasıdır. Maliyet fonksiyonunun gradyanını (türevini) hesaplayarak parametreleri "eğimin en dik olduğu yönün tersine" doğru iteratif olarak günceller.

**Gradyan:**
$$\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} X^T (\pi - y)$$

**Parametre Güncelleme Kuralı:**
$$\theta_{yeni} = \theta_{eski} - \alpha \frac{\partial J(\theta)}{\partial \theta}$$
Burada $\alpha$ öğrenme oranıdır (learning rate).

### Iterative Reweighted Least Squares (IRLS)
IRLS, Lojistik Regresyon gibi Genelleştirilmiş Doğrusal Modellerin (GLM) parametrelerini bulmak için kullanılan ve **Newton Metodu**'na dayanan güçlü bir algoritmadır. Sadece gradyanı (birinci türev) değil, aynı zamanda **Hessian matrisini** (ikinci türev) de kullanarak daha hızlı ve kararlı bir yakınsama sağlar.

**Newton Metodu Güncelleme Kuralı:**
$$\theta_{k+1} = \theta_k - H_k^{-1} g_k$$
Burada:
-   $g_k = X^T(\pi_k - y)$ gradyandır.
-   $H_k = X^T S_k X$ Hessian matrisidir.
-   $S_k$ ise her bir elemanı $\pi_i(1 - \pi_i)$ olan bir köşegen ağırlık matrisidir.

Her adımda $S_k$ matrisi yeniden hesaplandığı için yönteme "Yinelemeli Yeniden Ağırlıklandırılmış" En Küçük Kareler denir.

## 3. Veri Seti
Projede, kalp hastalığı riskini tahmin etmeyi amaçlayan **Framingham Kalp Hastalığı** veri seti kullanılmıştır. Veri seti, hastaların demografik ve klinik bilgilerini içermektedir. Hedef değişken `TenYearCHD` (10 yıl içinde koroner kalp hastalığı riski) olup, ikili (0 veya 1) bir değişkendir.

## 4. Uygulama Adımları

### Veri Ön İşleme
1.  **Eksik Verilerin Doldurulması:** Veri setindeki eksik değerler, `education`, `cigsPerDay`, `BPMeds` için mod; `totChol`, `BMI` için ise ortalama değerleri ile doldurulmuştur. `heartRate` ve `glucose` sütunlarındaki eksik veriye sahip satırlar ise doğrudan silinmiştir.
2.  **Veri Ayırma:** Veri seti, %80 eğitim ve %20 test seti olacak şekilde `train_test_split` ile ayrılmıştır. Sınıf dağılımının korunması için `stratify=y` parametresi kullanılmıştır.

### Dengesiz Veri Seti Problemi ve Çözümü: Undersampling
`TenYearCHD` hedef değişkeni oldukça dengesizdir (çoğunlukla 0 değeri). Bu durum, modelin çoğunluk sınıfını öğrenmeye meyilli olmasına ve azınlık sınıfını (1) tahmin etmede başarısız olmasına yol açar. Bu sorunu çözmek için **Undersampling (Eksik Örnekleme)** tekniği uygulanmıştır. Bu teknikte, çoğunluk sınıfından (0), azınlık sınıfının (1) örnek sayısına eşit sayıda rastgele örnek seçilerek dengeli bir eğitim seti oluşturulmuştur.

### Model Implementasyonu ve Eğitim
Her iki algoritma da (Gradient Descent ve IRLS) Python ve NumPy kullanılarak sıfırdan kodlanmıştır.
-   `sigmoid` ve `predict_probabilities` gibi yardımcı fonksiyonlar tanımlanmıştır.
-   `calculate_gradient` fonksiyonu gradyan formülünü uygular.
-   `IRLS_logistic_regression_with_loss` fonksiyonu ise Hessian matrisini de kullanarak Newton metodunu uygular.
-   Her iki model de dengelenmiş eğitim seti üzerinde eğitilmiş ve maliyetin (loss) her iterasyondaki değişimi görselleştirilmiştir.

## 5. Sonuçlar ve Değerlendirme

### Karşılaşılan Sorunlar
1.  **Gradient Descent Modelinin Başarısızlığı:** Notebook'taki test sonuçları, Gradient Descent ile eğitilen modelin son derece kötü performans gösterdiğini ortaya koymuştur (`accuracy: 0.15`). Model, tüm test örneklerini çoğunluk sınıfı olan "0" olarak tahmin etmeye meyilliydi. Herhangi bir çözüm bulamadım. Araştırdığımda bunun temel nedenlerinin:
    * **Öğrenme Oranının (Learning Rate) Hassasiyeti:** Gradient Descent, öğrenme oranına çok duyarlıdır. Optimal bir oran bulunamazsa, model ya çok yavaş yakınsar ya da "sıçramalar" yaparak minimum noktayı bulamaz.
    * **Özelliklerin Ölçeklendirilmemesi:** Veri setindeki farklı ölçeklerdeki(int,object) özellikler, gradyanın bazı yönlerde çok büyük, bazı yönlerde ise çok küçük olmasına neden olarak optimizasyonu zorlaştırır.
Olduğunu gördüm. 
2.  **Sayısal Kararlılık Sorunları:** Sigmoid fonksiyonunun hesaplanması sırasında, `np.exp(-z)` ifadesinde `z`'nin çok büyük negatif değerler alması durumunda "overflow" hatası alınmıştır. Bu sorun, `scipy.special.expit` gibi sayısal olarak daha kararlı bir sigmoid fonksiyonu kullanılarak çözülmüştür.

### Model Performanslarının Karşılaştırılması

| Metrik | Gradient Descent | IRLS (Newton Metodu) |
| :--- | :---: | :---: |
| **Accuracy** | 0.15 | **0.66** |
| **Precision (Sınıf 1)** | 0.15 | **0.25** |
| **Recall (Sınıf 1)** | 1.00 | **0.63** |
| **F1-Score (Sınıf 1)** | 0.27 | **0.36** |
| **Yakınsama Hızı** | Yavaş (100 iterasyon) | **Çok Hızlı** (5 iterasyon) |

**Değerlendirme:**
-   **Yakınsama:** IRLS, sadece 5 iterasyonda yakınsarken, Gradient Descent 100 iterasyonda bile optimal bir sonuca ulaşamamıştır. IRLS'in ikinci dereceden bilgi (Hessian) kullanması, onu çok daha verimli kılmaktadır.
-   **Doğruluk:** IRLS ile elde edilen %66'lık doğruluk oranı, Gradient Descent'in %15'lik oranına kıyasla çok daha üstündür.
-   **Sınıflandırma Raporu:** IRLS modelinin sınıflandırma raporu, her iki sınıf için de anlamlı `precision` ve `recall` değerleri ürettiğini göstermektedir. Gradient Descent modeli ise sınıf 1 için yüksek bir `recall` değerine sahip olsa da, bu durum modelin neredeyse her şeyi sınıf 1 olarak tahmin etme eğiliminden kaynaklanmaktadır (düşük `precision`).

Sonuç olarak, lojistik regresyon gibi modeller için IRLS (Newton Metodu), Gradient Descent'e göre çok daha hızlı, kararlı ve başarılı bir optimizasyon yöntemidir.
