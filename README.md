# DS-44 — Türkçe Metin Sınıflandırma (TF-IDF + Türkçe Ön İşleme)

**Modül**: NLP / Metin Sınıflandırma • **Süre**: 2-3 saat

## 🎯 Proje Senaryosu

Bir e-ticaret platformunda **data scientist** olarak çalışıyorsun. Platforma her gün binlerce **ürün yorumu** düşüyor. Müşteri deneyimi ekibi, bu yorumları tek tek okuyacak zamana sahip değil — ama memnuniyetsiz müşterileri hızlıca yakalamak istiyorlar. Senden, bir yorumu okuyup **pozitif mi negatif mi** olduğunu otomatik tahmin eden bir **duygu analizi (sentiment analysis)** modeli kurmanı istiyorlar.

Doğru çalışırsa negatif yorumlar anında destek ekibine düşer, pozitif yorumlar pazarlamada öne çıkarılır.

Bu projede iş **Türkçe metin** üzerine — ve Türkçe, İngilizce'den farklı zorluklar barındırır:

- **Türkçe'ye duyarlı küçük harf**: Python'un standart `lower()` metodu Türkçe için yanlış çalışır. `"I".lower()` → `"i"` verir ama Türkçe'de `"I" → "ı"` ve `"İ" → "i"` olmalıdır. (`"İSTANBUL"` → `"istanbul"`)
- **Noktalama / sayı temizliği**: Türkçe harfler (`çğıöşü`) korunmalı, geri kalanı (noktalama, rakam) atılmalı.
- **Türkçe duraklama kelimeleri (stopword)**: `ve`, `bir`, `bu`, `çok` gibi sık ama ayırt edici olmayan kelimeler temizlenir.

Sınıflandırma için **TF-IDF + LogisticRegression** kullanacaksın. TF-IDF'i **karakter n-gram** (`char_wb`) modunda çalıştıracağız: Türkçe'nin zengin ekli/çekimli (sondan eklemeli) yapısında karakter n-gramları kelime köklerini ve eklerini yakalar ve küçük veri setlerinde kelime n-gramlarından daha sağlam sonuç verir.

Ayrıca **Zeyrek** ile Türkçe **lemmatization (kök bulma)** göreceksin (`"kitaplar" → "kitap"`, `"geldim" → "gelmek"`). Zeyrek güçlü ama **yavaş** bir araçtır — bu yüzden onu ayrı bir fonksiyonda gösteriyoruz, ana pipeline'da kullanmıyoruz.

Bu projede şunları uygulayacaksın:
- ✅ **Türkçe'ye özgü metin ön işleme** (lowercase / noktalama / stopword)
- ✅ **TF-IDF** ile metni sayısal vektöre çevirme (karakter n-gram)
- ✅ **LogisticRegression** ile ikili sınıflandırma
- ✅ **Zeyrek** ile Türkçe kök bulma (lemmatization)
- ✅ **sklearn Pipeline** (ön işleme + vektörize + model tek akışta)
- ✅ **Stratified train/test split** ve **accuracy** ile değerlendirme

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd ds-44-turkce-siniflandirma

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

> **Not — Zeyrek ilk çağrı:** `lemmatize_words` ilk çağrıldığında Zeyrek morfolojik sözlüğünü yükler (birkaç saniye). Ana pipeline Zeyrek kullanmadığı için hızlıdır.

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 724 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📚 Veri Seti — Türkçe Ürün Yorumları (koda gömülü)

Bu projede veri **koda gömülüdür** — internet gerekmez. `load_data()` fonksiyonu, elle etiketlenmiş **~30 kısa Türkçe ürün yorumu** döndürür:

- **15 pozitif** (`label = 1`): "Bu ürün harika, çok memnun kaldım…", "Kargo çok hızlı geldi…"
- **15 negatif** (`label = 0`): "Berbat bir ürün, paramı çöpe attım…", "Hiç beğenmedim, kalitesiz…"

### Boyut & Target
- **~30 yorum × 2 sütun** (`text`, `label`)
- Target: `label` (`1` = pozitif, `0` = negatif)
- **Dengeli (balanced)** — %50 pozitif, %50 negatif. Accuracy anlamlı bir metriktir.

### Format
`load_data()` bir `pd.DataFrame` döndürür: sütunlar `['text', 'label']`.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki fonksiyonları sırayla doldur. `load_data` ve `TURKISH_STOPWORDS` **sana verildi** — kalan fonksiyonların `pass` kısmını doldur.

1. **`load_data()`** — (HAZIR) etiketli Türkçe yorumları DataFrame döndürür
2. **`turkish_lower(text)`** — Türkçe'ye duyarlı küçük harf (`İ→i`, `I→ı`)
3. **`clean_text(text)`** — noktalama + rakam temizliği (Türkçe harfler kalır)
4. **`preprocess(text)`** — clean_text + Türkçe stopword temizliği
5. **`lemmatize_words(words)`** — Zeyrek ile kök bulma (ayrı, ana pipeline'da kullanılmaz)
6. **`build_pipeline()`** — TfidfVectorizer(char_wb, 2-4) + LogisticRegression
7. **`train_model(pipe, X, y)`** — fit + dön
8. **`evaluate(pipe, X, y)`** — accuracy döndür
9. **`predict_sentiment(pipe, text)`** — "pozitif" / "negatif"
10. **`run_pipeline()`** — uçtan uca akış, özet dict dön

> `TURKISH_STOPWORDS` (liste) ve `_get_analyzer()` (Zeyrek yardımcısı) dosyada hazır verilmiştir.

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] **Türkçe'ye özgü** metin ön işleme yapabileceksin (lowercase / noktalama / stopword)
- [x] **TF-IDF** ile metni sayıya çevirebileceksin (karakter n-gram)
- [x] **LogisticRegression** ile ikili sınıflandırma yapabileceksin
- [x] **Zeyrek** ile Türkçe kök bulmayı (lemmatization) gösterebileceksin
- [x] **sklearn Pipeline** ile ön işleme + model adımlarını birleştirebileceksin
- [x] **Stratified split** + **accuracy** ile değerlendirme yapabileceksin

## 🧪 Testler

Test dosyası: `tests/test_question.py` (12 test)

Tümü pass olmalı:
- `load_data` yapısı (~30 satır, `['text','label']`, dengeli, sınıflar {0,1})
- `turkish_lower` doğru ("İSTANBUL"→"istanbul", "IRMAK"→"ırmak")
- `clean_text` noktalama + rakam siler, Türkçe harf korur
- `TURKISH_STOPWORDS` bir liste ve yaygın kelimeleri içeriyor
- `preprocess` stopword'leri çıkarır
- `lemmatize_words` Zeyrek köklerini bulur ("kitaplar"→"kitap")
- `build_pipeline` tipi doğru (TfidfVectorizer + LogisticRegression)
- Model accuracy eşik üstü (test setinde **>= 0.8**)
- `predict_sentiment` net pozitif/negatif cümleleri doğru sınıflar
- `run_pipeline` uçtan uca çalışır

## 📊 Beklenen Sonuçlar

```
Sınıf dağılımı : %50 pozitif / %50 negatif (dengeli)
Test accuracy  : ~0.89 (TF-IDF char n-gram + LogisticRegression)
Örnek tahmin   : pozitif yorum → "pozitif", negatif yorum → "negatif"
Lemmatization  : "kitaplar" → "kitap", "geldim" → "gelmek" (Zeyrek)
```

## 💡 İpuçları

- **Türkçe lowercase**: önce `text.replace('İ','i').replace('I','ı')`, sonra `.lower()`. Sıra önemli.
- **clean_text** regex: `re.sub(r'[^a-zçğıöşü\s]', ' ', text)` → Türkçe harf + boşluk dışını siler.
- **Pipeline ham string alır** — sen elle vektörize etme, `preprocessor=preprocess` işi TF-IDF içinde yapar.
- **char_wb** (kelime sınırı içi karakter n-gram) Türkçe için kelime n-gramından daha iyi: küçük sette ek/kök kalıplarını yakalar.
- **Zeyrek yavaş** — sadece `lemmatize_words` içinde, döngüde `analyzer.lemmatize(w)` çağır. Sonuç `[(kelime, [lemma...])]` formatında; ilk lemma `result[0][1][0]`.
- Zeyrek log gürültüsü dosyanın başında `logging...setLevel(ERROR)` ile zaten bastırıldı.
- `predict_sentiment`'ta tek string değil **liste** geç: `pipe.predict([text])`.

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42` ve `test_size=0.3` değerlerini değiştirme (testler fail olur)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
