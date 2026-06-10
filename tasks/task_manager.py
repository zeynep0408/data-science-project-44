"""
DS-44 — Türkçe Metin Sınıflandırma (TF-IDF + Türkçe Ön İşleme)

Bir e-ticaret platformunda data scientist'sin. Ürün yorumlarını otomatik olarak
POZİTİF (1) ve NEGATİF (0) olarak ayıran bir duygu analizi modeli kuruyorsun.
Türkçe metinle çalışmak özel dikkat ister:
  - Türkçe'ye duyarlı küçük harf ("İSTANBUL" → "istanbul", "IRMAK" → "ırmak")
  - Noktalama / sayı temizliği (Türkçe harfler çğıöşü korunur)
  - Türkçe duraklama kelimeleri (stopword) temizliği
Sınıflandırma için TF-IDF (karakter n-gram) + LogisticRegression kullanacaksın.
Ayrıca Zeyrek ile kelime köklerini bulmayı (lemmatization) göreceksin — ama
Zeyrek yavaş olduğu için ana pipeline'da KULLANILMAZ, ayrı bir fonksiyondur.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""

import re
import logging

import numpy as np
import pandas as pd

# Zeyrek log gürültüsünü bastır (analiz sırasında çok sayıda DEBUG/INFO basar)
logging.getLogger('zeyrek').setLevel(logging.ERROR)
logging.getLogger('zeyrek.morphology').setLevel(logging.ERROR)
logging.getLogger('zeyrek.rulebasedanalyzer').setLevel(logging.ERROR)


# Türkçe duraklama (stopword) kelimeleri — yaygın, ayırt edici olmayan kelimeler.
# Bu listeyi preprocess() içinde kullanacaksın. İstersen genişletebilirsin.
TURKISH_STOPWORDS = [
    've', 'ile', 'de', 'da', 'ki', 'bir', 'bu', 'şu', 'o', 'için', 'çok',
    'ama', 'fakat', 'ancak', 'gibi', 'kadar', 'daha', 'en', 'her', 'hem',
    'ya', 'veya', 'ne', 'mi', 'mu', 'mı', 'mü', 'ise', 'çünkü', 'yani',
    'göre', 'sonra', 'önce', 'şey', 'bende', 'sende', 'onlar', 'biz', 'siz',
    'ben', 'sen', 'olarak', 'olan', 'oldu', 'diye', 'hiç', 'tüm', 'birçok',
]


# Zeyrek analizörü — global, tek sefer kur (kurulum maliyetli)
_ANALYZER = None


def _get_analyzer():
    """Zeyrek MorphAnalyzer'ı tembel (lazy) oluştur, tekrar kullan."""
    global _ANALYZER
    if _ANALYZER is None:
        import zeyrek
        _ANALYZER = zeyrek.MorphAnalyzer()
    return _ANALYZER


# 1. Veriyi yükle — koda gömülü etiketli Türkçe yorumlar (HAZIR VERİLDİ)
def load_data():
    """
    Etiketli Türkçe ürün yorumlarını DataFrame olarak döndürür.
    1 = pozitif, 0 = negatif.

    Bu fonksiyon SANA VERİLDİ — veri koda gömülüdür. Aşağıdaki listeleri
    'text' ve 'label' sütunlu bir DataFrame'e çevirip döndürmen yeterli.

    Returns:
        pd.DataFrame: sütunlar ['text', 'label'] (~30 satır, dengeli)
    """
    pozitif = [
        "Bu ürün harika, çok memnun kaldım kesinlikle tavsiye ederim",
        "Kargo çok hızlı geldi ve paketleme mükemmeldi teşekkürler",
        "Muhteşem bir deneyimdi, herkese gönül rahatlığıyla öneririm",
        "Fiyatına göre performansı şahane, beklentimin çok üstünde",
        "Ürün kaliteli ve sağlam, uzun süredir kullanıyorum hiç sorun yok",
        "Çok beğendim, rengi ve dokusu tam istediğim gibi mükemmel",
        "Satıcı ilgili ve nazikti, sorularıma hızlıca cevap verdi",
        "Bayıldım bu ürüne, arkadaşlarıma da aldırdım çok güzel",
        "Gerçekten kaliteli bir ürün, parasının hakkını fazlasıyla veriyor",
        "Süper bir alışveriş oldu, kesinlikle tekrar sipariş vereceğim",
        "Harika çalışıyor, hiçbir problem yaşamadım çok mutluyum",
        "Beklentimin üzerinde çıktı, fevkalade bir kalite gerçekten",
        "Çok şık ve kullanışlı, herkese tavsiye ederim bayıldım",
        "Mükemmel hizmet aldım, ürün eksiksiz ve sapasağlam geldi",
        "Tam aradığım ürün, son derece memnunum teşekkür ederim",
    ]
    negatif = [
        "Berbat bir ürün, kesinlikle tavsiye etmiyorum paramı çöpe attım",
        "Kargo çok geç geldi ve paket ezilmişti rezalet bir hizmet",
        "Hiç beğenmedim, kalitesiz ve dayanıksız bir an önce iade edeceğim",
        "Fiyatına göre çok kötü, beklentimin çok altında berbat",
        "Ürün bozuk geldi, çalışmıyor bile büyük hayal kırıklığı",
        "Rezalet, ilk kullanımda kırıldı asla almayın boşa para",
        "Satıcı ilgisizdi, sorularıma günlerce cevap vermedi çok kötü",
        "Korkunç bir deneyim, bir daha asla sipariş vermem berbat",
        "Kalitesiz malzeme, kısa sürede yıprandı hiç memnun değilim",
        "Çok pişman oldum bu alışverişten, tam bir fiyasko olmuş",
        "Ürün resimdekiyle alakasız, kandırıldım gibi hissediyorum kötü",
        "Dayanıksız ve özensiz, ilk haftada bozuldu berbat kalite",
        "İğrenç bir koku geldi üründen, kullanılamaz halde rezalet",
        "Param boşa gitti, hiçbir işe yaramıyor çok sinir bozucu",
        "Hayal kırıklığı yarattı, kötü işçilik kesinlikle almayın",
    ]
    texts = pozitif + negatif
    labels = [1] * len(pozitif) + [0] * len(negatif)
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df


# 2. Türkçe'ye duyarlı küçük harfe çevirme
def turkish_lower(text):
    """
    Türkçe'ye duyarlı lowercase uygula.

    Python'un str.lower() metodu 'I' → 'i' yapar; ama Türkçe'de:
      'I' → 'ı'   ve   'İ' → 'i'
    olmalıdır. Bu yüzden önce bu iki harfi ELLE dönüştür, sonra .lower() çağır.

    Örn: "İSTANBUL" → "istanbul", "IRMAK" → "ırmak", "İyi" → "iyi"

    Args:
        text: str

    Returns:
        str: Türkçe'ye duyarlı küçük harfli metin

    İpucu:
    - text = text.replace('İ', 'i').replace('I', 'ı')
    - return text.lower()
    """
    pass


# 3. Metni temizle — noktalama ve sayıları at
def clean_text(text):
    """
    Metni temizle:
    1. turkish_lower ile küçük harfe çevir
    2. Türkçe harfler (a-z + çğıöşü) ve boşluk DIŞINDAKİ her şeyi (noktalama,
       rakam) boşlukla değiştir
    3. Fazla boşlukları teke indir, baştaki/sondaki boşlukları kırp

    Örn: "Harika, çok güzel!!! 123" → "harika çok güzel"

    Args:
        text: str

    Returns:
        str: temizlenmiş metin

    İpucu:
    - import re
    - text = turkish_lower(text)
    - text = re.sub(r'[^a-zçğıöşü\\s]', ' ', text)
    - text = re.sub(r'\\s+', ' ', text).strip()
    """
    pass


# 4. Ön işleme — temizle + stopword çıkar
def preprocess(text):
    """
    Tam ön işleme akışı:
    1. clean_text(text) (lowercase + noktalama temizliği)
    2. .split() ile kelimelere ayır
    3. TURKISH_STOPWORDS içindeki kelimeleri çıkar
    4. Kalan kelimeleri tek boşlukla birleştirip string döndür

    Örn: "bu ürün ve kargo çok güzel" → "ürün kargo güzel"
    ('bu', 've', 'çok' stopword olduğu için atılır)

    Args:
        text: str

    Returns:
        str: temizlenmiş + stopword'süz metin

    İpucu:
    - cleaned = clean_text(text)
    - stop = set(TURKISH_STOPWORDS)  # set → hızlı arama
    - kept = [w for w in cleaned.split() if w not in stop]
    - return ' '.join(kept)
    """
    pass


# 5. Zeyrek ile lemmatization (kök bulma) — opsiyonel/ağır
def lemmatize_words(words):
    """
    Zeyrek morfolojik analizörü ile her kelimenin kökünü (lemma) bul.
    Örn: ["kitaplar", "geldim"] → ["kitap", "gelmek"].

    NOT: Zeyrek YAVAŞTIR — bu yüzden ana pipeline'da (build_pipeline)
    KULLANILMAZ. Bu fonksiyon sadece Türkçe kök bulmayı göstermek/öğrenmek
    içindir, ayrı test edilir.

    Args:
        words: list of str (kelimeler)

    Returns:
        list of str: her kelimenin kökü (lemma), küçük harfli.
                     Kök bulunamazsa kelimenin kendisi döner.

    İpucu:
    - analyzer = _get_analyzer()  # yukarıda hazır verilen yardımcı
    - result = analyzer.lemmatize(w)
      → result formatı: [(orijinal_kelime, [lemma1, lemma2, ...])]
      → ilk lemma: result[0][1][0]
    - Zeyrek bazen kökü büyük harfle döndürür → turkish_lower ile küçült
    - Kök bulunamazsa (result boşsa) kelimenin kendisini ekle
    """
    pass


# 6. TF-IDF + LogisticRegression pipeline kur
def build_pipeline():
    """
    sklearn Pipeline:
    - 'tfidf': TfidfVectorizer
        → preprocessor=preprocess (yukarıda yazdığın Türkçe ön işleme:
          lowercase + noktalama + stopword temizliği)
        → analyzer='char_wb', ngram_range=(2, 4)
          (kelime sınırı içinde KARAKTER n-gramları — Türkçe'nin zengin
           ekli/çekimli yapısında kelime köklerini ve eklerini yakalar,
           küçük veri setinde kelime n-gramlarından daha sağlam çalışır)
    - 'clf': LogisticRegression(max_iter=1000, C=10)

    Returns:
        sklearn.pipeline.Pipeline

    İpucu:
    - from sklearn.feature_extraction.text import TfidfVectorizer
    - from sklearn.linear_model import LogisticRegression
    - from sklearn.pipeline import Pipeline
    - TfidfVectorizer(preprocessor=preprocess, analyzer='char_wb',
                      ngram_range=(2, 4))
    """
    pass


# 7. Modeli eğit
def train_model(pipe, X, y):
    """
    Pipeline'ı fit et ve döndür.

    Args:
        pipe: build_pipeline'dan dönen pipeline
        X: ham metinler (pd.Series — vektörize EDİLMEMİŞ)
        y: etiketler (0/1)

    Returns:
        Pipeline: fit edilmiş pipeline

    İpucu: pipe.fit(X, y); return pipe
    """
    pass


# 8. Modeli değerlendir → accuracy
def evaluate(pipe, X, y):
    """
    Verilen X, y üzerinde tahmin yap, accuracy (doğruluk) döndür.

    Args:
        pipe: eğitilmiş pipeline
        X: metinler
        y: gerçek etiketler

    Returns:
        float: accuracy

    İpucu:
    - from sklearn.metrics import accuracy_score
    - y_pred = pipe.predict(X)
    - return float(accuracy_score(y, y_pred))
    """
    pass


# 9. Tek bir cümle için duygu tahmini
def predict_sentiment(pipe, text):
    """
    Bir cümle al, "pozitif" veya "negatif" döndür.

    Args:
        pipe: eğitilmiş pipeline
        text: str

    Returns:
        str: "pozitif" (tahmin=1) veya "negatif" (tahmin=0)

    İpucu:
    - Pipeline tek string değil liste bekler → [text] geç
    - pred = int(pipe.predict([text])[0])
    - return "pozitif" if pred == 1 else "negatif"
    """
    pass


# 10. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    """
    Uçtan uca akış:
    1. load_data
    2. train/test split:
       train_test_split(df['text'], df['label'], test_size=0.3,
                        stratify=df['label'], random_state=42)
    3. build_pipeline → train_model
    4. evaluate (test accuracy)
    5. İki örnek tahmin:
       - Pozitif: "Bu ürün gerçekten harika, çok memnun kaldım tavsiye ederim"
       - Negatif: "Berbat bir ürün, hiç beğenmedim paramı çöpe attım"

    Returns:
        dict: {
            'test_accuracy': float,
            'sample_positive': str ("pozitif" beklenir),
            'sample_negative': str ("negatif" beklenir),
        }

    İpucu: from sklearn.model_selection import train_test_split
    """
    pass


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Test accuracy     : {result['test_accuracy']:.4f}")
    print(f"  Pozitif örnek     : {result['sample_positive']}")
    print(f"  Negatif örnek     : {result['sample_negative']}")
