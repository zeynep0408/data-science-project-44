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

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
    global _ANALYZER
    if _ANALYZER is None:
        import zeyrek
        _ANALYZER = zeyrek.MorphAnalyzer()
    return _ANALYZER


# 1. Veriyi yükle — koda gömülü etiketli Türkçe yorumlar (HAZIR VERİLDİ)
def load_data():
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
    text = text.replace('İ', 'i').replace('I', 'ı')
    return text.lower()
    pass


# 3. Metni temizle — noktalama ve sayıları at
def clean_text(text):
    text = turkish_lower(text)
    # Türkçe harfler + boşluk dışındaki her şeyi boşlukla değiştir
    text = re.sub(r'[^a-zçğıöşü\s]', ' ', text)
    # Fazla boşlukları teke indir
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    pass


# 4. Ön işleme — temizle + stopword çıkar
def preprocess(text):
    cleaned = clean_text(text)
    words = cleaned.split()
    stop = set(TURKISH_STOPWORDS)
    kept = [w for w in words if w not in stop]
    return ' '.join(kept)
    pass


# 5. Zeyrek ile lemmatization (kök bulma) — opsiyonel/ağır
def lemmatize_words(words):
    analyzer = _get_analyzer()
    lemmas = []
    for w in words:
        result = analyzer.lemmatize(w)
        # result: [(orijinal_kelime, [lemma1, lemma2, ...])]
        if result and result[0][1]:
            lemma = result[0][1][0]
            lemmas.append(turkish_lower(lemma))
        else:
            lemmas.append(turkish_lower(w))
    return lemmas
    pass


# 6. TF-IDF + LogisticRegression pipeline kur
def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=preprocess, analyzer='char_wb', ngram_range=(2, 4),
        )),
        ('clf', LogisticRegression(max_iter=1000, C=10)),
    ])
    pass


# 7. Modeli eğit
def train_model(pipe, X, y):
    pipe.fit(X, y)
    return pipe

    pass


# 8. Modeli değerlendir → accuracy
def evaluate(pipe, X, y):
    from sklearn.metrics import accuracy_score
    y_pred = pipe.predict(X)
    return float(accuracy_score(y, y_pred))

    pass


# 9. Tek bir cümle için duygu tahmini
def predict_sentiment(pipe, text):
    pred = int(pipe.predict([text])[0])
    return "pozitif" if pred == 1 else "negatif"
    pass


# 10. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.3, stratify=df['label'],
        random_state=42,
    )
    pipe = build_pipeline()
    pipe = train_model(pipe, X_train, y_train)
    acc = evaluate(pipe, X_test, y_test)
    sample_pos = predict_sentiment(
        pipe, "Bu ürün gerçekten harika, çok memnun kaldım tavsiye ederim",
    )
    sample_neg = predict_sentiment(
        pipe, "Berbat bir ürün, hiç beğenmedim paramı çöpe attım",
    )
    return {
        'test_accuracy': acc,
        'sample_positive': sample_pos,
        'sample_negative': sample_neg,
    }
    pass


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Test accuracy     : {result['test_accuracy']:.4f}")
    print(f"  Pozitif örnek     : {result['sample_positive']}")
    print(f"  Negatif örnek     : {result['sample_negative']}")
