import pytest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_data, turkish_lower, clean_text, TURKISH_STOPWORDS,
    preprocess, lemmatize_words, build_pipeline, train_model,
    evaluate, predict_sentiment, run_pipeline,
)

from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────
# Modül-seviye cache — testler arası tekrar eğitme yok
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    return load_data()


@pytest.fixture(scope="module")
def split(df):
    return train_test_split(
        df['text'], df['label'], test_size=0.3, stratify=df['label'],
        random_state=42,
    )


@pytest.fixture(scope="module")
def trained_pipeline(split):
    X_train, X_test, y_train, y_test = split
    pipe = build_pipeline()
    return train_model(pipe, X_train, y_train)


# 1. load_data
def test_load_data_structure(df):
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['text', 'label']
    # ~30 etiketli yorum
    assert 20 < len(df) < 60
    # İki sınıf da var (0 ve 1)
    assert set(df['label'].unique()) == {0, 1}
    # Dengeli olmalı
    assert df['label'].sum() == len(df) - df['label'].sum()


# 2. turkish_lower — Türkçe'ye duyarlı küçük harf
def test_turkish_lower():
    assert turkish_lower("İSTANBUL") == "istanbul"
    assert turkish_lower("İyi") == "iyi"
    # Türkçe 'I' → 'ı' (İngilizce 'i' değil)
    assert turkish_lower("IRMAK") == "ırmak"


# 3. clean_text — noktalama ve sayıları siler
def test_clean_text_removes_punctuation():
    out = clean_text("Harika, çok güzel!!! 123")
    assert ',' not in out
    assert '!' not in out
    assert '1' not in out
    # Türkçe harfler korunur
    assert 'çok' in out
    assert 'güzel' in out
    # lowercase uygulanmış
    assert out == out.lower()


# 4. TURKISH_STOPWORDS — liste ve içerik
def test_turkish_stopwords_is_list():
    assert isinstance(TURKISH_STOPWORDS, list)
    assert len(TURKISH_STOPWORDS) > 5
    assert all(isinstance(w, str) for w in TURKISH_STOPWORDS)
    # Yaygın stopword'ler içermeli
    assert 've' in TURKISH_STOPWORDS
    assert 'bir' in TURKISH_STOPWORDS


# 5. preprocess — stopword çıkarır
def test_preprocess_removes_stopwords():
    out = preprocess("bu ürün ve kargo çok güzel bir deneyim")
    words = out.split()
    # 've', 'bu', 'bir', 'çok' stopword → çıkmalı
    assert 've' not in words
    assert 'bu' not in words
    assert 'bir' not in words
    # Anlamlı kelimeler kalmalı
    assert 'ürün' in words
    assert 'kargo' in words
    assert 'güzel' in words


# 6. lemmatize_words — Zeyrek ile kök bulma
def test_lemmatize_words():
    out = lemmatize_words(["kitaplar", "geldim"])
    assert isinstance(out, list)
    assert len(out) == 2
    # "kitaplar" → "kitap" (kök), "geldim" → "gelmek"
    assert out[0] == "kitap"
    assert "gel" in out[1]


# 7. build_pipeline — TF-IDF + LogisticRegression
def test_build_pipeline_type():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    names = dict(pipe.steps)
    assert 'tfidf' in names and 'clf' in names
    assert isinstance(names['tfidf'], TfidfVectorizer)
    assert isinstance(names['clf'], LogisticRegression)


# 8. train_model
def test_train_model_fits(trained_pipeline, split):
    X_train, X_test, y_train, y_test = split
    preds = trained_pipeline.predict(X_test)
    assert len(preds) == len(X_test)


# 9. evaluate — accuracy eşik üstü
def test_evaluate_accuracy(trained_pipeline, split):
    _, X_test, _, y_test = split
    acc = evaluate(trained_pipeline, X_test, y_test)
    assert 0 <= acc <= 1
    # Küçük ama temiz sette eşik üstü
    assert acc >= 0.8


# 10. predict_sentiment — net pozitif cümle
def test_predict_sentiment_positive(trained_pipeline):
    out = predict_sentiment(
        trained_pipeline,
        "Bu ürün gerçekten harika, çok memnun kaldım kesinlikle tavsiye ederim",
    )
    assert out == "pozitif"


# 11. predict_sentiment — net negatif cümle
def test_predict_sentiment_negative(trained_pipeline):
    out = predict_sentiment(
        trained_pipeline,
        "Berbat bir ürün, hiç beğenmedim paramı çöpe attım rezalet",
    )
    assert out == "negatif"


# 12. run_pipeline — uçtan uca
def test_run_pipeline_full():
    result = run_pipeline()
    assert set(result.keys()) >= {
        'test_accuracy', 'sample_positive', 'sample_negative'
    }
    assert result['test_accuracy'] >= 0.8
    assert result['sample_positive'] == "pozitif"
    assert result['sample_negative'] == "negatif"


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
