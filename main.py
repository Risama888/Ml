import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import ta
from scipy.signal import find_peaks
import pickle
import fractal
import schedule
import time

# Konfigurasi Telegram
TELEGRAM_TOKEN = '7795073622:AAFEHjnKKNAUv2SEwkhLpvblMqolLNjSP48'
TELEGRAM_CHAT_ID = '6157064978'

status_file = 'signal_status.json'
model_file = 'elliott_rf_model.pkl'
dataset_file = 'dataset_labels.csv'  # Pastikan dataset sudah ada dan lengkap

def load_status():
    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_status(status):
    with open(status_file, 'w') as f:
        json.dump(status, f)

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print("Gagal mengirim pesan Telegram:", response.text)
    except Exception as e:
        print("Error saat mengirim pesan Telegram:", str(e))

def get_price_data():
    df = yf.download('BTC-USD', period='3d', interval='15m')
    return df

def calculate_technical_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['Bollinger_Upper'] = bollinger.bollinger_hband()
    df['Bollinger_Lower'] = bollinger.bollinger_lband()
    df = df.dropna()
    return df

def create_features(df):
    df_feat = pd.DataFrame()
    df_feat['Close'] = df['Close']
    df_feat['MA20'] = df['MA20']
    df_feat['MA50'] = df['MA50']
    df_feat['RSI'] = df['RSI']
    df_feat['Bollinger_Upper'] = df['Bollinger_Upper']
    df_feat['Bollinger_Lower'] = df['Bollinger_Lower']
    return df_feat

def train_ml_model():
    try:
        df = pd.read_csv(dataset_file)
        features = ['Close', 'MA20', 'MA50', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower']
        X = df[features]
        y = df['Pattern_Label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model dilatih. Akurasi: {acc*100:.2f}%")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print("Model disimpan.")
    except Exception as e:
        print("Gagal melatih model:", e)

def load_ml_model():
    try:
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    except:
        print("Model belum tersedia. Silakan lakukan pelatihan terlebih dahulu.")
        return None

def predict_pattern(df_feat, model):
    pred = model.predict(df_feat)[-1]
    proba = max(model.predict_proba(df_feat)[-1])
    return pred, proba

def analyze_fractal(df):
    try:
        dim = fractal.higuchi(df['Close'].values, kmax=10)
        return dim
    except:
        return None

def identify_elliott_waves(df, min_wave_length=5):
    highs, _ = find_peaks(df['Close'], distance=min_wave_length)
    lows, _ = find_peaks(-df['Close'], distance=min_wave_length)
    points = []
    for idx in highs:
        points.append({'index': idx, 'type': 'high', 'price': df.iloc[idx]['Close']})
    for idx in lows:
        points.append({'index': idx, 'type': 'low', 'price': df.iloc[idx]['Close']})
    points = sorted(points, key=lambda x: x['index'])
    waves = []
    for i in range(1, len(points)):
        start = points[i-1]
        end = points[i]
        start_price = start['price']
        end_price = end['price']
        retracement = abs(end_price - start_price) / start_price if start_price != 0 else 0
        if start['type'] == 'high' and end['type'] == 'low':
            wave_type = 'corrective'
        elif start['type'] == 'low' and end['type'] == 'high':
            wave_type = 'impulsive'
        else:
            wave_type = 'neutral'
        if wave_type == 'corrective' and 0.382 < retracement < 0.618:
            wave_class = 'corrective'
        elif wave_type == 'impulsive' and retracement < 0.382:
            wave_class = 'impulsive'
        else:
            wave_class = wave_type
        waves.append({
            'start_idx': start['index'],
            'end_idx': end['index'],
            'type': wave_class,
            'retracement': retracement
        })
    return waves

def analyze_elliott_structure(waves):
    impulsive = [w for w in waves if w['type'] == 'impulsive']
    corrective = [w for w in waves if w['type'] == 'corrective']
    if len(impulsive) >= 3 and len(corrective) >= 2:
        return 'Elliott Pattern Detected'
    return 'No Clear Pattern'

def main():
    # Otomatisasi pelatihan model setiap hari pukul 00:00
    schedule.every().day.at("00:00").do(train_ml_model)
    print("Penjadwalan pelatihan model setiap hari pukul 00:00.")

    while True:
        schedule.run_pending()

        # Proses utama setiap interval (misalnya setiap 15 menit)
        # Jika ingin otomatis setiap 15 menit, aktifkan bagian berikut:
        try:
            # Data terbaru
            df = get_price_data()
            df = calculate_technical_indicators(df)
            df_feat = create_features(df)

            # Muat model
            model = load_ml_model()
            if model is None:
                print("Menunggu pelatihan model...")
                time.sleep(60*15)  # tunggu 15 menit dan cek lagi
                continue

            # Prediksi pola ML
            pattern_pred, confidence = predict_pattern(df_feat, model)

            # Analisis fractal
            fractal_dim = analyze_fractal(df)

            # Deteksi pola Elliott Wave
            waves = identify_elliott_waves(df)
            pattern_status = analyze_elliott_structure(waves)

            # Sinyal berdasarkan analisis
            latest = df.iloc[-1]
            latest_close = latest['Close']
            rsi = latest['RSI']

            signal = None
            sl_price = None
            tp_price = None

            if pattern_status == 'Elliott Pattern Detected':
                if rsi < 30:
                    signal = 'buy'
                    sl_price = latest_close * 0.98
                    tp_price = latest_close * 1.05
                elif rsi > 70:
                    signal = 'sell'
                    sl_price = latest_close * 1.02
                    tp_price = latest_close * 0.95
            else:
                if pattern_pred == 1 and confidence > 0.7:
                    if rsi < 30:
                        signal = 'buy'
                        sl_price = latest_close * 0.98
                        tp_price = latest_close * 1.05
                    elif rsi > 70:
                        signal = 'sell'
                        sl_price = latest_close * 1.02
                        tp_price = latest_close * 0.95

            # Kirim sinyal jika berbeda dari terakhir
            status = load_status()
            if signal:
                if status.get('last_signal') != signal:
                    msg = f"<b>[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]</b>\n"
                    msg += f"<b>Symbol:</b> BTC-USD\n"
                    msg += f"<b>Close:</b> {latest_close:.2f}\n"
                    msg += f"<b>RSI:</b> {rsi:.2f}\n"
                    msg += f"<b>Fractal Dimensi:</b> {fractal_dim}\n"
                    msg += f"<b>Pattern Elliot:</b> {pattern_status}\n"
                    msg += f"<b>Sinyal:</b> {signal.upper()}\n"
                    msg += f"Entry Price: {latest_close:.2f}\n"
                    msg += f"Stop Loss (SL): {sl_price:.2f}\n"
                    msg += f"Take Profit (TP): {tp_price:.2f}"
                    send_telegram_message(msg)
                    # Update status
                    status['last_signal'] = signal
                    status['entry_price'] = latest_close
                    status['sl_price'] = sl_price
                    status['tp_price'] = tp_price
                    save_status(status)
                else:
                    print(f"Sinyal {signal} sudah dikirim sebelumnya.")
            else:
                print("Tidak ada sinyal valid saat ini.")

        except Exception as e:
            print("Error dalam proses utama:", e)

        # Tunggu 15 menit sebelum proses berikutnya
        time.sleep(60*15)

if __name__ == "__main__":
    main()
