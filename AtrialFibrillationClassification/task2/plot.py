import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from ecg_dataset import ECG_dataset
from svm_plot import plot_decision_regions
import heartpy as hp


def extract_ecg_features(ecg_signal, sampling_rate):
    # 提取R波峰值
    wd, m = hp.process(ecg_signal, sample_rate=sampling_rate, bpmmin=1, bpmmax=1000)

    # 心率
    hr = m['bpm']

    # 心率变异性 - 时域
    hrv_time = m['sdnn']

    # 心率变异性 - 频域
    lf = m.get('lf', np.nan)  # 低频
    hf = m.get('hf', np.nan)  # 高频

    # R波峰值间隔
    rr_intervals = np.diff(wd['peaklist'])

    # 计算波形复杂度 (示例: 标准差)
    waveform_complexity = np.std(ecg_signal)

    return [hr, hrv_time, np.mean(rr_intervals), waveform_complexity]


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


if __name__ == "__main__":
    heart_rate_data = ECG_dataset('./')
    # d = heart_rate_data[1]
    # x = np.linspace(0, 3000, 3000)  # x轴坐标值
    # plt.plot(x, d[0], c='r')  # 参数c为color简写，表示颜色,r为red即红色
    # plt.show()  # 显示图像'''
    sample_rating = 100
    features = []
    label = []

    for i in range(len(heart_rate_data)):
        data = (heart_rate_data[i][0] - np.min(heart_rate_data[i][0])) / (np.max(heart_rate_data[i][0]) - np.min(heart_rate_data[i][0]))

        data = butter_bandpass_filter(data, 0.5, 40, sample_rating)
        p = extract_ecg_features(data, sample_rating)
        features.append(np.array(p))

        label.append(np.array(heart_rate_data[i][1]))

    label = np.array(label)
    features = np.array(features)
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 10000

    X_train, X_test, y_train, y_test = train_test_split(features[:, :2], label, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.fit_transform(X_test)

    svm = SVC()
    svm.fit(X_train_scaler, y_train[:, 0])

    pred = svm.predict(X_test_scaler)

    print(pred)
    print(y_test[:, 0])
    print("Accuracy:", accuracy_score(y_test[:, 0], pred))
    print("Classification Report:")
    print(classification_report(y_test[:, 0], pred))
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    for i in range(len(features)):
        plt.plot(features[i, 0], features[i, 1], marker=markers[label[i, 0]])
    plt.show()
