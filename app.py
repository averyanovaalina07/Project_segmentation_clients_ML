import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Анализ сегментации клиентов", layout="wide")

st.title("📊 Интерактивная сегментация клиентов (RFM + PCA)")
st.markdown("Передвигайте слайдер слева, чтобы изменить количество групп клиентов.")

# --- Функция загрузки данных с кэшированием (чтобы не ждать по 2 минуты) ---
@st.cache_data
def load_and_clean_data():
    # Используем твой путь к файлу
    df = pd.read_excel('data/Online_Retail.xlsx')
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    customers = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum',
        'StockCode': 'nunique'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSum': 'Monetary', 'StockCode': 'Variety'})

    customers['AvgOrderValue'] = customers['Monetary'] / customers['Frequency']
    return customers


@st.cache_data
def compute_elbow(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple[list, list]:
    """Run KMeans for k = k_min..k_max and return (K_range, inertias) for the elbow chart."""
    k_range = list(range(k_min, k_max + 1))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    return k_range, inertias


def explode_clusters(coords: np.ndarray, labels: np.ndarray, separation: float = 1.0) -> np.ndarray:
    """Cosmetic post-processing that visually pushes clusters apart on the 2D canvas.

    Each point is translated along the vector (global_center → cluster_center) by
    `separation` factor. Does NOT change clustering or any metric — only the picture.
    """
    coords = np.asarray(coords, dtype=float)
    if abs(separation - 1.0) < 1e-9:
        return coords

    out = np.zeros_like(coords)
    global_center = coords.mean(axis=0)
    unique_labels = sorted(set(labels))
    n_clusters = max(len(unique_labels), 1)

    for i, c in enumerate(unique_labels):
        mask = labels == c
        cluster_center = coords[mask].mean(axis=0)
        direction = cluster_center - global_center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            # Degenerate (cluster center coincides with global center):
            # fall back to a deterministic direction on a unit circle.
            angle = 2 * np.pi * i / n_clusters
            direction = np.array([np.cos(angle), np.sin(angle)])
        new_center = global_center + direction * separation
        offsets = coords[mask] - cluster_center
        out[mask] = new_center + offsets

    return out


# Загружаем данные
with st.spinner('Загрузка огромного файла Excel... Пожалуйста, подождите.'):
    customers = load_and_clean_data()

# --- Боковая панель (Sidebar) ---
st.sidebar.header("Настройки модели")
k_clusters = st.sidebar.slider("Количество кластеров (k)", min_value=2, max_value=10, value=5)
separation = st.sidebar.slider(
    "Раздвинуть кластеры (визуально)",
    min_value=1.0, max_value=5.0, value=1.0, step=0.25,
    help=(
        "Косметика: каждый кластер смещается от центра картинки. "
        "На метрики и кластеризацию не влияет — только на отрисовку."
    ),
)

# --- Обработка данных ---
features = ['Recency', 'Frequency', 'Monetary', 'Variety', 'AvgOrderValue']
X_log = np.log1p(customers[features])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(X_scaled)
customers['PCA1'] = pca_results[:, 0]
customers['PCA2'] = pca_results[:, 1]

# KMeans с динамическим k
kmeans = KMeans(n_clusters=k_clusters, init='k-means++', random_state=42)
customers['Segment'] = kmeans.fit_predict(X_scaled)

# Cosmetic: visually push clusters apart on the canvas (does not affect clustering).
exploded = explode_clusters(
    customers[['PCA1', 'PCA2']].to_numpy(),
    customers['Segment'].to_numpy(),
    separation=separation,
)
customers['PCA1'] = exploded[:, 0]
customers['PCA2'] = exploded[:, 1]

# --- Главная визуализация ---
title = f"Визуализация для {k_clusters} сегментов"
if separation > 1.0 + 1e-9:
    title += f" — раздвинуто ×{separation:g}"
st.subheader(title)

fig, ax = plt.subplots(figsize=(12, 7))
sns.scatterplot(x='PCA1', y='PCA2', hue='Segment', data=customers,
                palette='bright', s=60, alpha=0.7, ax=ax)
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
st.pyplot(fig)

if separation > 1.0 + 1e-9:
    st.caption(
        "🎨 Координаты раздвинуты косметически — это оформительский приём для наглядности. "
        "Расстояния на картинке больше не отражают реальное расстояние между клиентами."
    )

# --- Статистика по сегментам (под графиком) ---
st.subheader("Статистика по сегментам")
stats = customers.groupby('Segment').agg(
    Recency=('Recency', 'mean'),
    Frequency=('Frequency', 'mean'),
    Monetary=('Monetary', 'mean'),
    Variety=('Variety', 'mean'),
    AvgOrderValue=('AvgOrderValue', 'mean'),
    Покупателей=('Recency', 'size'),
)
st.dataframe(
    stats.style.format({
        'Recency': '{:.0f}',
        'Frequency': '{:.1f}',
        'Monetary': '£{:.0f}',
        'Variety': '{:.1f}',
        'AvgOrderValue': '£{:.2f}',
        'Покупателей': '{:.0f}',
    }),
    use_container_width=True,
)
st.info(
    "Среднее по сегменту для всех 5 признаков, на которых учится KMeans: "
    "Recency (давность, дней), Frequency (число заказов), Monetary (выручка), "
    "Variety (разных товаров), AvgOrderValue (средний чек), плюс размер сегмента."
)

# --- Bar chart: размер сегментов ---
st.subheader("Размер сегментов")
sizes = customers['Segment'].value_counts().sort_index()
bar_colors = [
    '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974',
    '#64B5CD', '#8C8C8C', '#D4A6C8', '#9E80B5', '#F7A35C',
]

fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
sizes.plot(kind='bar', ax=ax_bar, color=bar_colors[:len(sizes)])
ax_bar.set_xlabel('Сегмент')
ax_bar.set_ylabel('Покупателей')
ax_bar.tick_params(axis='x', rotation=0)
for i, v in enumerate(sizes):
    ax_bar.text(i, v + max(sizes) * 0.02, str(v), ha='center', fontweight='bold')
fig_bar.tight_layout()
st.pyplot(fig_bar)

# --- Метод локтя ---
st.subheader("Метод локтя — подбор количества кластеров")
st.write(
    "Ищем точку, где кривая резко перестаёт падать (характерный «излом») — "
    "это и есть оптимальное K. Красная пунктирная линия показывает "
    "текущее значение K из слайдера слева."
)

k_range, inertias = compute_elbow(X_scaled, k_min=2, k_max=10)

fig_elbow, ax_elbow = plt.subplots(figsize=(10, 5))
ax_elbow.plot(k_range, inertias, marker='o', linewidth=2, markersize=8, color='#4C72B0')
ax_elbow.axvline(x=k_clusters, color='#C44E52', linestyle='--', alpha=0.7,
                 label=f'Текущее K = {k_clusters}')
ax_elbow.set_xlabel('Количество кластеров (K)')
ax_elbow.set_ylabel('Инерция (сумма квадратов расстояний внутри кластеров)')
ax_elbow.set_xticks(k_range)
ax_elbow.legend()
ax_elbow.grid(True, alpha=0.3)
fig_elbow.tight_layout()
st.pyplot(fig_elbow)

# --- Поиск клиента ---
st.divider()
st.subheader("🔍 Проверить конкретного клиента")
search_id = st.text_input("Введите CustomerID (например, 12346.0):")

if search_id:
    try:
        user_data = customers.loc[float(search_id)]
        st.success(f"Клиент {search_id} относится к сегменту №{int(user_data['Segment'])}")
        st.write(user_data[features])
    except:
        st.error("Клиент с таким ID не найден.")

# --- Таблица всех данных ---
with st.expander("Посмотреть всю таблицу результатов"):
    st.write(customers)
