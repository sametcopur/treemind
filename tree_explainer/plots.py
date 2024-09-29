import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import seaborn as sns

sns.set_theme()


def plot_bar(values: np.ndarray, raw_score: float):
    # Sıfır olmayan sütunları bul ve ayıkla
    used_cols = np.where(np.logical_not(values == 0))[0]
    values = values[used_cols]

    # Verileri mutlak değerlerine göre büyükten küçüğe sırala
    sorted_indices = np.argsort(np.abs(values))
    adjusted_values = values[sorted_indices]
    used_cols = used_cols[sorted_indices]

    # Renk ayarları
    colors = ["green" if val > 0 else "red" for val in adjusted_values]

    # Barları çiz
    fig, ax = plt.subplots()
    bars = ax.barh(np.arange(len(used_cols)), adjusted_values, color=colors)

    # Orta nokta (sıfır) çizgisi
    ax.axvline(0, color="black", linewidth=0.8)

    # Y eksenini numaralandır
    ax.set_yticks(np.arange(len(used_cols)))
    ax.set_yticklabels([f"Column {i}" for i in used_cols])

    # X eksenine başlık ekle
    ax.set_xlabel("Contribution")

    # Barların ortasına beyaz renkte etiket ekle
    for index, bar in enumerate(bars):
        ax.text(
            bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{adjusted_values[index]:.3f}",
            ha="center",
            va="center",
            color="white",
        )

    # Sağ alt köşeye raw_score ekle
    plt.text(
        1,
        -0.1,
        f"Raw Score: {raw_score:.3f}",
        fontsize=10,
        ha="right",
        va="top",
        transform=ax.transAxes,
    )

    plt.title("Bar Plot of Contribution")
    plt.show()


def plot_values_points(values, raw_score, points, scale = 1):

    used_cols = np.where(np.logical_not(np.all(values == 0, axis=1)))[0]
    values = values[used_cols, :]
    points = [x for i, x in enumerate(points) if i in used_cols]

    fig, ax = plt.subplots(figsize=(len(values[0]) * 2 * scale, len(values)* scale))

    # Pozitif ve negatif değerler için renk haritaları
    cmap_pos = sns.color_palette("Greens", as_cmap=True)
    cmap_neg = sns.color_palette("Reds", as_cmap=True)

    # Değerlerin maksimum ve minimum sınırlarını bul
    max_value = np.max(values)
    min_value = np.min(values)

    n_rows, n_cols = values.shape
    bar_width = 1  # Bar genişliği
    bar_height = 0.8  # Bar yüksekliği
    spacing = 0  # Barlar tam bitişik

    # Her satır ve sütun için verileri göster
    for i, row in enumerate(values):
        for j, val in enumerate(row):
            if val != 0:  # 0 değerlerini görmezden geliyoruz
                # Pozitif ve negatif değerler için farklı renkler
                if val > 0:
                    color = cmap_pos(val / max_value)  # Pozitif değerler için yeşil ton
                else:
                    color = cmap_neg(
                        abs(val) / abs(min_value)
                    )  # Negatif değerler için kırmızı ton

                # Barları çiz
                ax.barh(
                    i,
                    bar_width,
                    left=j * (bar_width + spacing),
                    color=color,
                    height=bar_height,
                    alpha=0.9,
                )

                # Rengin parlaklığını hesapla
                color_hsv = rgb_to_hsv(
                    color[:3]
                )  # Sadece RGB kısmını alıyoruz (alpha yok)
                brightness = color_hsv[2]  # Value (brightness) kısmını alıyoruz

                # Parlaklığa göre yazı rengini ayarla (koyu renk için beyaz, açık renk için siyah)
                text_color = "white" if brightness < 0.5 else "black"

                # İlgili points değerini yaz (0.3f formatında, büyütülmüş font ile)
                ax.text(
                    j * (bar_width + spacing) + bar_width / 2,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=20,
                )

    # Y ekseni etiketleri
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f"Column {i}" for i in used_cols])

    # X eksenini ayarla (sütun sayısına göre genişlik)
    ax.set_xlim(0, n_cols)

    # Eksen çizgilerini kaldır (isteğe bağlı)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.text(
        0.98,
        0.98,
        f"Raw Score: {raw_score:.3f}",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=25,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    plt.show()


def create_intervals(point_row):
    intervals = []
    point_row = np.array(point_row)  # Numpy array formatına çevirdik
    intervals.append(f"(-inf, {point_row[0]:.3f}]")  # İlk aralık (-inf, first_value)

    for i in range(1, len(point_row) - 1):
        intervals.append(
            f"({point_row[i-1]:.3f}, {point_row[i]:.3f}]"
        )  # [prev_value, current_value)

    intervals.append(f"({point_row[-2]:.3f}, inf)")  # Son aralık [last_value, inf)
    return intervals


def plot_points(values, points, scale = 1):
    used_cols = np.where(np.logical_not(np.all(values == 0, axis=1)))[0]
    values = values[used_cols, :]
    points = [x for i, x in enumerate(points) if i in used_cols]

    # Verilen points listesini aralıklara çeviriyoruz
    points = [create_intervals(row) for row in points]

    fig, ax = plt.subplots(figsize=(len(values[0]) * 2 * scale, len(points)* scale))

    # Pozitif ve negatif değerler için renk haritaları
    cmap_pos = sns.color_palette("Greens", as_cmap=True)
    cmap_neg = sns.color_palette("Reds", as_cmap=True)

    # Değerlerin maksimum ve minimum sınırlarını bul
    max_value = np.max(values)
    min_value = np.min(values)

    n_rows = len(points)  # Satır sayısı (points'den alıyoruz)
    bar_width = 1  # Bar genişliği
    bar_height = 0.8  # Bar yüksekliği
    spacing = 0  # Barlar tam bitişik

    # Her satır ve sütun için verileri göster
    for i, row_points in enumerate(points):

        for j, point in enumerate(row_points):
            # Renk paletinden uygun rengi al (önceki grafikteki değerlerle aynı renkleri tutuyoruz)
            val = values[i, j] if j < values.shape[1] else 0  # Değer yoksa 0 al
            if val > 0:
                color = cmap_pos(val / max_value)  # Pozitif değerler için yeşil ton
            else:
                color = cmap_neg(
                    abs(val) / (abs(min_value) + 1e-8)
                )  # Negatif değerler için kırmızı ton

            # Barları çiz (Renkler korunuyor)
            ax.barh(
                i,
                bar_width,
                left=j * (bar_width + spacing),
                color=color,
                height=bar_height,
                alpha=0.9,
            )

            # Rengin parlaklığını hesapla
            color_hsv = rgb_to_hsv(color[:3])  # Sadece RGB kısmını alıyoruz (alpha yok)
            brightness = color_hsv[2]  # Value (brightness) kısmını alıyoruz

            # Parlaklığa göre yazı rengini ayarla (koyu renk için beyaz, açık renk için siyah)
            text_color = "white" if brightness < 0.5 else "black"

            # İlgili points değerini yaz (aralık formatında)
            ax.text(
                j * (bar_width + spacing) + bar_width / 2,
                i,
                point,
                ha="center",
                va="center",
                color=text_color,
                fontsize=15,
            )

    # Y ekseni etiketleri
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f"Column {i}" for i in used_cols])

    # X eksenini ayarla (satırdaki maksimum sütun sayısına göre genişlik)
    max_cols = max(len(row_points) for row_points in points)
    ax.set_xlim(0, max_cols)

    # Eksen çizgilerini kaldır (isteğe bağlı)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.show()


def plot_dependecy(df, figsize = (10, 8)):
    sub_column_name = df.columns[0]
    main_columns_name = df.columns[1]

    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        x=df[main_columns_name],
        y=df[sub_column_name],
        c=df["values"],  # Renkler values sütununa göre
        cmap="RdYlGn",  # Eski renk paleti
        s=(df["values"] - df["values"].min()) * 30
        + 100,  # Nokta boyutunu değerlerle dinamik olarak ayarlama
        edgecolor="black",  # Noktaların kenar rengi
        vmin=df["values"].min(),  # Renk geçişinin başlangıcı
        vmax=df["values"].max(),  # Renk geçişinin bitişi
    )

    # Colorbar ekleme
    cbar = plt.colorbar(scatter, label="Values")
    cbar.ax.tick_params(labelsize=12)  # Colorbar yazı boyutunu ayarlama

    # Eksen etiketleri ve başlık
    plt.xlabel(main_columns_name, fontsize=14)
    plt.ylabel(sub_column_name, fontsize=14)
    plt.title("Dependency Plot", fontsize=16)

    plt.grid(True, linestyle="--", alpha=0.7)

    plt.xlim(df[main_columns_name].min() - 1, df[main_columns_name].max() + 1)
    plt.ylim(df[sub_column_name].min() - 1, df[sub_column_name].max() + 1)

    plt.tight_layout()  # Grafik alanını optimize etme
    plt.show()


def plot_feature(df, figsize = (10, 6)):
    column_name = df.columns[0]

    plt.figure(figsize=figsize)

    sns.scatterplot(
        data=df,
        x=column_name,
        y="mean",
        color="blue",
        s=100,
        edgecolor="w",
        linewidth=0.5,
    )

    plt.errorbar(
        df[column_name],
        df["mean"],
        yerr=[df["mean"] - df["min"], df["max"] - df["mean"]],
        fmt="o",
        ecolor="gray",
        capsize=5,
        elinewidth=1,
        capthick=1,
    )

    plt.title(f"Contribution of {column_name}", fontsize=16, fontweight="bold")
    plt.xlabel(column_name, fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.7)
    plt.grid(True)

    plt.show()
