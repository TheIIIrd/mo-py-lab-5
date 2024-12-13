import numpy as np
import matplotlib.pyplot as plt


def arithmetic_mean_filter(signal, r, weights):
    """Применение арифметического среднего для фильтрации сигнала."""
    filtered_signal = []  # Список для хранения отфильтрованного сигнала
    n = len(signal)  # Длина сигнала

    # Проходим по каждому элементу сигнала
    for k in range(n):
        num = 0  # Инициализация суммы значений
        den = 0  # Инициализация суммы весов

        # Обработка соседних значений в окне r
        for i in range(-r + 1, r):
            idx = k + i  # Индекс соседнего элемента сигнала

            if 0 <= idx < n:  # Проверка, что индекс в пределах сигнала
                w = weights[abs(i)]  # Получаем вес текущего элемента
                den += w  # Увеличиваем сумму весов
                num += w * signal[idx]  # Увеличиваем сумму значений с учетом весов

        # Добавляем отфильтрованное значение как среднее арифметическое
        filtered_signal.append(num / den if den != 0 else 0)

    return np.array(filtered_signal)  # Возвращаем отфильтрованный сигнал


def manhattan_metric(original, filtered):
    """Вычисление манхэттенского расстояния между оригинальным и отфильтрованным сигналами."""
    return np.sum(np.abs(original - filtered))


def random_search(signal, noisy_signal, r, n_iter=100):
    """Случайный поиск оптимальных весов фильтра."""
    best_weights = None  # Наилучшие найденные веса
    best_J = float("inf")  # Наименьшее значение критерия

    for _ in range(n_iter):
        weights = np.random.rand(r)  # Генерация случайных весов
        weights = weights / np.sum(weights)  # Нормализация весов
        filtered_signal = arithmetic_mean_filter(
            noisy_signal, r, weights
        )  # Фильтрация сигнала
        J = manhattan_metric(signal, filtered_signal)  # Оценка качества фильтрации
        # Сравнение и сохранение лучших параметров
        if J < best_J:
            best_J = J
            best_weights = weights
    return best_weights, best_J  # Возврат наилучших весов и значения критерия


def passive_search(signal, noisy_signal, r, L=10):
    """Пассивный поиск оптимальных весов фильтра."""
    best_lambda = None  # Наилучшее значение лямбды
    best_J = float("inf")  # Наименьшее значение критерия
    lambdas = [l / L for l in range(L + 1)]  # Генерация лямбд

    for lamb in lambdas:
        weights = np.random.rand(r)  # Генерация случайных весов
        weights = weights / np.sum(weights)  # Нормализация весов
        filtered_signal = arithmetic_mean_filter(
            noisy_signal, r, weights
        )  # Фильтрация сигнала

        J = manhattan_metric(signal, filtered_signal)  # Оценка качества фильтрации

        # Сравнение и сохранение лучших параметров
        if J < best_J:
            best_J = J
            best_lambda = lamb
    return best_lambda, best_J  # Возврат наилучшей лямбы и значения критерия


def compute_criteria(original, filtered):
    """Вычисление критериев ω и δ."""
    omega = np.sum(np.abs(original - filtered)) / len(original)  # Среднее отклонение
    delta = np.max(np.abs(original - filtered))  # Максимальное отклонение

    return omega, delta  # Возврат критериев


def passive_search_criteria(signal, noisy_signal, r, L=10):
    """Пассивный поиск с вычислением критериев ω и δ."""
    lambdas = [l / L for l in range(L + 1)]  # Генерация лямбд
    omega_values = []  # Список для хранения значений ω
    delta_values = []  # Список для хранения значений δ
    weights_values = []  # Список для хранения весов

    for lamb in lambdas:
        weights = np.random.rand(r)  # Генерация случайных весов
        weights = weights / np.sum(weights)  # Нормализация весов
        filtered_signal = arithmetic_mean_filter(
            noisy_signal, r, weights
        )  # Фильтрация сигнала
        omega, delta = compute_criteria(signal, filtered_signal)  # Вычисление критериев
        omega_values.append(omega)  # Сохранение значения ω
        delta_values.append(delta)  # Сохранение значения δ
        weights_values.append(weights)  # Сохранение весов

    return (
        lambdas,
        omega_values,
        delta_values,
        weights_values,
    )  # Возврат результатов поиска


def main():
    # Исходные данные
    Xmin, Xmax = 0, np.pi  # Минимальное и максимальное значение x
    K = 100  # Количество точек в графике
    a = 0.25  # Максимальная амплитуда шума
    L = 10  # Количество лямбд для поиска
    r_values = [3, 5]  # Значения окна фильтрации

    # Генерация равномерно распределенных точек
    x_k = np.linspace(Xmin, Xmax, K + 1)

    # Создание исходного сигнала f_k
    f_k = np.sin(x_k) + 0.5  # Синусоидальный сигнал + смещение

    # Установка генератора случайных чисел для воспроизводимости
    np.random.seed(42)
    # Генерация шума
    noise = np.random.uniform(-a, a, len(f_k))
    # Создание зашумленного сигнала
    f_noisy = f_k + noise

    # Графическое отображение и фильтрация для разных значений r
    for r in r_values:
        # Случайный поиск весов
        best_weights, best_J = random_search(f_k, f_noisy, r)
        # Фильтрация зашумленного сигнала с найденными весами
        filtered_signal_random = arithmetic_mean_filter(f_noisy, r, best_weights)

        # Пассивный поиск критериев
        lambdas, omega_values, delta_values, weights_values = passive_search_criteria(
            f_k, f_noisy, r
        )

        # Пассивный поиск с вычислением наилучшей лямбды
        best_lambda, best_J_passive = passive_search(f_k, f_noisy, r)
        weights_passive = np.array(
            [best_lambda**i for i in range(r + 1)]
        )  # Создание весов на основе лямбды
        weights_passive = weights_passive / np.sum(
            weights_passive
        )  # Нормализация весов
        # Фильтрация зашумленного сигнала с пассивными весами
        filtered_signal_passive = arithmetic_mean_filter(f_noisy, r, weights_passive)

        # Визуализация сигналов
        plt.figure(figsize=(12, 6))
        plt.plot(
            x_k, f_k, label="Исходный сигнал $f_k$", linewidth=2
        )  # Исходный сигнал
        plt.plot(
            x_k, f_noisy, label="Зашумленный сигнал $f'_k$", linewidth=2
        )  # Зашумленный сигнал
        plt.plot(
            x_k,
            filtered_signal_random,
            label=f"Фильтрованный сигнал (случайный)",
            linewidth=2,
        )  # Фильтрованный сигнал

        # Настройка графика
        plt.title(f"Фильтрация сигнала с окном r={r}")
        plt.xlabel("$x_k$")
        plt.ylabel("$f_k$, $f'_k$")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Визуализация критериев ω и δ
        plt.figure(figsize=(8, 6))
        plt.scatter(
            omega_values,
            delta_values,
            c=lambdas,
            cmap="viridis",
            s=50,
            edgecolor="k",
            label="$\lambda_1$",
        )
        plt.scatter(0, 0, c='red', s=100, edgecolor='k', label="Utopia (0, 0)")

        # Аннотирование точек на графике
        for i, lamb in enumerate(lambdas):
            plt.annotate(
                f"{lamb:.2f}",
                (omega_values[i], delta_values[i]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="center",
            )
        plt.title(f"Критерии $\omega$ и $\delta$ для $r={r}$")
        plt.xlabel("Критерий $\omega$ (среднее отклонение)")
        plt.ylabel("Критерий $\delta$ (максимальное отклонение)")
        plt.grid(True)
        plt.show()

        # Вывод результирующей информации
        print(f"\nРезультаты для окна r={r}:")
        print("=" * 110)
        print(f"{'Лямбда':<10}{'Сумма ω и δ':<20}{'Вес':<45}{'ω':<15}{'δ':<15}")
        print("=" * 110)
        omegi = []  # Список для хранения суммы критериев

        for i in range(len(lambdas)):
            weighted_sum = omega_values[i] + delta_values[i]
            omegi.append(weighted_sum)  # Сумма ω и δ
            print(
                f"{lambdas[i]:<10.2f}{weighted_sum:<20.4f}{np.round(weights_values[i], 4)}{'':<20}{omega_values[i]:<15.4f}{delta_values[i]:<15.4f}"
            )

        # Наименьшая сумма критериев
        o = min(omegi)
        indd = omegi.index(o)  # Индекс наименьшей суммы
        print("=" * 110)
        print(
            f"h* = 0.{indd}, dis = {o:.4f}, J = {max(weights_values[indd]):.4f}, w = {omega_values[indd]:.4f}, d = {delta_values[indd]:.4f}"
        )


if __name__ == "__main__":
    main()  # Запуск основной программы
