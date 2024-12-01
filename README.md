# Поиск жесткого соответствия

## Формулировка задачи

### Исходные данные

Пусть заданы два множества точек:

- $`X = \{ \mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N \} \subset \mathbb{R}^n`$,
- $`Y = \{ \mathbf{y}_1, \mathbf{y}_2, \dots, \mathbf{y}_N \} \subset \mathbb{R}^n`$,

где каждая точка $`\mathbf{x}_i \in \mathbb{R}^n`$, $`\mathbf{y}_i \in \mathbb{R}^n`$. Для трехмерного случая ($`n = 3`$) эти множества представлены матрицами:

```math
X = \left(\begin{matrix}
      x_1 & y_1 & z_1 \\
      x_2 & y_2 & z_2 \\
      \vdots & \vdots & \vdots \\
      x_N & y_N & z_N \\
    \end{matrix}\right),
\quad
Y = \left(\begin{matrix}
      x'_1 & y'_1 & z'_1 \\
      x'_2 & y'_2 & z'_2 \\
      \vdots & \vdots & \vdots \\
      x'_N & y'_N & z'_N \\
    \end{matrix}\right).
```

Цель состоит в нахождении жесткого преобразования (с сохранением расстояний), включающего вращение, смещение и перестановку точек, чтобы преобразовать $`X`$ в $`Y`$.

### Формулировка в терминах векторов

Для каждого $`\mathbf{x}_i \in X`$ требуется найти соответствующую точку $`\mathbf{y}_{\sigma(i)} \in Y`$ и параметры $`R \in SO(3)`$, $`t \in \mathbb{R}^n`$, такие, что выполняется следующее соотношение:

```math
\mathbf{y}_{\sigma(i)} = R \mathbf{x}_i + t + \boldsymbol{\epsilon}_i, \quad \forall i = 1, \dots, N,
```

Здесь:

- $`R \in SO(3)`$ — матрица вращения ($`R^T R = I, \det(R) = 1`$),
- $`t`$ — вектор смещения,
- $`\sigma: \{1, \dots, N\} \to \{1, \dots, N\}`$ — перестановка индексов.
- $`\boldsymbol{\epsilon}_i \in \mathbb{R}^n`$ — вектор шума для точки $`i`$.

Целевая функция минимизации для нахождения $`R`$, $`t`$ и $`\sigma`$ формулируется как:

```math
\min_{R \in SO(3), t, \sigma} \sum_{i=1}^N \| R \mathbf{x}_i + t - \mathbf{y}_{\sigma(i)} + \boldsymbol{\epsilon}_i \|^2,
```

где $`\|\cdot\|`$ — евклидова норма.

### Формулировка в терминах матриц

Для упрощения записи перестановка $`\sigma`$ представляется в виде бинарной матрицы $`P \in \{0, 1\}^{N \times N}`$, где $`P_{ij} = 1`$ означает, что точка $`\mathbf{x}_i`$ соответствует $`\mathbf{y}_j`$. Тогда преобразование точек записывается в матричной форме:

```math
Y^T = R X^T P + t \begin{pmatrix} 1 & \dots & 1 \end{pmatrix}_{1 \times N} + E,
```

Целевая функция в матричной форме при условии, что $`\| \boldsymbol{\epsilon}_i \| \leq \epsilon_{\max}, \quad \forall i`$

```math
\min_{R \in SO(3), t, P} \| R X^T P + t \begin{pmatrix} 1 & \dots & 1 \end{pmatrix}_{1 \times N} - Y^T \|_F^2,
```

где

- $`\|\cdot\|_F`$ — фробениусова норма,
- $`R \in SO(3)`$ — матрица вращения ($`R^T R = I, \det(R) = 1`$),
- $`P`$ — матрица перестановки ($`P_{ij} \in \{0, 1\}`$; каждая строка имеет одну единицу — $`\forall i \sum_{j=1}^N P_{ij} = 1`$; каждый столбец имеет одну единицу — $`\forall j \sum_{i=1}^N P_{ij} = 1`$),
- $`E \in \mathbb{R}^{n \times N}`$ — матрица шума с $`E[:, i] = \boldsymbol{\epsilon}_i`$.


### Задача жесткой регистрации

Цель состоит в нахождении $`R`$, $`t`$ и $`P`$ ($`\sigma(i)`$), минимизирующих отклонение между облаками точек. Это ключевая задача для алгоритмов регистрации точек.

---

## Существующие методы

### Случай, когда соответствие между точками уже известно

Если соответствие между точками двух облаков $`X`$ и $`Y`$ известно заранее (то есть перестановка $`P`$ или $`\sigma`$ уже задана), задача сводится к нахождению жесткого преобразования, состоящего из матрицы вращения $`R`$ и вектора смещения $`t`$.

Эта задача имеет аналитическое решение, часто называемое **методом минимизации суммарного квадрата ошибок (least squares method)**. Вот основные шаги решения:

#### Постановка задачи

Целевая функция:

```math
\min_{R \in SO(3), t} \sum_{i=1}^N \| R \mathbf{x}_i + t - \mathbf{y}_i \|^2,
```

где $`\mathbf{x}_i \in X`$ и $`\mathbf{y}_i \in Y`$ — соответствующие точки.

#### Решение задачи

1. **Смещение центроидов:**

Вычисляем центроиды двух облаков точек:

```math
   \mathbf{\bar{x}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i, \quad
   \mathbf{\bar{y}} = \frac{1}{N} \sum_{i=1}^N \mathbf{y}_i.
```

Затем центрируем данные:

```math
   \mathbf{x}'_i = \mathbf{x}_i - \mathbf{\bar{x}}, \quad \mathbf{y}'_i = \mathbf{y}_i - \mathbf{\bar{y}}.
```

1. **Построение ковариационной матрицы:**
   Ковариационная матрица $`H`$ строится как:

```math
   H = \sum_{i=1}^N \mathbf{x}'_i (\mathbf{y}'_i)^T.
```

3. **Поиск матрицы вращения $`R`$:**
   Используем разложение $`H = U \Sigma V^T`$ (SVD). Матрица $`R`$ находится как:

```math
   R = V U^T.
```

Если $`\det(R) < 0`$, это означает, что произошло отражение, и необходимо скорректировать матрицу $`V`$:

```math
   V[:,3] \leftarrow -V[:,3].
```

4. **Поиск вектора смещения $`t`$:**
   Смещение вычисляется как:

```math
   t = \mathbf{\bar{y}} - R \mathbf{\bar{x}}.
```

#### Итоговое решение

После вычисления $`R`$ и $`t`$, преобразование каждой точки $`\mathbf{x}_i`$ задается формулой:

```math
\mathbf{y}_i = R \mathbf{x}_i + t.
```

#### Преимущества метода

- **Эффективность:** Решение является аналитическим и может быть найдено за $`O(N)`$ операций для вычисления центроидов и ковариационной матрицы, плюс $`O(1)`$ для SVD.
- **Устойчивость:** Метод минимизирует квадратичную ошибку между двумя облаками точек.

#### Недостатки

- Метод предполагает, что соответствие между точками известно, что не всегда выполняется в реальных задачах.

---

### Iterative Closest Point (ICP)

**Iterative Closest Point (ICP)** — это популярный итеративный алгоритм для жесткой регистрации облаков точек. Он используется, когда соответствие между точками двух облаков не известно, и задача состоит в нахождении оптимального преобразования $`R`$ и $`t`$, чтобы минимизировать отклонение между облаками.

#### Основная идея

Алгоритм ICP состоит из двух ключевых этапов:

1. **Определение соответствий:** Для каждой точки $`\mathbf{x}_i \in X`$ из первого облака находим ближайшую точку $`\mathbf{y}_j \in Y`$ во втором облаке.
2. **Обновление преобразования:** На основе найденных соответствий вычисляем жесткое преобразование (матрицу вращения $`R`$ и вектор смещения $`t`$) с помощью метода минимизации квадратичной ошибки, описанного в предыдущем разделе.

Эти два этапа повторяются до сходимости алгоритма.

---

#### Постановка задачи

Целевая функция ICP минимизирует суммарное квадратичное отклонение между двумя облаками точек:

```math
\min_{R \in SO(3), t} \sum_{i=1}^N \| R \mathbf{x}_i + t - \mathbf{y}_{\sigma(i)} \|^2,
```

где:

- $`\sigma(i)`$ — индекс ближайшей точки $`\mathbf{y}_{\sigma(i)} \in Y`$ для $`\mathbf{x}_i \in X`$,
- $`R`$ — матрица вращения ($`R^T R = I, \det(R) = 1`$),
- $`t`$ — вектор смещения.

---

#### Шаги алгоритма

1. **Инициализация:**

   - Установить начальное приближение для $`R`$ и $`t`$ (например, $`R = I, t = \mathbf{0}`$).

2. **Определение соответствий:**

   - Для каждой точки $`\mathbf{x}_i \in X`$ найти ближайшую точку $`\mathbf{y}_j \in Y`$:

```math
     j = \arg \min_k \| R \mathbf{x}_i + t - \mathbf{y}_k \|.
```

3. **Обновление $`R`$ и $`t`$:**

   - На основе найденных соответствий обновить $`R`$ и $`t`$, используя метод минимизации квадратичной ошибки (например, через SVD).

4. **Проверка сходимости:**

   - Вычислить изменение ошибки между текущей и предыдущей итерациями. Если ошибка уменьшается незначительно (меньше заданного порога $`\epsilon`$), остановить алгоритм.

5. **Итерация:**
   - Повторить шаги 2–4, пока не будет достигнута сходимость.

#### Преимущества ICP

- **Простота реализации:** Алгоритм легко реализовать и понять.
- **Гибкость:** Может быть применен к 2D и 3D облакам точек.
- **Высокая точность:** При хорошей начальной инициализации обеспечивает точную регистрацию.

#### Недостатки ICP

1. **Зависимость от начальной инициализации:**

   - Если начальное положение облаков точек сильно отличается от правильного, алгоритм может сойтись к локальному минимуму.

2. **Чувствительность к шуму и выбросам:**

   - Ближайшие точки могут быть выбраны неправильно, что снижает точность.

3. **Сложность вычислений:**

   - Определение ближайших соседей требует $`O(N \log N)`$ операций при использовании KD-деревьев, что может быть вычислительно дорого для больших облаков точек.

#### Вывод

ICP — это базовый метод регистрации облаков точек, который подходит для многих приложений, если начальное приближение _достаточно близко к правильному решению_. Для более сложных случаев, таких как большой шум, выбросы или отсутствие хорошего начального приближения, могут потребоваться более сложные методы.

### RANSAC (Random Sample Consensus)

**RANSAC** — это итеративный алгоритм, используемый для оценки параметров модели в присутствии шума и выбросов. Алгоритм особенно полезен, когда данные содержат большое количество выбросов, которые могут негативно влиять на точность традиционных методов, таких как минимизация наименьших квадратов.

#### Основная идея

RANSAC строит модель, используя минимальное подмножество точек, и проверяет её согласованность с остальными данными. Алгоритм случайно выбирает подмножество точек, чтобы найти модель, которая согласуется с наибольшим числом точек (называемых инлайнерами).

#### Постановка задачи

Цель RANSAC — найти параметры $`R`$ и $`t`$, которые максимизируют число инлайнеров, то есть точек, для которых отклонение от модели меньше заданного порога $`\epsilon`$:

Целевая функция:

```math
\max_{R \in SO(3), t} \sum_{i=1}^N \mathbf{I} \left( \| R \mathbf{x}_i + t - \mathbf{y}_{\sigma(i)} \| \leq \epsilon \right),
```

где:

- $`\mathbf{I}(\cdot)`$ — индикаторная функция (возвращает 1, если условие выполняется, иначе 0),
- $`\epsilon`$ — порог расстояния для классификации точки как инлайнера.

#### Шаги алгоритма RANSAC

1. **Инициализация:**

   - Установить параметры алгоритма:
     - $`k`$ — максимальное число итераций,
     - $`\epsilon`$ — порог расстояния для определения инлайнера,
     - $`d`$ — минимальное количество инлайнеров для принятия модели.

2. **Случайный выбор:**

   - Выбрать случайное подмножество точек из $`X`$ и $`Y`$ (например, три пары точек для 3D).

3. **Оценка модели:**

   - На основе выбранных точек вычислить $`R`$ и $`t`$ с помощью метода минимизации наименьших квадратов.

4. **Подсчет инлайнеров:**

   - Вычислить число точек из $`X`$ и $`Y`$, для которых выполняется:

```math
     \| R \mathbf{x}_i + t - \mathbf{y}_i \| \leq \epsilon.
```

5. **Сравнение модели:**

   - Если текущая модель имеет больше инлайнеров, чем предыдущая лучшая модель, обновить её.

6. **Итерация:**

   - Повторить шаги 2–5 $`k`$ раз или до тех пор, пока не будет найдено достаточное число инлайнеров ($`\geq d`$).

7. **Построение окончательной модели:**
   - Использовать найденные инлайнеры для окончательной оценки $`R`$ и $`t`$ с минимизацией наименьших квадратов.

#### Преимущества RANSAC

1. **Устойчивость к выбросам:**
   - Алгоритм может успешно работать в условиях, когда выбросы составляют значительную часть данных.
2. **Гибкость:**

   - Применим к широкому кругу задач, не только к регистрации облаков точек (например, для подгонки прямых, плоскостей, кругов и т.д.).

3. **Простота реализации:**
   - Алгоритм основан на простых итеративных шагах.

#### Недостатки RANSAC

1. **Сложность выбора параметров:**

   - Значения $`k`$, $`\epsilon`$ и $`d`$ сильно влияют на производительность алгоритма и могут требовать экспериментов для настройки.

2. **Высокая вычислительная стоимость:**

   - Для больших облаков точек и большого количества выбросов число итераций может быть значительным.

3. **Отсутствие гарантий точности:**
   - Если количество выбросов слишком велико, RANSAC может не найти корректную модель.

#### Применение к облакам точек

Для регистрации облаков точек RANSAC часто используется в сочетании с другими алгоритмами, такими как Iterative Closest Point (ICP). Сначала RANSAC помогает отфильтровать выбросы и грубо оценить начальное преобразование, а затем ICP уточняет результат.

#### Вывод

RANSAC — это хороший метод для работы с зашумленными данными, особенно полезный при наличии большого числа выбросов. Однако для сложных задач он может потребовать значительных вычислительных ресурсов и тонкой настройки параметров.

---

### RANSAC с максимизацией $`\text{inlierRate}`$

**RANSAC (Random Sample Consensus)** — это устойчивый метод оценки параметров модели, работающий в условиях шума и выбросов. В данной версии алгоритма цель состоит в максимизации коэффициента инлайнеров ($`\text{inlierRate}`$), который определяется как доля точек, согласующихся с моделью.

#### Основная идея

Коэффициент инлайнеров ($`\text{inlierRate}`$) рассчитывается как отношение числа инлайнеров к общему числу точек:

```math
\text{inlierRate} = \frac{\text{countInliers(correspondences, T)}}{N},
```

где:

- $`\text{countInliers}`$ — функция, подсчитывающая количество точек, согласующихся с текущей моделью $`T`$ (трансформацией $`R`$ и $`t`$),
- $`N`$ — общее число точек,
- $`\text{correspondences}`$ — соответствия между точками двух облаков.

Цель состоит в нахождении модели $`T = \{ R, t \}`$, которая максимизирует $`\text{inlierRate}`$.

#### Постановка задачи

Целевая функция принимает вид:

```math
\max_{R \in SO(3), t} \text{inlierRate} = \frac{\sum_{i=1}^N \mathbf{I} \left( \| R \mathbf{x}_i + t - \mathbf{y}_{\sigma(i)} \| \leq \epsilon \right)}{N},
```

где:

- $`\mathbf{I}(\cdot)`$ — индикаторная функция, равная 1, если расстояние меньше порога $`\epsilon`$, и 0 в противном случае,
- $`\sigma(i)`$ — соответствие между точками $`\mathbf{x}_i \in X`$ и $`\mathbf{y}_i \in Y`$,
- $`\epsilon`$ — порог для классификации точки как инлайнера.

#### Шаги алгоритма

1. **Инициализация:**

   - Установить параметры:
     - $`k`$ — максимальное количество итераций,
     - $`\epsilon`$ — порог расстояния для определения инлайнеров,
     - $`\text{bestInlierRate}`$ — текущий максимальный коэффициент инлайнеров (инициализируется нулем).

2. **Случайный выбор:**

   - Случайно выбрать минимальное подмножество точек из $`X`$ и $`Y`$ (например, три пары точек для 3D).

3. **Оценка модели:**

   - На основе выбранных точек вычислить $`R`$ и $`t`$ методом минимизации квадратичной ошибки.

4. **Подсчет коэффициента инлайнеров:**

   - Для всех точек из $`X`$ и $`Y`$ определить инлайнеры:

```math
     \text{Inliers} = \left\{ i \mid \| R \mathbf{x}_i + t - \mathbf{y}_i \| \leq \epsilon \right\}.
```

- Вычислить текущий коэффициент инлайнеров:

```math
     \text{inlierRate} = \frac{|\text{Inliers}|}{N}.
```

5. **Сравнение модели:**

   - Если $`\text{inlierRate} > \text{bestInlierRate}`$, обновить текущую лучшую модель и сохранить коэффициент.

6. **Итерация:**

   - Повторить шаги 2–5 $`k`$ раз или до достижения заданного значения $`\text{bestInlierRate}`$ (например, $`\geq 0.9`$).

7. **Финальная модель:**
   - Использовать найденные инлайнеры для окончательной оценки $`R`$ и $`t`$ методом минимизации квадратичной ошибки.

#### Преимущества этой версии RANSAC

1. **Нормализованный критерий:**

   - Использование $`\text{inlierRate}`$ позволяет оценивать качество модели независимо от размера данных.

2. **Более устойчивое сравнение:**

   - Подсчет доли инлайнеров ($`\text{inlierRate}`$) помогает избежать перекоса модели при работе с большими облаками точек.

3. **Адаптивность:**

   - Легче задавать критерии остановки, используя порог для $`\text{inlierRate}`$, чем абсолютное количество инлайнеров.

4. **Возможность для улучшений**
   - Можно задать начальные соответствия между точками с помощью локальных дескрипторов и потом отбрасывать выбросы на основе $`\text{inlierRate}`$ на каждом шаге RANSAC. [Таблица описания дескрипторов](<https://robotica.unileon.es/index.php?title=PCL/OpenNI_tutorial_4:_3D_object_recognition_(descriptors)>)

#### Недостатки

1. **Чувствительность к выбору $`\epsilon`$:**

   - Неправильно выбранный порог расстояния может привести к неверной классификации инлайнеров.

2. **Сложность вычислений:**
   - Для каждого итерационного шага необходимо проверять все точки на соответствие модели.

#### Применение

Эта версия RANSAC часто используется для задач, где важно учитывать относительное количество инлайнеров. Например:

1. В задачах регистрации облаков точек с выбросами.
2. В ситуациях, когда абсолютное количество инлайнеров варьируется, но их доля относительно общего числа точек остается ключевым критерием.

---

### Глобальная регистрация облаков точек

Глобальная регистрация облаков точек — это задача нахождения жесткого преобразования между двумя облаками точек, когда начальное приближение отсутствует, или облака имеют значительные различия в начальных положениях. В отличие от локальных методов, таких как ICP, глобальная регистрация не требует близкого начального выравнивания, что делает её более универсальной, но также и вычислительно сложной.

#### Основные подходы к глобальной регистрации

##### 1. **Feature-Based Global Registration (на основе признаков)**

Этот подход основывается на выделении устойчивых локальных признаков в облаках точек и поиске соответствий между этими признаками.

- **Шаги:**

  1. Выделение признаков: Используются локальные дескрипторы (например, FPFH, SHOT, ISS) для описания окрестностей каждой точки. [Таблица описания дескрипторов](<https://robotica.unileon.es/index.php?title=PCL/OpenNI_tutorial_4:_3D_object_recognition_(descriptors)>)
  2. Поиск соответствий: Соответствия между точками двух облаков находятся на основе сходства дескрипторов.
  3. Оценка преобразования: Используется метод, такой как RANSAC, для удаления выбросов и оценки жесткого преобразования ($`R`$, $`t`$).

- **Преимущества:**

  - Устойчивость к шуму и выбросам.
  - Работает при значительных отличиях в положении облаков.

- **Недостатки:**
  - Зависимость от качества дескрипторов.
  - Может быть вычислительно дорого при большом количестве точек.

##### 2. **Globally Optimal Registration**

Этот подход использует методы оптимизации для глобального поиска преобразования, минимизирующего ошибку между облаками точек. Примером является использование глобально оптимальных методов, таких как Go-ICP или Branch-and-Bound.

- **Шаги:**

  1. Задание целевой функции: Обычно это минимизация квадратичной ошибки между соответствующими точками.
  2. Глобальный поиск: Перебираются все возможные преобразования в пространстве вращения ($`SO(3)`$) и смещения ($`\mathbb{R}^3`$).
  3. Определение глобального минимума.

- **Преимущества:**

  - Гарантированное нахождение глобального оптимума.
  - Подходит для сильно зашумленных данных и больших смещений.

- **Недостатки:**
  - Высокая вычислительная сложность.

##### 3. **Probabilistic Global Registration**

Этот подход основывается на вероятностных моделях для оценки преобразования. Примером является использование методов, таких как Coherent Point Drift (CPD) или Gaussian Mixture Models (GMM).

- **Шаги:**

  1. Построение вероятностной модели: Каждое облако точек представляется как распределение (например, смеси гауссиан).
  2. Оптимизация: Максимизируется правдоподобие модели с учетом предполагаемого преобразования.
  3. Регуляризация: Для ограничения решений добавляются регуляризирующие условия, такие как жесткость преобразования.

- **Преимущества:**

  - Подходит для облаков с отсутствующими или несовпадающими точками.
  - Естественно учитывает шум и выбросы.

- **Недостатки:**
  - Требует сложной настройки параметров.
  - Зависит от качества аппроксимации распределений.

##### 4. **Fast Global Registration (FGR)**

**Fast Global Registration (FGR)** — это метод, основанный на оптимизации функции энергии. Он использует быстрые численные методы для глобального поиска решения.

- **Основные этапы:**

  1. Вычисление грубых соответствий с использованием дескрипторов.
  2. Построение энергии на основе найденных соответствий.
  3. Быстрая оптимизация с использованием специальных методов, таких как градиентный спуск.

- **Преимущества:**

  - Высокая скорость.
  - Не требует значительных вычислительных ресурсов.

- **Недостатки:**
  - Зависимость от качества начального набора соответствий.

#### Преимущества глобальной регистрации

- **Отсутствие начальной инициализации:** Глобальные методы подходят для облаков точек с произвольным относительным положением.
- **Устойчивость к выбросам и зашумленным данным:** Многие методы, такие как RANSAC или вероятностные модели, естественно учитывают шум.
- **Применимость к различным задачам:** Методы глобальной регистрации используются в робототехнике, реконструкции объектов и медицинской визуализации.

#### Недостатки глобальной регистрации

1. **Высокая вычислительная сложность:** Глобальные методы требуют значительных вычислительных ресурсов.
2. **Зависимость от предварительной обработки:** Качество дескрипторов или вероятностной модели напрямую влияет на точность.
3. **Чувствительность к выбору параметров:** Многие методы требуют тонкой настройки.
