#          IMPORTS
# ====================================

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import seaborn as sns
import plotly.express as px
from random import randint
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score,accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import time # для фичей отслеживания прогресса (не задеплоен)
import warnings
warnings.filterwarnings("ignore")

#          MAIN PAGE
# ========================================

st.markdown("<h1 style='text-align: center;'>Применение методов машинного обучения в анализе банкротства</h1>", unsafe_allow_html=True)

components.html(
    """
        <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&width=435&lines=Анализ+банкротства+компании" alt="Typing SVG" /></a>
        <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&width=435&lines=методами+искуственного+интеллекта" alt="Typing SVG" /></a>
    """
)

with open("./img.png", "rb") as f:
    st.image(f.read(), use_column_width=True)

st.write(
    """
        # Краткое описание задачи
        Эффективное и заблаговременное прогнозирование банкротства компаний имеет важно значение для всех участников рынка. По мере развития информационного общества традиционные методы выявления банкротства становятся менее эффективными и более трудозатратными. Поэтому сочетание традиционных методов с современными моделями искусственного интеллекта может быть эффективно применено в современных экономических условиях.

       Основная цель работы - оценить риск банкротства с помощью нескольких алгоритмов машинного обучения, сравнить результаты их работы, определить наилучшую модель и соответствующий набор признаков для прогнозирования банкротства компаний.
    """
)

st.write("""# Этапы разработки""")

#image = Image.open("./stages.jpg")
#st.image(image, output_format="auto", use_column_width="auto")

with open("./stages.png", "rb") as f:
    st.image(f.read(), use_column_width=True)


with st.expander("Описание пайплайна работы", expanded=True):

    st.write(
            """
                ### Этапы разработки
<b><i>1. Поиск и сбор данных:</b></i>
Был использован датасет из Тайваньского экономического журнала за период с 1999 по 2009 год. Банкротство компании было определено на основании правил ведения бизнеса Тайваньской фондовой биржи. (<a href="https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction">Ссылка на данные</a>)

<b><i>2. Обработка (препроцессинг):</b></i>
Удаление ненужных колонок, one hot encoding категориальных переменных, заполнение пропущенных значений. С использованием библиотек pandas, numpy, seaborn.

<b><i>3. Анализ статистических показателей и визуализация:</b></i>
Инструменты для этого - с использованием библиотек pandas, seaborn.

<b><i>4. Выбор моделей, обучение и валидация модели с ними (без фичей):</b></i>
С использованием библиотек scikit-learn, pandas, seaborn.

<b><i>5. Выбор моделей, обучение и валидация модели с ними (с фичами):</b></i>
С использованием библиотек scikit-learn, pandas, seaborn.
<
b><i>6. Сравнение результатов:</b></i>
Анализ и графическое представление работы алгоритмов. При некорректной работе или плохим результатом проводится п. 4 и п. 5.

<b><i>7. Оформление микросервиса Streamlit:</b></i>
С использованием библиотеки streamlit.
            """,
            unsafe_allow_html=True
        )

with st.expander("Описание пайплайна работы", expanded=True):

    st.write(
            """
               ### Информация о применении методов машинного обучения для бизнес-задач:
Прогнозирование финансовой  неустойчивости – важный  компонент управления  компанией.  Из-за банкротства акционеры теряют доходы, бизнес, нарушаются цепочки поставок, фискальные органы лишаются налоговых поступлений, госорганы фиксируют снижение экономического роста и повышение социальной напряженности, а работники вынуждены искать новую работу. Поэтому получение ясной картины  финансового  и имущественного  состояния компаний уже  много  лет  является  целью специалистов самых разных областей знаний.
Компании активно внедряют современные технологии в основную деятельность, автоматизируя большое  количество  бизнес-процессов.  Это  позволяет  выстраивать,  например, электронную экспертизу,  c  помощью  которой  возможно  оперативно  объединять  множество  экспертов  из  разных предметных  областей  для  полного  охвата  рассматриваемой  проблемы  и  принятия  коллективного решения. Электронная экспертиза также подразумевает взаимодействие людей с интеллектуальными системами,  которые  способны  строить  анализ  и  прогнозы  на  основе  более  широкого  пространства переменных. Так, в частности, методы машинного обучения применяются уже в значительном количестве бизнес-задач, в том числе для интеллектуальной аналитики больших данных, которые компания аккумулирует для построения моделей прогнозирования.
Однако  риски  банкротства  могут  также  возникнуть  из-за некорректного  стратегического  менеджмента.  Такой  менеджмент  подразумевает  разработку долгосрочных  целей  и  действий,  которые  позволят  достичь  более  высоких  результатов  в  будущем, например, стать лидирующей компанией в своей отрасли. Разрабатываемые при этом стратегии обычно носят амбициозный характер, поэтому цели компании в таком случае не направлены на пролонгацию сложившейся  динамики.  Правильный  анализ  стратегической  ситуации  также  важен  для прогнозирования банкротства. Существует множество методов для ее оценки, в том числе и на основе анализа больших данных.<a href="http://infosoc.iis.ru/article/view/509"> Источник<a>
            """,
            unsafe_allow_html=True
        )

#             INFO
# =====================================

st.write(
    """
        # 1. Информация о датасете
<b><i>Похожие наборы данных:</i></b>
 - <a href="https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data">The Boston House-Price Data</a>
 - <a href="https://www.kaggle.com/datasets/fedesoriano/gender-pay-gap-dataset">Gender Pay Gap Dataset</a>
 - <a href="https://www.kaggle.com/datasets/fedesoriano/california-housing-prices-data-extra-features">Spanish Wine Quality Dataset</a>

<b><i>Про сами данные:</i></b>
Данные были получены из Тайваньского экономического журнала за период с 1999 по 2009 год. Банкротство компании было определено на основании правил ведения бизнеса Тайваньской фондовой биржи.

<i>P.S. Обновлены имена столбцов и описание, чтобы упростить понимание данных (Y = выходной объект, X = входной объект).</i>
    """,
        unsafe_allow_html=True
)

st.write(
    """
<b><i>Источник:</i></b>
Deron Liang and Chih-Fong Tsai, deronliang '@' gmail.com; cftsai '@' mgt.ncu.edu.tw, National Central University, Taiwan.
<a href="https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction">The data was obtained from UCI Machine Learning Repository.</a>

<b><i>Статья:</i></b>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0377221716000412">Тык</a>
    """,
        unsafe_allow_html=True
)

data = pd.read_csv("./dataset.csv", sep=",")

st.write(""" ### Таблица с данными: """, data)

st.write(
    """
        # 2. Обработка (препроцессинг)
    """
)

#       PREPROCESS
# ==================================

st.write(""" ### Статистика:""")
st.code(
    """
        data.describe()
    """
)
st.text(data.describe())
st.code(
    """
        data.shape
    """
)
st.write(""" #### Shape данных (номер строк и столбцов):""")
st.text(data.shape)

#st.table(data) - лучше не запускать :)

data.columns = [i.title().strip() for i in list(data.columns)]
row = data.shape[0]
col = data.shape[1]

null_values = data.isnull().sum().sort_values(ascending=False).head()
st.code(
    """
        null_values = data.isnull().sum().sort_values(ascending=False).head()
    """
)
st.write(null_values)

st.code(
    """
        data.info()
    """
)
st.text(data.info)
st.write("""Поскольку пропущенных значений нет, мы можем перейти к анализу данных.""")

#          VISUALIZATIONS
# ==================================

#with open("./plot_1.png", "rb") as f:
    #st.image(f.read(), use_column_width=True)

#values = st.sidebar.slider("Target", int(data["Bankrupt?"]))
#values = [0,1]
#values = list(data["Bankrupt?"].count())

colors = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
          'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
          'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',
          'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r',
          'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
          'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
          'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary',
          'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm',
          'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare',
          'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar',
          'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot',
          'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r',
          'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r',
          'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket',
          'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20',
          'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight',
          'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']

value = randint(0, len(colors)-1)

# plot_1
counts = data['Bankrupt?'].value_counts()
f = px.bar(counts, title="Соотношение количества банкротов и не банкротов")
f.update_xaxes(title="Bankrupt?")
f.update_yaxes(title="Count")
st.plotly_chart(f)
#f.show() - для отображения в отдельной вкладке

st.write(
    """
Записи кажутся сильно несбалансированными. Таким образом, необходимо рассмотреть возможность балансировки набора данных с помощью методов повышения или понижения дискретизации.
    """
)
numeric_features = data.dtypes[data.dtypes != 'int64'].index
categorical_features = data.dtypes[data.dtypes == 'int64'].index

data[categorical_features].columns.tolist()

st.write(
    """
        С помощью data.info() мы заметили, что у нас есть большинство данных «float64». Категориальные данные различаются как двоичные 1 и 0, поэтому сохраняются как «int64». Мы разделяем числовые и категориальные данные для анализа нашего набора данных.
    """
)
st.code(
    """
        numeric_features = data.dtypes[data.dtypes != 'int64'].index
        categorical_features = data.dtypes[data.dtypes == 'int64'].index

        data[categorical_features].columns.tolist()
    """,
    language="python"
)
st.write(
    """
        Вывод консоли:

        ['Bankrupt?', 'Liability-Assets Flag', 'Net Income Flag']
    """
)
st.write(
    """
        Есть только три столбца категорийных данных, сначала рассмотрим эти столбцы.
    """
)

# plot_2
counts = data["Liability-Assets Flag"].value_counts()
f = px.bar(counts, title="Обязательства-активы")
f.update_xaxes(title="Liability-Assets Flag")
f.update_yaxes(title="Count")
st.plotly_chart(f)

st.write(
    """
        Поле «Обязательства-Активы» (Liability-Assets Flag) обозначает статус организации, где, если общая сумма обязательств превышает общую сумму активов, помеченное значение будет равно 1, в противном случае значение равно 0. В большинстве случаев активы организаций/компаний превышают их обязательства.
    """
)

# plot_3

st.header("Распределение количества банкротов по активам и обязательствам")
counts = data[['Liability-Assets Flag','Bankrupt?']].value_counts()
plt.figure(figsize=(8,7))
fig, ax = plt.subplots()
ax = sns.countplot(x = 'Liability-Assets Flag',hue = 'Bankrupt?',data = data,palette = colors[value])
st.pyplot(fig)

st.write(
    """
        Небольшая часть организаций терпит банкротство, хотя у них активов больше, чем обязательств.
    """
)

# plot_4
counts = data["Net Income Flag"].value_counts()
f = px.bar(counts, title="Чистый доход")
f.update_xaxes(title="Net Income Flag")
f.update_yaxes(title="Count")
st.plotly_chart(f)

st.write(
    """
         Поле «Чистый доход» (Net Income Flag) обозначает состояние дохода организации за последние два года, где, если чистый доход отрицателен за последние два года, отмеченное значение будет равно 1, в противном случае значение равно 0. Мы наблюдаем, что все отчеты демонстрируют убыток в течение последних двух лет.
    """
)

# plot_5
st.header("Распределение количества банкротов по чистому доходу")
counts = data[['Net Income Flag','Bankrupt?']].value_counts()
plt.figure(figsize=(8,7))
fig, ax = plt.subplots()
ax = sns.countplot(x = 'Net Income Flag',hue = 'Bankrupt?',data = data,palette = colors[value])
st.pyplot(fig)

#counts = data[['Net Income Flag','Bankrupt?']].value_counts()
#f = sns.countplot(x = 'Net Income Flag',hue = 'Bankrupt?',data = data,palette = colors[value])
#st.plotly_chart(f)

st.write(
    """
        Многие организации, понесшие убытки за последние два года, стабилизировали свой бизнес, избежав таким образом банкротства.
    """
)
positive_corr = data[numeric_features].corrwith(data["Bankrupt?"]).sort_values(ascending=False)[:6].index.tolist()
negative_corr = data[numeric_features].corrwith(data["Bankrupt?"]).sort_values()[:6].index.tolist()

positive_corr = data[positive_corr + ["Bankrupt?"]].copy()
negative_corr = data[negative_corr + ["Bankrupt?"]].copy()

#x_value = positive_corr.columns.tolist()[-1]
#y_value = positive_corr.columns.tolist()[:-1]

#x_value = negative_corr.columns.tolist()[-1]
#y_value = negative_corr.columns.tolist()[:-1]

st.write(
    """
        Для простоты мы анализируем шесть основных атрибутов с положительной и отрицательной корреляцией.
    """
)

st.write("""Атрибуты с положительной корреляцией: """)
with open("./corr_1.png", "rb") as f:
    st.image(f.read(), use_column_width=True)

with st.expander("i - Что значит корреляция", expanded=False):
    st.write(
    """
        <b><i>Корреляция</i></b> – это взаимосвязь двух или нескольких случайных параметров. Когда одна величина растет или уменьшается, другая тоже изменяется.
    """,
        unsafe_allow_html=True
    )

st.write(
    """
        Мы видим, что три атрибута — Отношение долга % (Debt Ratio %), Текущая ответственность к активам (Current Liability To Assets), Текущая ответственность к текущим активам (Current Liability To Current Assets) обычно высоки в организациях-банкротах.
    """
)

st.write("""Атрибуты с отрицательной корреляцией: """)
with open("./corr_2.png", "rb") as f:
    st.image(f.read(), use_column_width=True)

st.write(
    """
        Эти атрибуты показывают нам, что чем больше активы и доходы компании, тем меньше вероятность того, что организация обанкротится.
Давайте проверим соотношение шести верхних положительных и отрицательных атрибутов корреляции между собой.
    """
)

with open("./positive.png", "rb") as f:
    st.image(f.read(), use_column_width=True)

st.write(
    """
        Существует положительная связь между атрибутами, которые имеют высокую корреляцию с целевой переменной.
    """
)

with open("./negative.png", "rb") as f:
    st.image(f.read(), use_column_width=True)

st.write(
    """
        Существует положительная связь между атрибутами, которые имеют низкую корреляцию с целевой переменной.
    """
)

st.write(""" ## Нажав на кнопку ниже - можно построить интерактивную корреляционную матрицу""")

if st.button("Построить корреляционную матрицу !!!"):
    st.header("Корреляционная матрица")
    relation = positive_corr.columns.tolist()[:-1] + negative_corr.columns.tolist()[:-1]
    plt.figure(figsize=(8,7))
    fig, ax = plt.subplots()
    ax = sns.heatmap(data[relation].corr(),annot=True)
    st.pyplot(fig)

st.write(
    """
        Общая корреляция 12 лучших атрибутов приведена выше.
    """
)

st.write(
    """
        ### Резюме анализа
- Количество организаций, обанкротившихся за 10 лет с 1999 по 2000 год, невелико.
- Несколько компаний обладают большим количеством активов, что всегда является хорошим признаком для организации.
- Организация не может гарантировать, что не будет банкротом, хотя и владеет несколькими активами.
- Организации в наборе данных несут убытки за последние два года, поскольку их чистая прибыль представляется отрицательной.
- Очень немногие из организаций, имевших отрицательную прибыль за последние два года, терпят банкротство.
- Отмечено, что атрибуты «Отношение долга, %, текущие обязательства к активам, текущие обязательства к текущим активам» — это лишь некоторые из атрибутов, которые имеют высокую корреляцию с целевой переменной.
- Увеличение значений атрибутов «Отношение долга %, Текущие обязательства к активам, Текущие обязательства к оборотным средствам» приводит к большим убыткам организации, что приводит к банкротству.
- Увеличение значений признаков, имеющих отрицательную корреляцию с целевой переменной, помогает организации избежать банкротства.
- По-видимому, существует связь между атрибутами, имеющими высокую и низкую корреляцию с целевой переменной.
- Мы наблюдали несколько корреляций между 12 основными атрибутами, одним из которых является «Чистая стоимость / Активы и соотношение долга%», которые отрицательно коррелируют друг с другом.
    """
)

#                       ML
# ===========================================================

st.write("""# Машинное обучение""")
st.write("""### Нормализация данных""")

st.code(
    """
        numeric_features = data.dtypes[data.dtypes != 'int64'].index
        data[numeric_features] = data[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

        data[numeric_features] = data[numeric_features].fillna(0)
    """,
    language="python"
)

with st.expander("i - Что значит нормализация", expanded=False):
    st.write(
    """
        <b><i>Нормализация</i></b> – это процедура предобработки входной информации (обучающих, тестовых и валидационных выборок, а также реальных данных), при которой значения признаков во входном векторе приводятся к некоторому заданному диапазону, например, [0…1] или [-1…1] [1]
    """,
        unsafe_allow_html=True
    )

st.write(
    """
        Наш набор данных сильно несбалансирован. Таким образом, перед обучением модели нам нужно как то преобразовать эти данные. Давайте обозначим несколько этапов, которым мы должны следовать, когда сталкиваемся с несбалансированным набором данных:

- Деление набора данных на части для обучения и тестирования (80–20%). Мы сохраняем 20% в тестовый набор для окончательной оценки.
- С помощью кросс-валидации по К блокам (stratified K-fold cross validation) мы распределим 80% тренировочного набора на дальнейшее обучение и тестирование.
- Поскольку мы имеем дело с более чем 50 функциями, будем использовать Randomized Search Cross-Validation, поскольку этот метод лучше работает со многими функциями.
    """
)

#          MODELS SCORES DISPLAY FUNC WITHOUT FEATURE SELECTION
# ============================================================================

Models = pd.DataFrame(columns=['Algorithm','Model Score','Precision','Recall','F1 score','ROC-AUC score'])

def taining_without_feature_selection(Parameters, Model, Dataframe, Modelname):

    data = Dataframe.copy()

    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']

    #Traditional split of the dataset 80% - 20%
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_train, x_test, y_train, y_test = x_train.values, x_test.values, y_train.values, y_test.values

    #Proportional split of 80% data with respect to the class of the target feature ie. [1,0]
    sf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in sf.split(x_train, y_train):
        sf_x_train, sf_x_test = X.iloc[train_index], X.iloc[test_index]
        sf_y_train, sf_y_test = y.iloc[train_index], y.iloc[test_index]

    sf_x_train, sf_x_test, sf_y_train, sf_y_test = sf_x_train.values, sf_x_test.values, sf_y_train.values, sf_y_test.values

    model_parameter_sm = Parameters

    rand_model = RandomizedSearchCV(Model, model_parameter_sm, n_iter=4)

    #Identifying the best parameters through RandomizedSearchCV()
    for train, test in sf.split(sf_x_train, sf_y_train):
        pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_model)
        fitting_model = pipeline.fit(sf_x_train[train], sf_y_train[train])
        best_model = rand_model.best_estimator_

    #Evaluation with against 20% unseen testing data
    print()
    print("Evaluation Of Models")

    sm = SMOTE(sampling_strategy='minority', random_state=42)
    Xsm_train, ysm_train = sm.fit_resample(sf_x_train, sf_y_train)

    print()
    print("Random Model Evaluation")

    final_model_sm = rand_model.best_estimator_
    final_model_sm.fit(Xsm_train, ysm_train)

    prediction = final_model_sm.predict(x_test)

    print(classification_report(y_test, prediction))

    model = {}

    model['Algorithm'] = Modelname
    model['Model Score'] = str(round((accuracy_score(y_test, prediction)*100),2)) + "%"
    model['Precision'] = round(precision_score(y_test, prediction),2)
    model['Recall'] = round(recall_score(y_test, prediction),2)
    model['F1 score'] = round(f1_score(y_test, prediction),2)
    model['ROC-AUC score'] = round(roc_auc_score(y_test, prediction),2)

    return model

#                     SELECT OPTIONS
# ==========================================================================
st.write("""### Машинное обучение без отбора признаков""")

option = st.selectbox(
     "Какой алгоритм для обучения выберем?",
    ("K Nearest Neighbour", "Logistic Regression", "DecisionTree Classifier", "Random Forest Classifier", "Support Vector Classifier")
)

st.write('Выбрано:', option)
st.write(
    """
        После выбора нужно подождать, пока пройдет обучение - у некоторых алгоритмов процесс может растянуться на продолжительное время...
        Для быстрой проверки можно использовать K Nearest Neighbour и Logistic Regression
    """
)

with st.expander("О алгоритмах", expanded=False):
    st.write(
    """
        сюда описание про алгоритмы
    """,
        unsafe_allow_html=True
    )

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

@st.cache
def save_results(Models):
    Models = Models.append(TrainedModel,ignore_index=True)
    return Models

if option == "K Nearest Neighbour":
    #print("K Nearest Neighbour")
    TrainedModel = taining_without_feature_selection({"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}, KNeighborsClassifier(), data,"K Nearest Neighbour")
    save_results(Models)
    st.write(""" ### Результаты работы алгоритма: """, Models)
    csv = convert_df(Models)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='score_nneighbors.csv',
        mime='text/csv',
    )

if option == "Logistic Regression":
    #print("Logistic Regression")
    TrainedModel = taining_without_feature_selection({"penalty": ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}, LogisticRegression(solver='liblinear'), data, "Logistic Regression")
    save_results(Models)
    st.write(""" ### Результаты работы алгоритма: """, Models)
    csv = convert_df(Models)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='score_logisticregression.csv',
        mime='text/csv',
    )

if option == "DecisionTree Classifier":
    #print("DecisionTree Classifier")
    TrainedModel = taining_without_feature_selection({"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)),"min_samples_leaf": list(range(5,7,1))}, DecisionTreeClassifier(), data, "DecisionTree Classifier")
    save_results(Models)
    st.write(""" ### Результаты работы алгоритма: """, Models)
    csv = convert_df(Models)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='score_decisiontree.csv',
        mime='text/csv',
    )

if option == "Random Forest Classifier":
    #print("Random Forest Classifier")
    TrainedModel = taining_without_feature_selection({"max_depth": [3, 5, 10, None],"n_estimators": [100, 200, 300, 400, 500]},  RandomForestClassifier(), data, "Random Forest Classifier")
    save_results(Models)
    st.write(""" ### Результаты работы алгоритма: """, Models)
    csv = convert_df(Models)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='score_randomforest.csv',
        mime='text/csv',
    )

if option == "Support Vector Classifier":
    print("Support Vector Classifier")
    TrainedModel = taining_without_feature_selection({'C': [1,10,20],'kernel': ['rbf','linear']},  SVC(), data, "Support Vector Classifier")
    save_results(Models)
    st.write(""" ### Результаты работы алгоритма: """, Models)
    csv = convert_df(Models)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='score_supportvector.csv',
        mime='text/csv',
    )

st.write("### Общая таблица работы моделей: ")
if st.button("Отобразить (тыкать после обучения интересующих алгоритмов)"):
    # st.write(Models.sort_values('F1 score',ascending=False))
    st.write(Models)


#            MODELS SCORES DISPLAY FUNC WITH FEATURE SELECTION
# ================================================================================

Models_2 = pd.DataFrame(columns=['Algorithm','Model Score','Precision','Recall','F1 score','ROC-AUC score'])

@st.cache
def taining_with_feature_selection(Parameters, Model, Dataframe, Modelname):

    data = Dataframe.copy()

    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']

    '''
    Feature Selection Process:
    class sklearn.feature_selection.SelectKBest(score_func=<function>, k=<number of features>
        score_func - Scoring measure
        k - Total features to be returned
    '''

    fs = SelectKBest(score_func=f_classif, k=int((data.shape[1]*85)/100))

    X = fs.fit_transform(X, y)

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_train, x_test, y_train, y_test = x_train.values, x_test.values, y_train.values, y_test.values

    sf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in sf.split(x_train, y_train):
        sf_x_train, sf_x_test = X.iloc[train_index], X.iloc[test_index]
        sf_y_train, sf_y_test = y.iloc[train_index], y.iloc[test_index]

    sf_x_train, sf_x_test, sf_y_train, sf_y_test = sf_x_train.values, sf_x_test.values, sf_y_train.values, sf_y_test.values

    model_parameter_sm = Parameters

    rand_model = RandomizedSearchCV(Model, model_parameter_sm, n_iter=4)

    for train, test in sf.split(sf_x_train, sf_y_train):
        pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_model)
        fitting_model = pipeline.fit(sf_x_train[train], sf_y_train[train])
        best_model = rand_model.best_estimator_

    print()
    print("Evaluation Of Models")

    sm = SMOTE(sampling_strategy='minority', random_state=42)
    Xsm_train, ysm_train = sm.fit_resample(sf_x_train, sf_y_train)

    print()
    print("Random Model Evaluation")

    final_model_sm = rand_model.best_estimator_
    final_model_sm.fit(Xsm_train, ysm_train)

    prediction = final_model_sm.predict(x_test)

    print(classification_report(y_test, prediction))

    model = {}

    model['Algorithm'] = Modelname
    model['Model Score'] = str(round((accuracy_score(y_test, prediction)*100),2)) + "%"
    model['Precision'] = round(precision_score(y_test, prediction),2)
    model['Recall'] = round(recall_score(y_test, prediction),2)
    model['F1 score'] = round(f1_score(y_test, prediction),2)
    model['ROC-AUC score'] = round(roc_auc_score(y_test, prediction),2)

    return model


#                     SELECT OPTIONS
# ==========================================================================
with open("./polosca.jpg", "rb") as f:
    st.image(f.read(), use_column_width=True)

st.write("""### Машинное обучение с отбором признаков""")

option = st.selectbox(
     "Какой алгоритм для обучения с отбором признаков выберем?",
    ("K Nearest Neighbour", "Logistic Regression", "DecisionTree Classifier", "Random Forest Classifier", "Support Vector Classifier")
)

st.write('Выбрано:', option)
st.write(
    """
        После выбора нужно подождать, пока пройдет обучение - у некоторых алгоритмов процесс может растянуться на продолжительное время...
        Для быстрой проверки можно использовать K Nearest Neighbour и Logistic Regression
    """
)

@st.cache
def save_results(Models_2):
    Models_2 = Models_2.append(TrainedModel,ignore_index=True)
    return Models_2

if option == "K Nearest Neighbour":
    #print("K Nearest Neighbour")
    TrainedModel = taining_without_feature_selection({"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}, KNeighborsClassifier(), data,"K Nearest Neighbour")
    save_results(Models_2)
    st.write(""" ### Результаты работы алгоритма: """, Models_2)
    csv = convert_df(Models_2)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='score_nneighbors_fs.csv',
        mime='text/csv',
    )

if option == "Logistic Regression":
    #print("Logistic Regression")
    TrainedModel = taining_without_feature_selection({"penalty": ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}, LogisticRegression(solver='liblinear'), data, "Logistic Regression")
    save_results(Models_2)
    st.write(""" ### Результаты работы алгоритма: """, Models_2)
    csv = convert_df(Models_2)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='score_logisticregression_fs.csv',
        mime='text/csv',
    )

if option == "DecisionTree Classifier":
    #print("DecisionTree Classifier")
    TrainedModel = taining_without_feature_selection({"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)),"min_samples_leaf": list(range(5,7,1))}, DecisionTreeClassifier(), data, "DecisionTree Classifier")
    save_results(Models_2)
    st.write(""" ### Результаты работы алгоритма: """, Models_2)
    csv = convert_df(Models_2)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='score_decisiontree_fs.csv',
        mime='text/csv',
    )

if option == "Random Forest Classifier":
    #print("Random Forest Classifier")
    TrainedModel = taining_without_feature_selection({"max_depth": [3, 5, 10, None],"n_estimators": [100, 200, 300, 400, 500]},  RandomForestClassifier(), data, "Random Forest Classifier")
    save_results(Models_2)
    st.write(""" ### Результаты работы алгоритма: """, Models_2)
    csv = convert_df(Models_2)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='score_randomforest_fs.csv',
        mime='text/csv',
    )

if option == "Support Vector Classifier":
    print("Support Vector Classifier")
    TrainedModel = taining_without_feature_selection({'C': [1,10,20],'kernel': ['rbf','linear']},  SVC(), data, "Support Vector Classifier")
    save_results(Models_2)
    st.write(""" ### Результаты работы алгоритма: """, Models_2)
    csv = convert_df(Models_2)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='score_supportvector_fs.csv',
        mime='text/csv',
    )

st.write("### Общая таблица работы моделей: ")
if st.button("Отобразить (тыкать после обучения интересующих алгоритмов с отбором признаков)"):
    # st.write(Models.sort_values('F1 score',ascending=False))
    st.write(Models_2)

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")

if st.button("✨Получи приз, если дошел до самого конца!!"):
    st.balloons()

st.markdown(" ")

components.html(
    """
        <p align="center">Powered by <a href="https://github.com/Lyutikk">Gforce</a></p>
    """
)
