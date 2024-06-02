
# Инструкция по использованию ноутбука

Этот [ноутбук](https://colab.research.google.com/drive/1qjPHxCu_VB1XaQqnCjlZig0XFX61Q_I2?usp=sharing) предназначен для анализа данных из arXiv с использованием модели Top2Vec, визуализации результатов и создания интерфейса с помощью Gradio. Следуйте инструкциям ниже, чтобы использовать функционал ноутбука.

## Установка зависимостей

Для работы ноутбука необходимо установить несколько пакетов. В начале ноутбука уже есть команды для установки всех необходимых зависимостей:

```python
!pip install arxivscraper top2vec top2vec[sentence_encoders] tensorflow==2.8.0 tensorflow-probability==0.16.0 gradio
```

## Работа с данными

### 1. Создание нового датасета

Для создания нового датасета из статей arXiv выполните следующие шаги:

- Раскомментируйте и выполните блоки кода для скачивания и обработки данных:
  ```python
  import arxivscraper
  import pandas as pd

  def scrape_and_save(category, start_year, end_year):
      """Сбор данных arXiv для заданной категории и диапазона лет."""
      for year in range(start_year, end_year):
          scraper = arxivscraper.Scraper(category=category,
                                         date_from=f'{year}-01-01',
                                         date_until=f'{year+1}-01-01')
          df = pd.DataFrame(scraper.scrape())
          df.to_csv(f'arxiv_{category}_{year}.csv', index=False)
          print(f'Data for {year} saved.')

  def combine_and_process(file_names):
      """Объединение нескольких CSV-файлов в один DataFrame и обработка данных."""
      df_list = []
      for file_name in file_names:
          df_temp = pd.read_csv(file_name, dtype={'id': str}, low_memory=False)
          text_columns = ['title', 'abstract', 'categories', 'doi', 'authors', 'url']
          df_temp[text_columns] = df_temp[text_columns].astype(str)
          date_columns = ['created', 'updated']
          df_temp[date_columns] = pd.to_datetime(df_temp[date_columns], errors='coerce')
          df_list.append(df_temp)
      df_combined = pd.concat(df_list, ignore_index=True)
      df_combined[text_columns] = df_combined[text_columns].fillna('None')
      return df_combined

  scrape_and_save(category='cs', start_year=2010, end_year=2024)
  file_names = [f'arxiv_cs_{year}.csv' for year in range(2010, 2024)]
  df_combined = combine_and_process(file_names)

  try:
      df_combined.to_parquet('combined_data.parquet', index=False)
      print("File successfully saved in Parquet format.")
  except Exception as e:
      print(f"Error saving file: {e}")
  ```

### 2. Использование существующего датасета

Если вы хотите использовать существующий датасет, скачайте его с помощью команды:

```python
!wget https://huggingface.co/datasets/CCRss/arxiv_papers_cs/resolve/main/arxiv_cs_from2010to2024-01-01.parquet
```

Для чтения данных используйте следующий код:

```python
import pandas as pd

file_name = '/content/arxiv_cs_from2010to2024-01-01.parquet'  # путь к файлу
df = pd.read_parquet(file_name)

print(df.shape)
df.head()
```

## Обучение модели Top2Vec

### Загрузка обученной модели

Для экономии времени можно загрузить уже обученную модель с Hugging Face:

```python
!wget https://huggingface.co/CCRss/topic_modeling_top2vec_scientific-texts/resolve/main/top2vec_model_arxiv_cs_from2010to2024-01-01
```

### Инициализация модели

Загрузите модель и присвойте ей документы:

```python
from top2vec import Top2Vec
model = Top2Vec.load("/content/top2vec_model_arxiv_cs_from2010to2024-01-01")
```

## Работа с функциями Top2Vec

### Получение информации о темах

Используйте следующий код для получения информации о темах и представлении документов:

```python
topic_sizes, topic_nums = model.get_topic_sizes()
data = []

for topic_num in topic_nums:
    _, _, document_ids = model.search_documents_by_topic(topic_num=topic_num, num_docs=topic_sizes[topic_num])
    for doc_id in document_ids:
        data.append({'document_id': doc_id, 'topic_num': topic_num})

df_new = pd.DataFrame(data)
df_new = df_new.sort_values(by='document_id').reset_index(drop=True)
df['topic_num'] = df_new['topic_num']

topic_words, word_scores, topic_nums = model.get_topics()
def create_topic_representation(words, scores):
    return ', '.join(words[:5])

topic_representations = [create_topic_representation(words, scores) for words, scores in zip(topic_words, word_scores)]
topic_representation_dict = dict(zip(topic_nums, topic_representations))
df['topic_representation'] = df['topic_num'].map(topic_representation_dict)
```

## Визуализация

Для визуализации трендов тем и групп используйте следующий код:

```python
import plotly.graph_objects as go

def visualize_topic_trend_plotly(topic_num, topic_group):
    publications = publications_per_topic_year.loc[topic_num]
    changes = growth_per_topic.loc[topic_num]
    relative_growth = relative_growth_per_topic.loc[topic_num] * 100
    relative_growth = relative_growth.fillna(0)
    years = publications.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=publications, mode='lines+markers', name='Количество публикаций', marker=dict(size=8, color='blue')))
    fig.add_trace(go.Bar(x=years, y=changes, name='Изменение количества публикаций', marker_color='orange', opacity=0.6))
    fig.add_trace(go.Scatter(x=years, y=relative_growth, mode='lines', name='Относительный рост (%)', yaxis='y2', line=dict(color='green', width=2, dash='dash')))

    fig.update_layout(
        title=f'Анализ трендов для темы {topic_num} ({topic_group})',
        xaxis_title='Год',
        yaxis_title='Количество публикаций',
        yaxis2=dict(title='Относительный рост (%)', overlaying='y', side='right', range=[-100, 100]),
        legend=dict(x=1.05, y=1, traceorder='reversed', font_size=16),
        barmode='overlay',
        template='plotly_white'
    )

    fig.show()

topic_num_to_analyze = 92
topic_group = df_filtered[df_filtered['topic_num'] == topic_num_to_analyze]['topic_group'].iloc[0]
visualize_topic_trend_plotly(topic_num_to_analyze, topic_group)
```

## Интерфейс Gradio

Для создания веб-интерфейса используйте следующий код:

```python
import gradio as gr

def get_info(input_value):
    try:
        topic_num = int(input_value)
        html_table, plot = get_topic_analysis(topic_num)
    except ValueError:
        html_table, plot = get_group_analysis(input_value)
    return html_table, plot

iface = gr.Interface(
    fn=get_info,
    inputs=gr.Textbox(label="Номер темы или название тематической группы"),
    outputs=[
        gr.HTML(label="Информация"),
        gr.Plot(label="Анализ тренда")
    ],
    title="Анализ тем и тематических групп",
    description="Введите номер темы или название тематической группы для получения информации."
)

iface.launch(debug=True, share=True)
```
