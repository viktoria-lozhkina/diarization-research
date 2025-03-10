# diarization-research

## Модели для диаризации
### 1. PyAnnotate <br>
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) <br>

<details>
  <summary>Подробно о модели.</summary>
  
  1. Разделяет аудиозапись на сегменты, где активен каждый говорящий. <br>
  2. Присваивает каждому сегменту уникальный идентификатор говорящего. <br>

  Модель умеет работать с перекрывающейся речью, многоканальными аудио, возможна работа в реальном времени.<br>
  
  **Технические детали**<br>
  - Модель: pyannote/speaker-diarization-3.1.<br>

  - Фреймворк: PyTorch.<br>
  
  - Предобученные веса: Доступны через Hugging Face Model Hub. (Необходимо проверить условия лицензии для коммерческого использования.) <br>
  
  - Языки: Многоязычная поддержка, но с акцентом на английский язык.<br>
  
  - Требования к оборудованию:<br>
  
    - Минимум: CPU (рекомендуется многоядерный процессор).<br>
    
    - Оптимально: GPU (например, NVIDIA с поддержкой CUDA).<br>
</details>

<details>
  <summary>Конфигурация для пайплайна.</summary>
 
```
  version: 3.1.0
  pipeline:
    name: pyannote.audio.pipelines.SpeakerDiarization
    params:
      clustering: AgglomerativeClustering
      embedding: pyannote/wespeaker-voxceleb-resnet34-LM
      embedding_batch_size: 32
      embedding_exclude_overlap: true
      segmentation: pyannote/segmentation-3.0
      segmentation_batch_size: 32
  params:
    clustering:
      method: centroid
      min_cluster_size: 12  # Минимальное количество сегментов, необходимых для формирования кластера.
      threshold: 0.7045654963945799   # Пороговое значение для определения, насколько близко должны быть сегменты, чтобы быть объединенными в один кластер.
    segmentation:
      min_duration_off: 0.0  # Паузы не учитываются.
```
1) **Сегментация:** Аудио разделяется на сегменты, где активен каждый говорящий.

2) **Извлечение эмбеддингов:** Для каждого сегмента извлекаются векторные представления (эмбеддинги) с использованием модели wespeaker-voxceleb-resnet34-LM.

3) **Кластеризация:** Сегменты группируются по говорящим с использованием агломеративной кластеризации.
</details>

<details>
  <summary>Выполнение диаризации с помощью модели.</summary>
  
  ```
  from pyannote.audio import Pipeline, Audio
  import torch
  
  
  class EndpointHandler:
      def __init__(self, path=""):
          # initialize pretrained pipeline
          self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
  
          # send pipeline to GPU if available
          if torch.cuda.is_available():
              self._pipeline.to(torch.device("cuda"))
  
          # initialize audio reader
          self._io = Audio()
  
      def __call__(self, data):
          inputs = data.pop("inputs", data)
          waveform, sample_rate = self._io(inputs)
  
          parameters = data.pop("parameters", dict())
          diarization = self.pipeline(
              {"waveform": waveform, "sample_rate": sample_rate}, **parameters
          )
  
          processed_diarization = [
              {
                  "speaker": speaker,
                  "start": f"{turn.start:.3f}",
                  "end": f"{turn.end:.3f}",
              }
              for turn, _, speaker in diarization.itertracks(yield_label=True)
          ]
          return {"diarization": processed_diarization}
```
Этот код:

1. Загружает предобученный пайплайн для диаризации.

2. Читает аудиофайл и преобразует его в waveform.

3. Применяет пайплайн для разделения аудио на сегменты с идентификацией говорящих.

4. Возвращает результат.
</details>

<details>
  <summary>Формат результата.</summary>
  
  ```
  {
    "diarization": [
      {"speaker": "SPEAKER_01", "start": "0.000", "end": "2.345"},
      {"speaker": "SPEAKER_02", "start": "2.346", "end": "5.678"},
      ...
    ]
  }
  ```
</details>

<details>
  <summary>Fine-tuned версии от пользователей HaggingFace</summary>

  |Модель|Результаты|Гиперпараметры|Данные для дообучения|
  |-|--------|---|--|
  |[JSWOOK/pyannote_3_fine_tuning](https://huggingface.co/JSWOOK/pyannote_3_fine_tuning)|Loss: 0.3134 <br> Model Preparation Time: 0.0048<br> Der: 0.0888<br> False Alarm: 0.0134<br> Missed Detection: 0.0337<br> Confusion: 0.0417|learning_rate: 5e-05<br> train_batch_size: 32<br> eval_batch_size: 32<br> seed: 42<br> optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08<br> lr_scheduler_type: cosine<br> num_epochs: 10|[diarizers-community/voxconverse dataset](https://huggingface.co/datasets/diarizers-community/voxconverse)|
  |[nairaxo/diarization-model-ary](https://huggingface.co/nairaxo/diarization-model-ary) <br><br> [статья автора о диаризации](https://arxiv.org/pdf/2412.12143) > новый подход в идентификации **одного** говорящего в реальном времени, комбинация офлайн и онлайн методов|Loss: 0.3412<br> Der: 0.1116<br> False Alarm: 0.0194<br> Missed Detection: 0.0267<br> Confusion: 0.0655|learning_rate: 0.001<br> train_batch_size: 32<br> eval_batch_size: 32<br> seed: 42<br> optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments<br> lr_scheduler_type: cosine<br> num_epochs: 5|[talkbank/callhome](https://huggingface.co/datasets/talkbank/callhome), спонтанные телефонные разговоры, языки: англ., кит., япон., нем., испанский|
  |||||
  
</details>

<details>
  <summary>Другие полезные модели от PyAnnotate</summary>

  - Для сегментации: [pyannote/segmentation-3.0](https://huggingface.1319lm.top/pyannote/segmentation-3.0) (описана ниже)
  - Для эмбеддинга: [pyannote/embedding](https://huggingface.1319lm.top/pyannote/embedding)
</details>

### 2. NVIDIA
[nvidia/diar_sortformer_4spk-v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) <br>
<details>
  <summary>Подробно о модели.</summary>
  Предобученная модель для диаризции от NVIDIA. Поддерживает распознавание до 4 спикеров.

  - **Архитектура:** Sortformer ([Статья "Sortformer: Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens"](https://arxiv.org/pdf/2409.06656)) — новая архитектура для диаризации, основана на трансформерах + механизм self-attention. Новизна заключается в сортировке и перестановке элементов последовательности.

*Модель динамически перестраивает входные данные (например, спектрограммы) в порядке, который лучше подходит для разделения звука. Это позволяет модели более эффективно выделять целевые источники звука.*

  - На вход подается моноканальное аудио (обычно .wav с ЧД 16 кГц).
  - На выходе получаем матрицу TхS:
    - S — максимальное количество говорящих. T — общее количество фреймов, включая заполненные нулями. Каждый фрейм соответствует сегменту аудио длительностью 0,08 секунды.
Каждый элемент матрицы T x S представляет вероятность активности говорящего в диапазоне [0, 1]. Например, элемент матрицы a(150, 2) = 0,95 указывает на 95% вероятность активности второго говорящего в течение временного диапазона [12,00, 12,08] секунд.

  **Ограничения:**

  - Оптимизирована на распознавание до 4 говорящих. Если спикеров больше — может снизиться качество.
  - Работает в офлайн-режиме (The model operates in a non-streaming mode (offline mode)).
  - Максимальная продолжительность тестовой записи зависит от доступной памяти GPU. Для модели RTX A6000 48GB предел составляет около 12 минут.
  - Модель была обучена на общедоступных наборах речевых данных, в основном на английском языке. В результате: а) Производительность может ухудшиться на русской речи. и б) Производительность также может ухудшиться на данных, не похожих на обучающие, например записи в шумных условиях.
</details>

<details>
  <summary>Датасеты.</summary>
  
  Sortformer был обучен на комбинации 2030 часов реальных разговоров и 5150 часов или имитированных аудиосмесей, созданных симулятором речевых данных NeMo.

  Training Datasets (Real conversations)
  
  - Fisher English (LDC)
  - 2004-2010 NIST Speaker Recognition Evaluation (LDC)
  - Librispeech
  - AMI Meeting Corpus
  - VoxConverse-v0.3
  - ICSI
  - AISHELL-4
  - Third DIHARD Challenge Development (LDC)
  - 2000 NIST Speaker Recognition Evaluation, split1 (LDC)

Training Datasets (Used to simulate audio mixtures)

- 2004-2010 NIST Speaker Recognition Evaluation (LDC)
- Librispeech
</details>

<details>
  <summary>Как обучалась модель.</summary>
  
  - Обучалась на 8 nodes of 8×NVIDIA Tesla V100 GPUs.
  - Использовались 90 секундные обучающие образцы.
  - batch size: 4.
  - [Пример скрипта для обучения](https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_tasks/diarization/neural_diarizer/sortformer_diar_train.py).
  - [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_tasks/diarization/conf/neural_diarizer/sortformer_diarizer_hybrid_loss_4spk-v1.yaml).
</details>

[Репозиторий от NVIDIA NeMo на Github](https://github.com/NVIDIA/NeMo/tree/main/examples/speaker_tasks/diarization) , посвящен задачам диаризации.

### 3. SpeechBrain
[speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) <br>
<details>
  <summary>Подробно о модели.</summary>
    
  **Внимание:** модель предназначется для верификации говорящих, но её можно адаптировать для диаризации (возможно, добавив этап кластеризации).
  <details>
  <summary>Возможный код для адаптации.</summary>
  
  ```
  # Извлечение эмбеддингов
  from speechbrain.pretrained import EncoderClassifier
  classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
  embeddings = classifier.encode_batch("audio.wav")

  # Кластеризация эмбеддингов
  from sklearn.cluster import KMeans # алгоритм k-means для кластеризации данных
  kmeans = KMeans(n_clusters=num_speakers)
  labels = kmeans.fit_predict(embeddings)
  ```

</details>
   
  **Архитектура:** ECAPA-TDNN (Enhanced CNN-Augmented TDNN) — современная архитектура для извлечения эмбеддингов.
   
  * Сверточные слои (CNN): Для извлечения локальных признаков.
  * Механизмы внимания (Attention): Для учета глобальных зависимостей.
  * Канальное внимание (Channel Attention): Для улучшения качества эмбеддингов.

**Входные данные**

* Аудио: Модель принимает на вход аудиосигнал (например, в формате .wav).
* Частота дискретизации: 16 кГц.

**Выходные данные**

* Эмбеддинги: Модель возвращает векторные представления размерностью 192.

</details>

<details>
  <summary>Данные для обучения.</summary>
  
  * Модель обучена на датасете VoxCeleb (Voxceleb 1+ Voxceleb2), который содержит более 100 000 записей речи от более чем 1 000 говорящих.
  * VoxCeleb включает записи из интервью, ток-шоу и других публичных выступлений.

</details>


[speechbrain/emotion-diarization-wavlm-large](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) <br>
<details>
  <summary>Подробно о модели.</summary>

  **Внимание:** данная модель обучена для диаризации **эмоций,** а не спикеров. Но можно глянуть. Статья про модель [SPEECH EMOTION DIARIZATION: WHICH EMOTION APPEARS WHEN?](https://arxiv.org/pdf/2306.12991)

  * Обучена для распознавания базовых эмоций (happy, sad, angry, neutral)
  * Собран отдельный датасет the Zaion Emotion Dataset (ZED).
    * Он содержит 180 высказываний длительностью от 1 до 15 секунд и охватывает 73 спикера разного возраста и пола.
  * EDER(Emotion Diarization Error Rate) = 29.7.
</details>

## Опенсорсные инструменты
### 4. The LIUM SpkDiarization toolkit
Статья ["LIUM SPKDIARIZATION: AN OPEN SOURCE TOOLKIT FOR DIARIZATION"](https://hal.science/hal-01433518/document) <br>
Статья на Interspeech 2013 ["An Open-source State-of-the-art Toolbox for Broadcast News Diarization"](https://www.isca-archive.org/interspeech_2013/rouvier13_interspeech.pdf)

<details>
  <summary>Подробнее.</summary>
  Это открытый инструмент для диаризации, разработанный лабораторией LIUM. Интструмент создан для обработки радио- и телепередач. Он предоставляет готовое решение для задач мультимедиа и позволяет разрабатывать новые системы диаризации.<br>

  * Инструмент включает в себя все необходимые пакеты и модели (UBM, модели для определения пола и речи/не-речи).
  * Инструмент может использовать сторонние функции, такие как MFCC, HTK, SPro и текстовые файлы.
  
  
  Используются следующие методы:

  - Извлечение признаков — MFCC.
  - Кластеризация — алгоритм гауссовых смесей (GMM).
  - Сегмантация — результаты сохраняются в формате RTTM.

  * **Важно:** инструмент написан на Java. Перед работой следует установить Java, а также скачать инструмент с [официального сайта.](https://projets-lium.univ-lemans.fr/spkdiarization/) <br>
    *Разработан на Java для минимизации проблем с зависимостями и работает как самостоятельный JAR-файл.*
  * В статьях рассказывают про два подхода: Single- and cross-show diarization (идентификация говорящего в одной записи и идентификация одного человека в нескольких разных записях). Для cross-show diarization лучше подходит ILP кластеризация.

  * Лучшие результаты DER (на корпусе ESTER 2):

  ||Single-show DER|Cross-show DER|
  |-|-|-|
  |CLR|11.27 %|20.43 %|
  |ILP|8.35 %| 17.51 %|

</details>

### 5. ALIZÉ
Статья ["ALIZE 3.0 - Open Source Toolkit for State-of-the-Art Speaker Recognition"](https://www.isca-archive.org/interspeech_2013/larcher13_interspeech.pdf)

<details>
  <summary>Подробнее.</summary>
  ALIZE — это открытая платформа для распознавания говорящих.
  
  [Официальный сайт](https://alize.univ-avignon.fr/).

  **Структура инструмента:**
  ```
  | LIA_RAL
    | LIA_SpkDet # Набор инструментов для выполнения всех задач, необходимых для системы аутентификации по голосу, — обучение модели, нормализация признаков, нормализация оценок и т. д.
    | LIA_SpkSeg # Для диаризации, то что нужно.
    | LIA_Utils # Утилиты для работы с различными форматами данных, используемыми в ALIZÉ, — GMM, функциями и т. д.
    | LIA_SpkTools # Библиотека, на которой основаны другие части; она предоставляет функции высокого уровня поверх ALIZE-core.
  ```

  **Ссылки на гитхаб:**

  * [LIA_RAL](https://github.com/ALIZE-Speaker-Recognition/LIA_RAL)
  * [LIA_SpkSeg](https://github.com/ALIZE-Speaker-Recognition/LIA_RAL/tree/master/LIA_SpkSeg)

  * Туториал по работе с инструментами [LIA_SpkSeg](https://alize.univ-avignon.fr/#:~:text=Tutorial%20for%20LIA_SpkSeg%20%E2%80%94%20Top%2Ddown%20Speaker%20Segmenting%20and%20Clustering%20System)

    Инструмент работает в два этапа *(Top-down Speaker Segmenting and Clustering System):* сегментация спикеров (без идентификации) + кластеризация.
  
  **Преимущества:**

  * Хорошо совмещается с другими решениями, теоретически можно встроить в Виспер.
  * Опенсорсный.
  
  **Недостатки:**

  * Довольно давний инструмент — 2013 год.

    
  
</details>

## Датасет для обучения моделей диаризации
**1) VoxConverse**
[diarizers-community/voxconverse](https://huggingface.co/speechbrain/emotion-diarization-wavlm-large)

<details>
  <summary>О датасете.</summary>
  50 часов звучащей речи. Преимущественно английский язык.
  
  Датасет содержит следующие данные:

  - **Аудиозаписи**
  
    - Формат: .wav (16 кГц, моно).
    
    - Источники: Публичные видео с YouTube (например, интервью, дебаты, ток-шоу).
  
  - **Аннотации**
  
    - Формат: RTTM (Rich Transcription Time Marked).
    
    - Содержание: Временные метки для каждого сегмента речи с указанием идентификатора говорящего.
  
  **Плюсы**
  1. Реальная и разнообразная речь: ютуб-шоу, дебаты.
  2. Открыт и готов к использованию.
  3. Аннотации прописаны вручную.
</details>

**2) AMI Metting Corpus**
[The AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
<details>
  <summary>О датасете.</summary>
  100 часов записей совещаний.

  * Формат: видео, mp4. Придется переводить в wav.
  * Язык: английский (не родной язык для спикеров).
  * Запись велась в 3 разных помещениях.
  * Ручная орфографическая транскрипция.
  * Лицензия на корпус позволяет пользователям копировать, распространять и отображать данные для любых целей при условии указания проекта AMI.
</details>

[Задача диаризации с использованием данного корпуса от SpeechBrain.](https://github.com/speechbrain/speechbrain/blob/develop/recipes/AMI/Diarization/README.md)

<details>
  <summary>Другие датасеты в обзоре SpeechBrain.</summary>
  
  * SpeechBrain собрали перечень датасетов для разных задач обработки речи ([GitHub](https://github.com/speechbrain/speechbrain/tree/develop/recipes)). Из интересного -> [Speaker recognition на данных VoxCeleb](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb/SpeakerRec).

  * Датасетов для диаризации больше не представлено.

</details>

**3) DIHARD**
В РАЗРАБОТКЕ

## Решения для сегментации говорящих
Ниже собраны интересные решения для задачи разделения говорящих (без диаризации = без идентификации конкретного говорящего).

### Модель от PyAnnotate
[pyannote/segmentation-3.0](https://huggingface.1319lm.top/pyannote/segmentation-3.0)

<details>
  <summary>О модели.</summary>
  Модель возвращает временные метки (start, end) для каждого сегмента, где активна речь. <br>
  
  **Преимущества:**
  
  - Легко интегрируется в другие решения от библиотеки PyAnnotate.audio <br>
  - Хорошо работает с шумными аудио.
  - Обучена на комбинации стандартных датасетов (AISHELL, AliMeeting, AMI, AVA-AVD, DIHARD, Ego4D, MSDWild, REPERE, and VoxConverse).
  - 
    **Недостатки:**
  
  - Требует много вычислительных ресурсов.<br>
  - Модель обучена в основном на английской речи, может хуже справляться с другими языками.
  - Может хуже справляться с нетипичными сценариями (данные обучения — дебаты, ток-шоу, интервью, телефонные разговоры).
</details>

### Модель от SpeechBrain
[speechbrain/vad-crdnn-libriparty](https://huggingface.co/speechbrain/vad-crdnn-libriparty)

<details>
  <summary>О модели.</summary>
  Модель возвращает матрицу вида: номер сегмента + старт + конец + наличие/отсутствие речи.

  ```
  segment_001  0.00  2.57 NON_SPEECH
  segment_002  2.57  8.20 SPEECH
  segment_003  8.20  9.10 NON_SPEECH
  segment_004  9.10  10.93 SPEECH
  segment_005  10.93  12.00 NON_SPEECH
  segment_006  12.00  14.40 SPEECH
  segment_007  14.40  15.00 NON_SPEECH
  segment_008  15.00  17.70 SPEECH
  ```
  * Принимает на вход аудио формата: моно, 16 кГц.
  * Обучалась на следующих датасетах:

     - LibriParty: https://drive.google.com/file/d/1--cAS5ePojMwNY5fewioXAv9YlYAWzIJ/view?usp=sharing
     - Musan: https://www.openslr.org/resources/17/musan.tar.gz
     - CommonLanguage: https://zenodo.org/record/5036977/files/CommonLanguage.tar.gz?download=1

</details>

### Архитектура SepFormer
<details>
  <summary>Подробнее</summary>
  
  Архитектура нейросети, основанная на трансформерах и механизме self-attention.<br>
  **Характеристики:** 
  
  - Масштабируется (работает с 2-3 и более говорящими).<br>
  - Учитывает контекст аудио (self-attention анализирует глобальные зависимости в аудиосигнале).<br>
  - Эффективно работает с шумом.<br>
    
  **Недостатки:**
  
  - Трансформеры требует много вычислительных ресурсов.<br>
  - Для обучения SepFormer требуется большой объем размеченных данных.

  **Модели:**
  
  - [speechbrain/sepformer-wham](https://huggingface.1319lm.top/speechbrain/sepformer-wham)
</details>

### Инструмент audioseg

<details>
  <summary>Подробнее</summary>
  Это инструмент для сегментации аудио, он делит аудиозаписи на сегменты: речь, музыка, тишина и шум. Инструмент способен находить целевого говорящего. <br> 

  * [Ссылка на GitHub](https://github.com/oliverwatts/audioseg/blob/master/README.md)
  * **Формат входных данных:** аудио, 16 кГц, формата .wav
  * Формат выходных данных:

  ```
  0.0000000 19.7126313 m
  ```
  Последний символ-буква означает:
  ```
  m: music only
  s: speech from target presenter
  ms: speech from target presenter with music in background
  p: audio from package (pkg)
  ```
  * **Интервал обработки сигнала:** 10 мс
  * Теоретически совместим с Whisper.

</details>

<details>
  <summary></summary>
  текст
  ```
  п
  ```
</details>
