# diarization-research

## Модели

pyannote/speaker-diarization-3.1
[ссылка](https://huggingface.co/pyannote/speaker-diarization-3.1) <br>

<details>
  <summary>Подробно о модели.</summary>
  
  Модель для диаризации: <br>
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
  <summary></summary>
  текст
  ```

  ```
</details>
