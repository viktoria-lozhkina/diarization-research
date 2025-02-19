# diarization-research

## Модели

pyannote/speaker-diarization-3.1
[ссылка](https://huggingface.co/pyannote/speaker-diarization-3.1) <br>

<details>
  <summary>Конфигурация для пайплайна</summary>
 
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
  <summary></summary>
  текст
</details>

<details>
  <summary></summary>
  текст
</details>
