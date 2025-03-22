# Instrucciones para el Reto de Transcripción y Resumen de Dictados Médicos

## Objetivo

El objetivo es implementar un pipeline para transcribir audios de dictados médicos y generar resúmenes de las transcripciones. Se deben comparar diferentes enfoques, incluyendo modelos tipo BERT y modelos LLMs, para evaluar cuál es el más adecuado para este caso. Además, se evaluará tu experiencia utilizando PyTorch y tu capacidad para seguir buenas prácticas de programación.

## Descripción del Dataset

El dataset consiste en un conjunto de archivos de audio con dictados médicos.

## Tareas

### 1. Preparación de Datos

- Identifica programáticamente si existen grabaciones defectuosas o con ruido excesivo.
- Documenta cualquier paso de limpieza o normalización de audio que consideres necesario.

### 2. Transcripción Automática (ASR)

- Implementa o integra un modelo de ASR y genera los archivos de texto correspondientes a cada audio.

### 3. Generación de Resúmenes

- **Modelo BERT-like:** Usa o fine-tunea un modelo tipo BERT (PromptBert, DistilBERT, etc.) para realizar el resumen de las transcripciones.
- **Modelo LLM:** Utiliza un LLM para obtener los resúmenes.
- Compara ambos enfoques.

### 4. Arquitectura

- Imagina que tu solución será desplegada en la nube. Proporciona un diagrama de arquitectura proponiendo el flujo, desde la carga de audios, la transcripción (ASR), el preprocesamiento de texto y el modelo final de generación de resúmenes.
