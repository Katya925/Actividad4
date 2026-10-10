# Evaluación comparativa e implementación de tres modelos preentrenados de Hugging Face para tareas de análisis de sentimientos de reseñas en español. 

## 📌 Tabla de Contenidos

- [📖 Descripción del Caso de Uso](#caso-de-uso)
- [🛠️ Criterios Técnicos y Prácticos de Selección](#criterios-seleccion)
- [📊 Origen y Estructura del Dataset](#dataset)
- [📈 Tabla Comparativa de Modelos y Resultados](#tabla-resultados)
- [⚡ Análisis de Trade-Offs: Eficiencia frente a Precisión](#trade-offs)
- [💡 Decisiones de Ajuste Mínimo vs. Fine-Tuning y Justificación Final](#ajuste-minimo)
- [⚠️ Umbrales, Riesgos y Plan de Mitigación](#riesgos)
- [💻 Instrucciones de Uso e Instalación](#instalacion)
- [📂 Estructura del Repositorio](#estructura)

  
## <a name="caso-de-uso"></a>📖 Descripción del Caso de Uso
El objetivo de este proyecto es analizar, seleccionar e implementar un modelo de Inteligencia Artificial capaz de realizar la clasificación automática de sentimientos de un conjunto de reseñas de usuarios en español. El sistema procesa comentarios en lenguaje natural para asignarlos en las categorías (**POS**: Positivo, **NEG**: Negativo, **NEU**: Neutral). 
La meta es tener un **80% de Accuracy (0.80)**.

## <a name="criterios-seleccion"></a>🛠️ Criterios Técnicos y Prácticos de Selección
Para la evaluación se consideraron los siguientes criterios: 
1. **Capacidad y Dominio Lingüístico:** Arquitecturas basadas en *Transformers* entrenadas para el procesamiento de texto en español o corpus multilingües. 
2. **Diversidad de Arquitecturas:** Evaluación comparativa entre variaciones de RoBERTa en español (`RoBERTuito`), BERT en español (`BETO`) y variantes multilingües (`mBERT`). 
3. **Eficiencia Computacional:** Compatibilidad con procesamiento acelerado en **GPU T4**, con medición de tiempos de procesamiento por lote y latencia promedio por muestra en milisegundos. 
4. **Licenciamiento y Apertura:** Repositorios de acceso abierto que aseguren la reproducibilidad, investigación y transferencia tecnológica.

## <a name="dataset"></a>📊 Origen y Estructura del Dataset   
-**Origen:** Dataset del repositorio público `luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish` del sitio Kaggle.com utilizando la librería `kagglehub`.
El dataset contiene 50,000 reseñas de películas con un sentimiento positivo o negativo (debido al tamaño del archivo a este repositorio se subieron sólo 20,000 reseñas.)

-**Estructura del Dataset:**
| # | review_en | review_es | sentiment | sentimiento |
| :---- | :---: | :---: | :---: | :---: |
| integer | string | string | string | string |
| No.registro | reseña inglés | reseña español | sentimiento inglés | sentimiento español |

-**Muestra de Evaluación:** Se extrajo una muestra estratificada de $N=300$ registros válidos ($151POS/149NEG$), empleando una semilla 
fija (`SEED = 42`) para garantizar reproducibilidad exacta. Se realiza limpieza de los datos eliminando caractéres especiales, espacios redundantes y registros duplicados.
Etiqueta categórica de polaridad normalizada a `POS` (Positivo) y `NEG` (Negativo). 

## <a name="tabla-resultados"></a>📈 Tabla Comparativa de Modelos y Resultados  
Resultados obtenidos de la ejecución en Google Colab con aceleración GPU Tesla T4:

| Criterio / Modelo | BETO Sentiment | RoBERTuito | Multilingual BERT   |
| :---- | :---: | :---: | ----- |
| **Identificador en Hugging Face** | finiteautomata/beto-sentiment-analysis | pysentimiento/robertuito-sentiment-analysis | nlptown/bert-base-multilingual-uncased-sentiment |
| **Accuracy** | **0.7833** | 0.6633 | 0.6500 |
| **Precision (Macro)** | 0.8288 | **0.8354** | 0.7088 |
| **Recall (Macro)** | **0.7885** | 0.6670 | 0.6521 |
| **F1-Score (Macro)** | **0.7554** | 0.7258 | 0.6385 |
| **F1-Score (Weighted)** | **0.7570** | 0.7270 | 0.6397 |
| **Predicciones NEU asignadas** | 56 | 63 | 68 |
| **Latencia Total Promedio (s)** | 8.633 s | **2.699 s** | 6.438 s |
| **Latencia / Muestra (ms)** | 28.78 ms | **8.99 ms** | 21.46 ms |
| **Desviación Estándar Latencia (ms)** | 1.05 ms | **0.54 ms** | 0.08 ms |

![][image1]  
![][image2]

![][image3]![][image4]![][image5]

## <a name="trade-offs"></a>⚡ Análisis de Trade-Offs  
Eficiencia frente a Precisión :  
1. **BETO Sentiment (`finiteautomata`):**
- **Precisión:** Presenta el desempeño técnico más alto del experimento, con un **Accuracy del 78.33%** y un **F1-Score Macro de 0.7554**.   
- **Eficiencia:** Registra la latencia más elevada ($28.78ms/muestra$), derivada de la tokenización detallada sobre secuencias extensas.   
2. **RoBERTuito (`pysentimiento`):**  
- **Precisión:** Muestra un rendimiento intermedio (**Accuracy de 66.33%** y **F1-Score Macro de $0.7258$**), destacando con la mayor **Precision Macro (0.8354)**.   
- **Eficiencia:** Es la arquitectura más rápida y eficiente, registrando únicamente **$8.99ms/muestra$** ($3.2$ veces más veloz que BETO).   
3. **Multilingual BERT (`nlptown`):**   
- **Precisión:** Alcanza el menor rendimiento (**Accuracy de 65.00%** y **F1-Score Macro de $0.6385$**).   
- **Eficiencia / Sesgo:** Su diseño nativo orientado a 5 clases (estrellas) incrementa la dispersión al mapearse hacia un esquema binario, asignando 68 muestras a la categoría neutral.

## <a name="ajuste-minimo"></a>💡 Decisiones de Ajuste Mínimo vs. Fine-Tuning y Justificación del modelo  
### Análisis del Umbral de Negocio  
El umbral objetivo es del **80.00% de Accuracy**.   
El modelo preentrenado líder (**BETO Sentiment**) alcanzó un **78.33%**, a $1.67%$ del objetivo.   
### Decisión Técnica   
**Fase de Despliegue Inicial (Ajuste Mínimo): **ajustes mínimos** a través de la re-calibración de los umbrales de probabilidad (eliminando la asignación estricta a la clase neutral `NEU`), permitiendo superar el umbral del 80% sin incurrir en costos computacionales de reentrenamiento.  
**Fase Secundaria (Fine-Tuning):** Si se requiere adaptar el modelo a un léxico altamente especializado, se justifica la ejecución de Fine-Tuning mediante técnicas eficientes como PEFT/LoRA.   
### Modelo Recomendado  
Se selecciona **`finiteautomata/beto-sentiment-analysis` (BETO Sentiment)** como el modelo principal del prototipo debido a su superioridad en precisión, calidad predictiva y alineación con el idioma español.

## <a name="riesgos"></a>⚠️ Umbrales, Riesgos y Plan de Mitigación   
- **Riesgo 1: Muestras clasificadas erróneamente como Neutras (\`NEU\`).**  
- *Diagnóstico:* Asignación de clases intermedias en modelos preentrenados binarios.  
- *Mitigación:* Calibración de umbrales probabilísticos de salida para forzar clasificación binaria en los casos requeridos.  
- **Riesgo 2: Incremento de latencia en procesamiento por lotes masivo.**   
- *Diagnóstico:* Latencia de $28.78ms$ en BETO para volúmenes altos.   
- *Mitigación:* Implementación de cuantización a 8 bits (`bitsandbytes`) o exportación del modelo al formato ONNX Runtime.

## <a name="instalacion"></a>💻 Instrucciones de Uso e Instalación
### 1. Clonar el repositorio ```bash git clone https\://github.com/tu-usuario/tu-repositorio.git cd tu-repositorio ```  
### 2. Instalación de dependencias ```bash pip install \-r requirements.txt ```  
### 3. Ejecución en Google Colab  
1. Abrir el notebook `notebooks/Actividad4.ipynb` en Google Colab.  
2. Configurar la GPU T4 (`Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU T4`).  
3. Ejecutar las celdas secuencialmente. Se incluye la semilla `SEED = 42` para asegurar reproducibilidad.

## <a name="estructura"></a>📂 Estructura del Repositorio  
```text
.  
├── data/              		# Dataset de kagglehub  
├── notebooks/         		# Actividad4.ipynb (Notebook ejecutable en Google Colab)  
├── README.md          	 # Documentación técnica general del proyecto
└── requirements.txt   		# Lista de dependencias del entorno Python  
