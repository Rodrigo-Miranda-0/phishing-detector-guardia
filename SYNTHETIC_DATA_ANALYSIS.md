# Análisis de la Generación de Datos Sintéticos para Detección de Phishing

## Resumen

Este documento presenta un análisis detallado de los desafíos encontrados durante la generación de datos sintéticos para el entrenamiento de un modelo de detección de correos de phishing en español. Se documentan las limitaciones técnicas, los problemas de generalización y las lecciones aprendidas que son relevantes para la investigación en aprendizaje automático con datos sintéticos.

---

## 1. Introducción

### 1.1 Contexto del Problema

La detección automatizada de correos electrónicos de phishing representa un desafío significativo en el ámbito de la ciberseguridad corporativa. En el contexto hispanohablante, este problema se agrava por la escasez de conjuntos de datos etiquetados disponibles públicamente, ya que:

1. Los conjuntos de datos existentes están predominantemente en inglés
2. Los correos corporativos reales contienen información confidencial
3. Las características lingüísticas del phishing varían entre idiomas y culturas

### 1.2 Hipótesis Inicial

Se planteó que la generación de datos sintéticos mediante Modelos de Lenguaje Grande (LLMs) podría proporcionar un conjunto de datos de entrenamiento viable para un clasificador binario de phishing, permitiendo entrenar un modelo inicial antes de disponer de datos reales.

---

## 2. Metodología

### 2.1 Diseño del Pipeline de Generación

Se implementó un pipeline de tres etapas:

```
Generación de Prompts → Generación de Correos (LLM) → Limpieza de Datos
```

#### 2.1.1 Generación de Prompts

Se diseñaron prompts paramétricos que combinan múltiples dimensiones:

| Dimensión | Valores |
|-----------|---------|
| Arquetipo de phishing | Suspensión de cuenta, reseteo de contraseña, factura vencida, CEO fraud, etc. (10 tipos) |
| Arquetipo legítimo | Invitación a reunión, notificación de factura, confirmación de envío, etc. (8 tipos) |
| Tono | Formal, neutral, informal |
| Urgencia | Sin urgencia, moderada, alta |
| Enlaces | Con enlace de login, con enlace de pago, sin enlaces |
| Adjuntos | Con mención de PDF, sin adjuntos |
| Mezcla de idiomas | Español puro, español con anglicismos |

Esta combinación produjo **540 prompts únicos** para la generación.

#### 2.1.2 Modelos de Lenguaje Utilizados

Se evaluaron dos modelos para la generación:

1. **Llama 3.1 8B Instruct** (Meta) - Modelo alineado con filtros de seguridad
2. **Dolphin-Mistral** - Modelo sin censura basado en Mistral 7B

### 2.2 Métricas de Evaluación

Se utilizó la arquitectura **DistilBERT Multilingual** para el clasificador, evaluando:
- Accuracy, Precision, Recall, F1-Score en conjunto de validación
- Matriz de confusión
- Pruebas con ejemplos manuales externos al conjunto de datos

---

## 3. Resultados y Obstáculos Encontrados

### 3.1 Problema 1: Rechazo del Modelo por Filtros de Seguridad

**Observación:** Llama 3.1 8B Instruct rechazó generar aproximadamente el **33% de los prompts de phishing**, produciendo respuestas como:

> *"Lo siento, pero no puedo cumplir con esa solicitud."*
> *"I cannot write a phishing email."*

**Impacto cuantitativo:**

| Métrica | Valor |
|---------|-------|
| Prompts de phishing enviados | 300 |
| Respuestas rechazadas | 182 (60.7%) |
| Correos de phishing generados válidos | 107 |

**Análisis:** Los modelos de lenguaje alineados con RLHF (Reinforcement Learning from Human Feedback) incluyen salvaguardas que detectan solicitudes potencialmente maliciosas, incluso cuando el propósito es investigación legítima.

**Solución implementada:** Migración al modelo Dolphin-Mistral, que carece de estas restricciones. Este modelo generó **199 correos de phishing válidos** con una tasa de rechazo del 0%.

### 3.2 Problema 2: Detección de Idioma y Calidad

**Observación:** Un subconjunto de correos generados presentaba:
- Texto completamente en inglés (3 casos)
- Longitud insuficiente (<150 caracteres)
- Mezcla excesiva de idiomas

**Solución implementada:** Desarrollo de un algoritmo de limpieza con las siguientes reglas:

```python
CRITERIOS_DE_FILTRADO = {
    "longitud_minima": 150,
    "longitud_maxima": 2000,
    "indicadores_español_minimos": 2,
    "patrones_rechazo": ["lo siento", "i cannot", "i can't"],
}
```

### 3.3 Problema 3: Sobreajuste a Patrones Sintéticos

**Observación crítica:** El modelo entrenado alcanzó métricas perfectas en validación:

| Métrica | Valor |
|---------|-------|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1-Score | 1.0000 |

**Matriz de confusión en validación:**
```
              Predicho
              Legit  Phish
Real  Legit    47      0
      Phish     0     37
```

Sin embargo, al evaluar con **8 correos escritos manualmente** (no sintéticos):

| Categoría | Correctos | Accuracy |
|-----------|-----------|----------|
| Phishing | 0/4 | 0% |
| Legítimo | 4/4 | 100% |
| **Total** | **4/8** | **50%** |

**Diagnóstico:** El modelo aprendió a distinguir el "estilo sintético" del LLM generador, no las características reales del phishing. Específicamente:

1. **Sesgo de distribución:** Los correos sintéticos de phishing y legítimos provienen de distribuciones lingüísticas distintas pero artificiales.
2. **Patrones superficiales:** El modelo detectó marcadores léxicos o estructurales específicos del proceso de generación.
3. **Falta de variabilidad:** Los LLMs tienden a producir texto con patrones repetitivos que no reflejan la diversidad del lenguaje humano real.

---

## 4. Análisis Teórico

### 4.1 El Problema de la Transferencia de Dominio

El fenómeno observado corresponde a un caso de **domain shift** (desplazamiento de dominio), donde:

$$P_{sintético}(X, Y) \neq P_{real}(X, Y)$$

Donde $P_{sintético}$ representa la distribución conjunta de características y etiquetas en datos sintéticos, y $P_{real}$ la distribución en el mundo real.

### 4.2 Implicaciones para el Aprendizaje con Datos Sintéticos

Los resultados sugieren que la generación sintética mediante LLMs presenta limitaciones fundamentales cuando:

1. El dominio objetivo requiere **variabilidad estilística alta** (como correos escritos por múltiples actores)
2. Las clases a distinguir son **semánticamente similares** (phishing vs. legítimo comparten estructura de correo)
3. No existe **validación cruzada con datos reales** durante el desarrollo

### 4.3 Falacia de la Validación Interna

La métrica de 100% en validación ilustra un riesgo metodológico importante: **la validación con datos de la misma distribución sintética no garantiza generalización**. Este fenómeno se puede formalizar como:

$$\mathcal{L}_{val}^{sintético} \ll \mathcal{L}_{test}^{real}$$

Donde $\mathcal{L}$ representa la función de pérdida en cada conjunto.

---

## 5. Recomendaciones y Trabajo Futuro

### 5.1 Estrategias de Mitigación Inmediata

1. **Incorporación de datos reales:** Incluso 50-100 correos reales de phishing podrían mejorar significativamente la generalización.

2. **Aumento de datos (Data Augmentation):**
   - Introducción de errores ortográficos aleatorios
   - Variación en formato y estructura
   - Paráfrasis mediante modelos adicionales

3. **Generación adversarial:** Utilizar un modelo discriminador durante la generación para producir ejemplos más difíciles.

### 5.2 Enfoques Alternativos

1. **Sistemas híbridos:** Combinar clasificador ML con reglas basadas en heurísticas (detección de URLs sospechosas, análisis de headers, etc.)

2. **Few-shot learning:** Utilizar modelos pre-entrenados con capacidad de clasificación con pocos ejemplos

3. **Aprendizaje semi-supervisado:** Incorporar correos no etiquetados reales para regularizar las representaciones aprendidas

### 5.3 Consideraciones Éticas

El uso de modelos sin censura (como Dolphin) para generar contenido de phishing, aunque sea con fines de investigación, plantea consideraciones éticas que deben documentarse:

- El dataset generado debe mantenerse en entornos controlados
- No debe utilizarse para entrenar sistemas ofensivos
- La investigación debe orientarse exclusivamente a la defensa

---

## 6. Conclusiones

La generación de datos sintéticos mediante LLMs representa una herramienta prometedora pero con limitaciones significativas para el entrenamiento de clasificadores de texto. Los principales hallazgos de este estudio son:

1. **Los filtros de seguridad de LLMs alineados** representan un obstáculo técnico para la generación de datos de clases "sensibles" como phishing.

2. **La validación interna con datos sintéticos es insuficiente** para evaluar la capacidad de generalización del modelo.

3. **El sobreajuste a patrones sintéticos** es un riesgo real que puede producir métricas engañosamente optimistas.

4. **La diversidad y realismo de los datos** son más importantes que la cantidad cuando se trabaja con datos sintéticos.

Estos hallazgos subrayan la importancia de validar modelos entrenados con datos sintéticos utilizando conjuntos de prueba reales antes de cualquier despliegue en producción.

---

## Referencias

- Meta AI. (2024). Llama 3.1 Model Card. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.

---

*Documento generado como parte del desarrollo del Prototipo 1 - Detector de Phishing Corporativo en Español*

