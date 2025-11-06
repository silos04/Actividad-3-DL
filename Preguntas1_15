# ACTIVIDAD 3 DL
_Javier Silos Esteban_

---

## 1) Visión general

### P1. “Mapa rápido de Deep Learning”

El Deep Learning (DL) es una rama del Machine Learning que aprende representaciones jerárquicas de los datos para transformar datos crudos (como píxeles) en predicciones (como "gato"). Nace del Perceptrón (1958), pero despega gracias a Backpropagation (1986) y explota en la era moderna (2012) con GPUs y Big Data, impulsando la IA generativa. Resuelve el problema del ML clásico de tener que diseñar "features" a mano, ya que el DL las aprende automáticamente. Sus límites principales son la necesidad de grandes cantidades de datos, el alto coste computacional y la tendencia al sobreajuste. Un ejemplo sencillo es un clasificador que distingue fotos de perros y gatos.

*Mi duda es: ¿Cómo se gestiona eficientemente el "vanishing gradient" (desvanecimiento del gradiente) en redes extremadamente profundas, más allá de solo usar ReLU?*

### P2. “Del ML al DL en una imagen mental”

El DL aprende representaciones en capas porque imita un proceso de abstracción jerárquico. A diferencia del ML clásico, donde un experto debe definir "features" (ej. "medir la curvatura de la nariz"), el DL las descubre solo. Cada capa aprende features basándose en la salida de la capa anterior, volviéndose progresivamente más abstractas.

**Dibujo (Texto):** `Datos crudos (Píxeles) → Capa 1 (Aprende bordes y texturas) → Capa 2 (Aprende combinaciones: ojos, narices) → Capa 3 (Aprende conceptos: "cara de gato") → Salida ("Gato")`

### P3. “MLP en 60 segundos”

Un MLP (Perceptrón Multicapa) es una red neuronal compuesta por capas densas; cada capa tiene un conjunto de pesos (W) y sesgos (b). El **forward pass** es el proceso de cálculo hacia adelante: las entradas atraviesan la red, capa por capa, multiplicándose por los pesos, sumando los sesgos y aplicando funciones de activación, hasta generar una predicción final. **Backprop** (retropropagación) es el proceso inverso: calcula el error (gradiente) en la salida y lo propaga hacia atrás para ajustar los pesos (W) y sesgos (b) en la dirección que minimiza dicho error.

Arquitecturas como CNN o Transformers son extensiones que reemplazan las ineficientes capas densas del MLP por operaciones especializadas (convoluciones o auto-atención), pero siguen usando backprop como motor de optimización.

### P4. “Ejemplo numérico guiado (Forward 2–3–1)”

Asumamos una red 2-3-1 (Entrada: 2 neuronas, Oculta: 3 neuronas, Salida: 1 neurona) y activación ReLU para la capa oculta.

* **Entrada (a0):** `[1.0, 2.0]`
* **Pesos Oculta (W1):** `[[0.1, 0.5], [0.2, 0.3], [0.4, 0.1]]`
* **Sesgos Oculta (b1):** `[0.1, 0.1, 0.1]`
* **Pesos Salida (W2):** `[0.5, 0.2, 0.1]`
* **Sesgo Salida (b2):** `[0.0]`

**Cálculo Capa Oculta (z1 = W1 * a0 + b1):**
* `z1_1 = (1.0 * 0.1) + (2.0 * 0.5) + 0.1 = 1.2`
* `z1_2 = (1.0 * 0.2) + (2.0 * 0.3) + 0.1 = 0.9`
* `z1_3 = (1.0 * 0.4) + (2.0 * 0.1) + 0.1 = 0.7`

**Activación Oculta (a1 = ReLU(z1)):**
* `a1 = [1.2, 0.9, 0.7]` (Ninguno es < 0)

**Cálculo Capa Salida (z2 = W2 * a1 + b2):**
* `z2 = (1.2 * 0.5) + (0.9 * 0.2) + (0.7 * 0.1) + 0.0`
* `z2 = 0.6 + 0.18 + 0.07 = 0.85`

**Activación Salida (a2 = z2):** (Asumimos lineal para regresión)
* **Predicción = 0.85**

**Conclusión:** La predicción que sale (0.85) es el resultado de la red para la entrada [1.0, 2.0]. Si esto fuera un problema de regresión (ej. predecir una nota de examen), 0.85 sería la predicción del modelo.

### P5. “Por qué importa la pérdida”

Elijo la **Entropía Cruzada (Cross-Entropy)**.

Con mis palabras, la entropía cruzada mide la "sorpresa" o el desajuste entre lo que el modelo predijo (ej. "80% gato, 20% perro") y lo que era la verdad (ej. "100% gato, 0% perro"). Si el modelo está muy seguro y muy equivocado (ej. predice "99% perro" cuando era "gato"), la entropía cruzada lo penaliza enormemente.

Minimizar cambia los pesos porque el algoritmo de backprop usa el gradiente (la derivada) de esta función de pérdida. Al "minimizar la sorpresa", la red se ve forzada a ajustar sus pesos (W y b) para que la probabilidad asignada a la clase correcta sea lo más alta posible.

---

## 2) Temas Relevantes (P6 - P15)

### P6. Activaciones sin dolor

* **Sigmoide:** Rango (0, 1). Ventaja: Interpretable como probabilidad. Problema: Satura rápido (gradientes se desvanecen).
* **Tanh:** Rango (-1, 1). Ventaja: Centrada en cero (mejor que sigmoide). Problema: Satura.
* **ReLU:** Rango [0, ∞). Ventaja: Muy rápida, no satura en positivo, evita desvanecimiento. Problema: "ReLU muerta" (si z<0 persistente).
* **Leaky ReLU:** Rango (-∞, ∞). Ventaja: Soluciona la "ReLU muerta". Problema: Añade un hiperparámetro (la "fuga").

**Elección para capa oculta: ReLU.** La elijo porque es computacionalmente eficiente y, lo más importante, combate el problema del "vanishing gradient" (gradientes que se desvanecen) mucho mejor que Sigmoide o Tanh, permitiendo que redes más profundas entrenen eficazmente.

### P7. Arquitectura MLP mínima

n entradas, h neuronas ocultas, m salidas.

**Dibujo:**
`(Capa 0: Entrada) → (Capa 1: Oculta) → (Capa 2: Salida)`
`a(0) → a(1) → a(2)`
`(n neuronas) → (h neuronas) → (m neuronas)`

* **W(1), b(1):** Conectan la Entrada con la Oculta.
    * `W(1)` es la matriz de pesos de forma (h, n).
    * `b(1)` es el vector de sesgos de forma (h,).
* **W(2), b(2):** Conectan la Oculta con la Salida.
    * `W(2)` es la matriz de pesos de forma (m, h).
    * `b(2)` es el vector de sesgos de forma (m,).

### P8. Forward pass con ecuaciones
