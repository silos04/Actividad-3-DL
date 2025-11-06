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
z(l) = W(l) * a(l-1) + b(l) a(l) = σ(z(l))

**Flujo (Izquierda→Derecha):** El proceso comienza con las entradas a(0). Para cada capa l, primero calculamos z(l), que es la suma ponderada (multiplicación por pesos W(l)) de las activaciones de la capa anterior a(l-1), más el sesgo b(l). Luego, aplicamos la función de activación σ (sigma) a z(l) para introducir no linealidad y obtener la salida a(l), que se convierte en la entrada para la siguiente capa.

**Si cambias la activación (σ):** Cambias el comportamiento de la red. Si cambias de Sigmoide a ReLU , permites que las activaciones a(l) tomen valores mucho mayores (no están "comprimidas" entre 0 y 1), lo que a menudo acelera el entrenamiento pero puede requerir ajustar la tasa de aprendizaje.

### P9. Backprop sin miedo

En mis palabras: **δ(l) (delta) es el "error local"** de la capa l.

Mide cuánto contribuyó cada neurona de esa capa antes de la activación (en el valor z(l)) al error total de la predicción final. Si δ(l) es grande para una neurona, significa que esa neurona tuvo mucha "culpa" (positiva o negativa) en el error de la red.

Ayuda a saber cuánto cambiar cada peso porque, para calcular el gradiente de un peso W(l), multiplicas el error δ(l) de la neurona de llegada por la activación a(l-1) de la neurona de partida.

### P10. Pérdida y decisión

* **MSE (Mean Squared Error):** Mide la distancia cuadrática promedio entre la predicción y el valor real.
* **Entropía Cruzada (Cross-Entropy):** Mide el desajuste probabilístico entre la distribución predicha y la distribución real (la "sorpresa").

**Cuándo usar:**
* Usaría **MSE para Regresión**: cuando predigo un número continuo (ej. el precio de una casa, la temperatura de mañana).
* Usaría **Entropía Cruzada para Clasificación**: cuando predigo una categoría (ej. spam/no-spam , qué animal hay en una foto ).

### P11. Ejercicio numérico corto (Backprop)

* **Contexto:** La regla de actualización es: `W_nuevo = W_viejo - η * (∂L/∂W)`.
* **Gradiente:** El gradiente (`∂L/∂W`) es proporcional a `δ_llegada * a_partida`.
* **Datos:**
    * Error de llegada: `δ(2)` es negativo (ej. -0.5).
    * Activación de partida: `a(1)` es grande y positiva (ej. 1.0).
* **Cálculo:** El gradiente (`∂L/∂W`) ≈ (negativo) * (positivo) = **Negativo** (ej. -0.5).
* **Actualización:** `W_nuevo = W_viejo - η * (Negativo)`
    `W_nuevo = W_viejo + (η * Positivo)`
* **Respuesta:** El peso sube (aumenta).
* **Razonamiento:** El gradiente negativo indica que la función de pérdida disminuye si el peso aumenta. Por lo tanto, el optimizador (que resta el gradiente) termina sumando un valor.

### P12. Historia express

1.  **1958 (Perceptrón):** Se crea el Perceptrón, la primera neurona artificial que puede aprender. Cambio: Nace la idea, pero se demuestra que es limitada (no puede resolver el XOR ).
2.  **1986 (Backpropagation):** Se populariza la retropropagación. Cambio: Se proporciona el algoritmo clave para entrenar redes multicapa (MLPs), superando los límites del Perceptrón.
3.  **2012–2025 (Era DL / ImageNet):** El "Big Bang" del DL (ej. AlexNet gana ImageNet). Cambio: La disponibilidad masiva de GPUs (cómputo paralelo) y Big Data (grandes datasets) permite que las arquitecturas profundas (que ya existían teóricamente) se entrenen eficazmente y superen a todos los métodos clásicos.

### P13. Limitaciones del MLP → CNN

1.  **Ignora la estructura espacial:** Un MLP "aplana" la imagen (ej. 28x28 -> 784x1). Pierde la información de que un píxel está *cerca* de otro. Para un MLP, el píxel (0,0) está igual de relacionado con (0,1) que con (27,27).
2.  **Exceso de parámetros:** En un MLP denso, cada neurona de la primera capa oculta debe conectarse a *cada* píxel de la imagen. Esto es computacionalmente ineficiente y propenso al sobreajuste.

**Cómo lo resuelven las CNNs:** Las CNNs usan convoluciones, que son filtros pequeños que respetan la localización (miran solo vecindarios de píxeles). Además, comparten pesos (el mismo filtro detector de bordes se usa en toda la imagen), reduciendo drásticamente los parámetros.

### P14. Métrica de éxito y sobreajuste

* **El problema:** Se llama **Sobreajuste (Overfitting)**. El modelo está "memorizando" los datos de entrenamiento (train) perfectamente (por eso su accuracy sube), pero no está aprendiendo a generalizar a datos nuevos (por eso la accuracy de validación baja).
* **Remedio 1: Regularización.** Añadir Dropout o regularización L2 para penalizar la complejidad del modelo y forzarlo a generalizar.
* **Remedio 2: Data Augmentation.** Generar más datos de entrenamiento (ej. rotando, recortando o cambiando el color de las imágenes) para que el modelo tenga más variedad y no pueda memorizar tan fácilmente. (Otros remedios válidos: Early Stopping o simplificar la red).

### P15. Mini-resumen personal

* **Qué entendí hoy:** Entendí el flujo completo del MLP: cómo los datos fluyen (forward) , cómo se mide el error (pérdida) , y cómo ese error se usa para ajustar los pesos (backprop).
* **Qué me falta:** Me falta la intuición práctica de cómo depurar una red que no aprende. ¿Es el learning rate? ¿La inicialización de pesos? ¿O es que los datos no sirven?
* **Ejemplo real donde lo aplicaría:** Usaría un MLP para un problema de clasificación tabular. Por ejemplo, predecir si un cliente abandonará un servicio (churn) basándome en sus datos no-estructurados (tiempo de uso, tipo de plan, tickets de soporte), donde una CNN o Transformer no aplicarían directamente.
