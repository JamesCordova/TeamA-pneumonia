const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, PageBreak,
} = require("docx");

const REPO = "D:/Ingenieria de Sistemas/Neumonia repo/TeamA-pneumonia";
const EQS = JSON.parse(fs.readFileSync(
  path.join(REPO, "pruebasg/results/tesis/ecuaciones/manifiesto.json"), "utf8"));

function p(t, o = {}) {
  return new Paragraph({
    alignment: o.align || AlignmentType.JUSTIFIED,
    spacing: { after: o.after ?? 125, line: 310 },
    children: [new TextRun({ text: t, bold: !!o.bold, italics: !!o.italics })],
  });
}
function h1(t) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 60 },
    shading: { type: ShadingType.CLEAR, fill: "1F4E79" },
    indent: { left: 110, right: 110 },
    children: [new TextRun({ text: "  " + t, bold: true, size: 24, color: "FFFFFF", font: "Arial" })],
  });
}
function h2(t) {
  return new Paragraph({
    spacing: { before: 250, after: 90 },
    children: [new TextRun({ text: t, bold: true, size: 22, color: "1F4E79" })],
  });
}
function rot(t) {
  return new Paragraph({
    spacing: { before: 150, after: 50 },
    children: [new TextRun({ text: t, bold: true, size: 19, color: "7A5C00",
                             allCaps: true })],
  });
}
/** Texto que se inserta tal cual en la tesis. */
function nuevo(t) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 70, after: 120, line: 310 },
    indent: { left: 400, right: 230 },
    shading: { type: ShadingType.CLEAR, fill: "F1F6EE" },
    children: [new TextRun({ text: t, size: 20 })],
  });
}
/** Entrada de la lista «Donde:» */
function donde(sim, desc) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 50, line: 290 },
    indent: { left: 700, right: 230, hanging: 180 },
    children: [
      new TextRun({ text: sim + "  ", bold: true, size: 20 }),
      new TextRun({ text: ": " + desc, size: 20 }),
    ],
  });
}
function eq(id) {
  const m = EQS[id];
  if (!m) throw new Error("falta ecuación " + id);
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(m.file),
                              transformation: { width: m.w, height: m.h } })],
  });
}
function nota(t, fill) {
  return new Paragraph({
    spacing: { before: 110, after: 140 },
    shading: { type: ShadingType.CLEAR, fill: fill || "EEF3F8" },
    indent: { left: 170, right: 170 },
    children: [new TextRun({ text: t, size: 19 })],
  });
}
function tabla(widths, enc, filas, titulo, fsz = 17) {
  const W = (i) => widths[i] ?? widths[widths.length - 1];
  const out = [];
  if (titulo) out.push(new Paragraph({
    spacing: { before: 140, after: 55 },
    children: [new TextRun({ text: titulo, size: 18, italics: true })] }));
  const head = new TableRow({
    tableHeader: true,
    children: enc.map((t, i) => new TableCell({
      width: { size: W(i), type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "E6E6E6" },
      children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, size: fsz })] })] })),
  });
  const body = filas.map((f) => new TableRow({
    children: f.map((t, i) => new TableCell({
      width: { size: W(i), type: WidthType.DXA },
      children: [new Paragraph({ spacing: { before: 20, after: 20 },
                                 children: [new TextRun({ text: String(t), size: fsz })] })] })),
  }));
  out.push(new Table({ columnWidths: widths,
                       width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
                       rows: [head, ...body] }));
  out.push(new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "" })] }));
  return out;
}

const c = [];

// ======================= PORTADA =======================
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 70 },
  children: [new TextRun({ text: "EXPLICACIÓN DETALLADA DE LAS FÓRMULAS NUEVAS", bold: true, size: 28 })] }));
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "Capítulo III · apartados 3.6.1, 3.8.3, 3.10 y 3.11.4, y cierre del 3.11.6",
                           italics: true, size: 21 })] }));
c.push(p(
  "Las dieciocho fórmulas incorporadas en esos cuatro apartados se presentan aquí con el mismo " +
  "tratamiento que el documento da a las fórmulas de los apartados anteriores: la expresión, la " +
  "lista de símbolos encabezada por «Donde:», y a continuación la explicación de qué mide, cómo se " +
  "calcula y cómo se lee. Se añade en cada caso el rango de valores observado en las 25 series del " +
  "estudio, de modo que la fórmula no quede en el plano abstracto."
));
c.push(p(
  "El texto sombreado en verde es el que se inserta en la tesis. Lo demás son indicaciones de " +
  "ubicación."
));
c.push(...tabla([1500, 4000, 3287],
  ["Apartado", "Fórmulas que contiene", "Nº"],
  [["3.6.1", "Vector de perfil, fuerza de tendencia, fuerza estacional, índice de dispersión, entropía espectral y proporción de semanas sin casos", "6"],
   ["3.8.3", "Diferencia pareada, estadístico de la prueba y conjunto de modelos equivalentes", "3"],
   ["3.10", "Función del criterio, regla de descarte, ruido de Poisson, umbrales, cobertura, exceso de error y ahorro", "7"],
   ["3.11.4", "Importancia global y atribución agregada por familia", "2"]]));

// ======================================================================
// 3.6.1
// ======================================================================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("3.6.1  Perfil estadístico de la serie"));
c.push(p(
  "Las seis explicaciones siguientes se insertan inmediatamente después de cada fórmula del " +
  "apartado, en el orden en que aparecen.", { italics: true }));

// --- 1 vector
c.push(h2("Fórmula 1 · Vector de perfil"));
c.push(eq("t_perfil"));
c.push(rot("Donde"));
c.push(donde("v(y)", "vector de perfil de la serie y; es la entrada del criterio de selección"));
c.push(donde("n", "longitud de la serie en semanas"));
c.push(donde("ȳ", "media de casos semanales, que mide la escala o volumen de la serie"));
c.push(donde("ID", "índice de dispersión (fórmula 4)"));
c.push(donde("F_T", "fuerza de tendencia (fórmula 2)"));
c.push(donde("F_S", "fuerza estacional (fórmula 3)"));
c.push(donde("H", "entropía espectral normalizada (fórmula 5)"));
c.push(donde("ρ₁, ρ₅₂", "autocorrelación en los rezagos de una y cincuenta y dos semanas"));
c.push(donde("p₀", "proporción de semanas sin casos notificados (fórmula 6)"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La expresión no es una ecuación que se resuelva sino la definición de un objeto: el perfil es " +
  "el vector que reúne todos los descriptores calculados sobre la serie. Su papel en la propuesta " +
  "es el de interfaz entre la caracterización estadística y la decisión de modelado: la Etapa 3 lo " +
  "produce y la Etapa 7 lo consume. Cada componente es un número real obtenido a partir de la " +
  "propia serie, sin intervención de información externa a ella."
));
c.push(nuevo(
  "Lo determinante del vector no es qué contiene sino qué no contiene. No incluye el nombre del " +
  "departamento, su población, su región natural ni la enfermedad registrada. Dos series con el " +
  "mismo perfil son, para el criterio, el mismo problema, con independencia de dónde y de qué " +
  "procedan. Esta propiedad es la que permite afirmar que el criterio construido sobre el perfil " +
  "es transferible a otras series de notificación obligatoria y no un resultado particular de la " +
  "neumonía en el Perú."
));
c.push(nuevo(
  "Los puntos suspensivos de la expresión indican que el perfil admite descriptores adicionales. " +
  "La implementación calcula treinta, de los cuales la versión del criterio evaluada en esta " +
  "investigación emplea dos: el volumen medio y la fuerza estacional. Los restantes quedan " +
  "disponibles para la versión siguiente del criterio."
));

// --- 2 y 3 fuerzas
c.push(h2("Fórmulas 2 y 3 · Fuerza de tendencia y fuerza estacional"));
c.push(eq("t_ft"));
c.push(eq("t_fs"));
c.push(rot("Donde"));
c.push(donde("T_t", "componente de tendencia de la descomposición STL en la semana t"));
c.push(donde("S_t", "componente estacional en la semana t"));
c.push(donde("R_t", "componente residual en la semana t"));
c.push(donde("Var(·)", "varianza muestral de la componente indicada sobre toda la serie"));
c.push(donde("max(0, ·)", "truncamiento a cero, que evita valores negativos sin sentido"));
c.push(rot("Explicación"));
c.push(nuevo(
  "Ambas medidas responden a la misma pregunta aplicada a componentes distintas: cuánta de la " +
  "variabilidad de la serie explica esa componente. La construcción es la misma en los dos casos. " +
  "Se toma la varianza del residuo —lo que queda sin explicar— y se la compara con la varianza de " +
  "la componente de interés sumada a ese mismo residuo. Si la componente no aportara nada, ambas " +
  "varianzas serían iguales, el cociente valdría uno y la medida resultaría cero. Si la componente " +
  "explicara casi toda la variabilidad, el residuo sería muy pequeño frente a la suma, el cociente " +
  "se aproximaría a cero y la medida a uno."
));
c.push(nuevo(
  "La razón de comparar contra la componente más el residuo, y no contra la varianza total de la " +
  "serie, es que así cada medida aísla su propia componente de la otra. La fuerza estacional de " +
  "una serie con tendencia muy marcada no queda artificialmente reducida por esa tendencia, porque " +
  "la tendencia no interviene en el denominador."
));
c.push(nuevo(
  "El truncamiento en cero cubre un caso que puede darse en la práctica: cuando la componente " +
  "estimada es puro ruido, la varianza del residuo puede superar ligeramente la de la suma y el " +
  "cociente exceder la unidad, lo que produciría un valor negativo carente de interpretación. En " +
  "esa situación la medida se fija en cero, que es su lectura correcta: la componente no aporta " +
  "información."
));
c.push(rot("Valores observados en el estudio"));
c.push(...tabla([2500, 2000, 2000, 2287],
  ["Medida", "Mínimo", "Máximo", "Lectura"],
  [["Fuerza de tendencia F_T", "0,100 (Tumbes)", "0,829 (Huánuco)",
    "Casi todas las series tienen tendencia apreciable"],
   ["Fuerza estacional F_S", "0,072 (Tumbes)", "0,661 (Puno)",
    "La estacionalidad separa con nitidez la sierra del resto"]],
  "Rango de ambas medidas en las 25 series departamentales"));
c.push(nuevo(
  "El umbral de 0,60 que emplea el criterio de selección se sitúa, a la vista de estos valores, en " +
  "el extremo alto de la distribución observada: solo las series de sierra lo superan. Ese umbral " +
  "no se eligió por esa razón, sino que procede de la literatura de descomposición, pero conviene " +
  "señalar que separa efectivamente un grupo reconocible de series."
));

// --- 4 dispersión
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h2("Fórmula 4 · Índice de dispersión"));
c.push(eq("t_disp"));
c.push(rot("Donde"));
c.push(donde("s²", "varianza muestral de los casos semanales"));
c.push(donde("ȳ", "media de casos semanales"));
c.push(rot("Explicación"));
c.push(nuevo(
  "El índice de dispersión contrasta la variabilidad observada contra la que cabría esperar si las " +
  "notificaciones semanales se comportaran como un proceso de Poisson. Esa distribución tiene la " +
  "particularidad de que su varianza es igual a su media, de modo que el cociente entre ambas vale " +
  "exactamente uno. Cualquier desviación respecto de la unidad mide, por tanto, cuánto se aparta " +
  "la serie de ese comportamiento de referencia."
));
c.push(nuevo(
  "La comparación con Poisson no es arbitraria: es la distribución que describe el número de " +
  "eventos independientes ocurridos en un intervalo fijo, que sería el caso si cada enfermo se " +
  "contagiara con independencia de los demás. Un índice superior a uno —sobredispersión— indica " +
  "que esa independencia no se cumple, y en una enfermedad transmisible es exactamente lo " +
  "esperable: los casos se agrupan en brotes, y a ello se suman las irregularidades del propio " +
  "proceso de notificación."
));
c.push(nuevo(
  "El índice es adimensional, lo que permitiría compararlo entre series de distinto volumen. Debe " +
  "advertirse, sin embargo, que no es independiente de la escala: una serie de mayor volumen " +
  "tiende a presentar un índice mayor, porque la varianza crece más que proporcionalmente respecto " +
  "de la media cuando hay agrupamiento. Por eso el índice se interpreta junto con el volumen y no " +
  "en su lugar."
));
c.push(rot("Valores observados en el estudio"));
c.push(nuevo(
  "El índice recorre desde 2,80 en Tacna hasta 37,24 en Cajamarca. Las 25 series presentan " +
  "sobredispersión sin excepción, lo que descarta el supuesto de Poisson puro para todas ellas y " +
  "confirma la pertinencia de incorporar este descriptor al perfil. La sobredispersión es más " +
  "acusada en las series de mayor volumen: Lima alcanza 28,23 frente al 2,80 de Tacna, que es la " +
  "de menor volumen del estudio."
));

// --- 5 entropía
c.push(h2("Fórmula 5 · Entropía espectral normalizada"));
c.push(eq("t_ent"));
c.push(rot("Donde"));
c.push(donde("f(ωᵢ)", "densidad espectral estimada en la frecuencia ωᵢ"));
c.push(donde("pᵢ", "densidad espectral normalizada, que suma uno sobre todas las frecuencias"));
c.push(donde("m", "número de frecuencias evaluadas"));
c.push(donde("ln m", "factor de normalización que acota el resultado al intervalo de cero a uno"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La medida se construye en tres pasos. Primero se estima la densidad espectral de la serie, que " +
  "indica cuánta de su variabilidad corresponde a cada frecuencia: una serie con un ciclo anual " +
  "marcado concentra energía en la frecuencia correspondiente a 52 semanas, mientras que una serie " +
  "irregular la reparte entre todas. Segundo, esa densidad se divide entre su suma total, con lo " +
  "que se convierte en una distribución de probabilidad sobre las frecuencias. Tercero, se calcula " +
  "la entropía de Shannon de esa distribución y se la divide entre el logaritmo del número de " +
  "frecuencias, que es su valor máximo posible."
));
c.push(nuevo(
  "El resultado queda acotado entre cero y uno y admite una lectura directa como medida de " +
  "predictibilidad. Un valor próximo a cero corresponde a un espectro concentrado en pocas " +
  "frecuencias, es decir a una serie con estructura periódica reconocible y por tanto más " +
  "predecible. Un valor próximo a uno corresponde a un espectro plano, que es el del ruido blanco: " +
  "toda frecuencia contribuye por igual y no hay estructura que un modelo pueda aprovechar."
));
c.push(nuevo(
  "La normalización por el logaritmo del número de frecuencias es necesaria porque la entropía de " +
  "Shannon crece con el número de categorías sobre las que se calcula. Sin ella, dos series de " +
  "distinta longitud —que producen un número distinto de frecuencias estimadas— no serían " +
  "comparables entre sí."
));
c.push(rot("Valores observados en el estudio"));
c.push(nuevo(
  "La entropía recorre desde 0,589 en Arequipa hasta 0,947 en Madre de Dios. El ordenamiento " +
  "resulta coherente con la fuerza estacional: las series de sierra, que son las de estacionalidad " +
  "más marcada, presentan la entropía más baja —Puno 0,600 y Cusco 0,633—, mientras que las series " +
  "de bajo volumen y estacionalidad débil se aproximan al ruido blanco, con Madre de Dios y Tacna " +
  "por encima de 0,91. Que dos descriptores calculados por procedimientos completamente distintos " +
  "—uno por descomposición y otro por análisis espectral— ordenen las series de forma concordante " +
  "respalda la consistencia del perfil."
));

// --- 6 ceros
c.push(h2("Fórmula 6 · Proporción de semanas sin casos"));
c.push(eq("t_p0"));
c.push(rot("Donde"));
c.push(donde("1[·]", "función indicadora, que vale uno si la condición se cumple y cero si no"));
c.push(donde("y_t", "casos notificados en la semana t"));
c.push(donde("n", "número total de semanas de la serie"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La expresión cuenta cuántas semanas registraron cero casos y divide ese recuento entre el total " +
  "de semanas. Es el descriptor más simple del perfil y, en las series de bajo volumen, el de " +
  "mayor consecuencia práctica."
));
c.push(nuevo(
  "Su relevancia es doble. Por el lado del modelado, una proporción alta de ceros indica que la " +
  "serie se acerca al régimen de conteo escaso, en el que las variables de rezago aportan poca " +
  "información —el rezago de una semana vale cero con frecuencia— y los modelos que suponen una " +
  "variable continua se ajustan mal. Por el lado de la evaluación, determina si el error porcentual " +
  "absoluto medio es calculable: esa métrica divide entre el valor observado y queda indefinida " +
  "cuando éste es cero."
));
c.push(rot("Valores observados en el estudio"));
c.push(nuevo(
  "La proporción va desde cero en Lima, que no registra ninguna semana sin casos en veinticuatro " +
  "años, hasta 0,324 en Tacna, donde casi una de cada tres semanas carece de notificaciones. Le " +
  "siguen Moquegua con 0,306 y Tumbes con 0,179. En esos tres departamentos el error porcentual " +
  "absoluto medio resulta inaplicable sobre una fracción considerable del período, lo que " +
  "justifica la decisión, declarada en el apartado 3.8.2, de adoptarlo únicamente como métrica de " +
  "contexto."
));

// ======================================================================
// 3.8.3
// ======================================================================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("3.8.3  Determinación del conjunto de modelos estadísticamente equivalentes"));

c.push(h2("Fórmula 7 · Diferencia pareada de error"));
c.push(eq("t_dif"));
c.push(rot("Donde"));
c.push(donde("d_k", "diferencia de error absoluto entre los dos modelos en el origen k"));
c.push(donde("e_k^(A), e_k^(B)", "error cometido por cada modelo en el origen de pronóstico k"));
c.push(donde("K", "número total de orígenes de pronóstico de la validación"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La expresión define la cantidad sobre la que se realiza toda la comparación. En lugar de " +
  "contrastar el error medio de un modelo contra el error medio del otro, se calcula la diferencia " +
  "entre ambos errores en cada origen de pronóstico por separado, y es esa serie de diferencias la " +
  "que se analiza."
));
c.push(nuevo(
  "El procedimiento recibe el nombre de pareado porque cada diferencia enfrenta a los dos modelos " +
  "sobre exactamente la misma semana. Esto tiene una consecuencia estadística importante: la " +
  "dificultad intrínseca de esa semana —que puede ser un pico epidémico o un período " +
  "excepcionalmente estable— afecta por igual a ambos modelos y se cancela al restar. Lo que queda " +
  "en d_k es únicamente la diferencia atribuible a los modelos, despojada de la variabilidad " +
  "propia de la semana evaluada."
));
c.push(nuevo(
  "El signo de la diferencia es informativo: un valor negativo indica que el modelo A erró menos " +
  "que el B en ese origen. Si los dos modelos fueran equivalentes, las diferencias se repartirían " +
  "alrededor de cero sin predominio de ningún signo."
));

c.push(h2("Fórmula 8 · Estadístico de la prueba"));
c.push(eq("t_t"));
c.push(rot("Donde"));
c.push(donde("d̄", "media de las K diferencias pareadas"));
c.push(donde("s_d", "desviación estándar muestral de esas diferencias"));
c.push(donde("K", "número de orígenes de pronóstico"));
c.push(donde("t_{K−1}", "distribución t de Student con K − 1 grados de libertad"));
c.push(rot("Explicación"));
c.push(nuevo(
  "El estadístico contrasta la hipótesis nula de que ambos modelos tienen el mismo error esperado, " +
  "es decir de que la media verdadera de las diferencias es cero. Su construcción es la habitual: " +
  "la media observada de las diferencias se divide entre su error estándar, que es la desviación " +
  "estándar dividida entre la raíz del número de observaciones."
));
c.push(nuevo(
  "La interpretación del cociente es directa. El numerador mide cuán lejos de cero está la " +
  "diferencia media observada; el denominador mide cuánta incertidumbre tiene esa media por " +
  "proceder de una muestra finita. El estadístico expresa, por tanto, a cuántos errores estándar " +
  "de distancia del cero se encuentra la diferencia. Un valor pequeño indica que la diferencia " +
  "observada cabe dentro de lo que la variabilidad muestral produciría aunque los modelos fueran " +
  "idénticos."
));
c.push(nuevo(
  "El uso de la distribución t de Student, y no de la normal, responde a que la desviación " +
  "estándar de las diferencias se estima a partir de la propia muestra en lugar de conocerse de " +
  "antemano. Los K − 1 grados de libertad reflejan esa estimación: una de las K observaciones " +
  "queda determinada una vez fijada la media."
));
c.push(nota(
  "La validez del estadístico exige que las diferencias puedan tratarse como observaciones " +
  "independientes. Esa condición se satisface por el diseño del esquema de validación: con " +
  "horizonte de una semana y desplazamiento de cuatro semanas entre orígenes, las semanas " +
  "evaluadas en orígenes sucesivos no se solapan. Si el horizonte igualara o superara al " +
  "desplazamiento, las ventanas se solaparían y el supuesto dejaría de sostenerse."
));
c.push(rot("Valores en el estudio"));
c.push(nuevo(
  "El número de orígenes es de 183 o 184 en los departamentos con serie completa, y desciende " +
  "hasta 79 en Moquegua, cuya serie se trunca en 2008 por sub-registro inicial. El nivel de " +
  "significación adoptado es de 0,05."
));

c.push(h2("Fórmula 9 · Conjunto de modelos estadísticamente equivalentes"));
c.push(eq("t_empate"));
c.push(rot("Donde"));
c.push(donde("ℰᵢ", "conjunto de modelos equivalentes al mejor en la serie i"));
c.push(donde("ℳ", "conjunto de familias de modelo evaluadas"));
c.push(donde("p_m", "valor p de la comparación del modelo m contra el de menor error"));
c.push(donde("α", "nivel de significación, fijado en 0,05"));
c.push(donde("m*ᵢ", "modelo de menor error medio en la serie i"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La expresión define el resultado que sustituye a la elección de un ganador único. El conjunto " +
  "reúne a todos los modelos cuya comparación contra el de menor error arroja un valor p igual o " +
  "superior al nivel de significación, es decir, aquellos cuya diferencia respecto del mejor no " +
  "alcanza a declararse significativa. La unión con el propio mejor modelo garantiza que el " +
  "conjunto nunca resulte vacío."
));
c.push(nuevo(
  "Debe subrayarse qué significa y qué no significa la pertenencia a este conjunto. No afirma que " +
  "los modelos tengan el mismo error, que es algo que los datos no pueden demostrar, sino que la " +
  "evidencia disponible no basta para distinguirlos. Es una afirmación sobre el poder del " +
  "experimento tanto como sobre los modelos: con más orígenes de pronóstico, algunas de esas " +
  "diferencias podrían volverse significativas."
));
c.push(nuevo(
  "La consecuencia metodológica es la que motiva el apartado. Si el conjunto contiene un solo " +
  "modelo, la elección está determinada por el error. Si contiene varios, el error ya no " +
  "discrimina entre ellos y la elección debe apoyarse en otros criterios —costo computacional, " +
  "interpretabilidad, disponibilidad de intervalos de predicción—, que son precisamente los que el " +
  "criterio de la Etapa 7 incorpora."
));

// ======================================================================
// 3.10
// ======================================================================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("3.10  Etapa 7: Criterio de selección automática de modelo"));

c.push(h2("Fórmula 10 · Función del criterio"));
c.push(eq("t_selector"));
c.push(rot("Donde"));
c.push(donde("s", "el criterio de selección, entendido como una función"));
c.push(donde("v(y)", "perfil estadístico de la serie y, definido en el apartado 3.6.1"));
c.push(donde("m", "familia de modelo recomendada"));
c.push(donde("ℳ", "conjunto de familias disponibles: SARIMA, Random Forest, XGBoost y LSTM"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La expresión formaliza el aporte de la investigación como un objeto matemático preciso: una " +
  "función cuyo dominio es el espacio de perfiles y cuyo recorrido es el conjunto de familias de " +
  "modelo. Plantearlo así, y no como un procedimiento descrito en prosa, tiene dos ventajas. " +
  "Delimita con exactitud qué recibe y qué devuelve el criterio, y permite evaluarlo como se " +
  "evalúa cualquier función de decisión, comparando su salida contra la que habría sido correcta."
));
c.push(nuevo(
  "Dos propiedades del dominio y del recorrido definen el alcance de la propuesta. Respecto del " +
  "dominio, la función recibe el perfil y no la identificación de la serie: no conoce el " +
  "departamento, la región ni la enfermedad, de modo que su aplicabilidad no queda restringida al " +
  "caso de estudio. Respecto del recorrido, devuelve una familia de modelo y no una configuración " +
  "de hiperparámetros: el criterio decide qué familia emplear, y el ajuste dentro de la familia " +
  "elegida sigue el procedimiento del apartado 3.7.5."
));

c.push(h2("Fórmula 11 · Regla de descarte"));
c.push(eq("t_descarte"));
c.push(rot("Donde"));
c.push(donde("n_sec", "número de secuencias de entrenamiento que la ventana permite construir"));
c.push(donde("m_tr", "longitud de la ventana de entrenamiento en semanas"));
c.push(donde("L", "retrospección de la red recurrente, es decir el número de semanas que recibe como entrada"));
c.push(donde("N_par", "número de parámetros entrenables de la red"));
c.push(donde("ℳ ← ℳ∖{LSTM}", "eliminación de la familia recurrente del conjunto de candidatas"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La regla establece una condición de identificabilidad que puede verificarse antes de entrenar " +
  "ningún modelo. Una red recurrente se alimenta de secuencias: cada ejemplo de entrenamiento es " +
  "una ventana de L semanas consecutivas. Una serie de entrenamiento de m_tr semanas permite " +
  "construir aproximadamente m_tr − L secuencias de ese tipo, porque las primeras L semanas se " +
  "consumen en formar la primera ventana."
));
c.push(nuevo(
  "La condición compara ese número de ejemplos con el número de parámetros que la red debe " +
  "estimar. Cuando los parámetros superan a los ejemplos, el sistema está subdeterminado: existen " +
  "infinitas combinaciones de pesos que reproducen los datos de entrenamiento y nada en ellos " +
  "permite escoger entre unas y otras. El modelo puede ajustarse, pero lo que aprende no está " +
  "determinado por la evidencia sino por la inicialización y por el procedimiento de optimización."
));
c.push(nuevo(
  "Con los valores de esta investigación la condición se cumple con amplio margen. La ventana de " +
  "entrenamiento es de 520 semanas y la retrospección de 52, lo que produce 468 secuencias. La " +
  "arquitectura empleada tiene 17 076 parámetros entrenables, verificados contra el conteo que " +
  "reporta la propia biblioteca. La razón resulta de aproximadamente 37 parámetros por cada " +
  "secuencia disponible, de modo que la familia se descarta sin necesidad de entrenarla."
));
c.push(nota(
  "El valor de esta regla no está en el ahorro de una corrida sino en que la decisión se toma a " +
  "partir de una propiedad medible de los datos y no de un resultado experimental. Esa es la " +
  "diferencia entre descartar una familia porque funcionó mal y descartarla porque se puede " +
  "demostrar que no está en condiciones de funcionar."
));

c.push(h2("Fórmulas 12 y 13 · Ruido relativo de Poisson y umbrales de volumen"));
c.push(eq("t_poisson"));
c.push(eq("t_umbral"));
c.push(rot("Donde"));
c.push(donde("λ", "media del conteo semanal, que en una distribución de Poisson coincide con su varianza"));
c.push(donde("√λ", "desviación estándar del conteo, por ser la raíz de la varianza"));
c.push(donde("CV", "coeficiente de variación: desviación estándar dividida entre la media"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La primera expresión deriva el ruido relativo de un conteo de Poisson. La distribución tiene la " +
  "propiedad de que su varianza es igual a su media, de donde su desviación estándar es la raíz de " +
  "la media. El coeficiente de variación, que expresa la desviación estándar como fracción de la " +
  "media, se simplifica entonces al inverso de la raíz de la media."
));
c.push(nuevo(
  "El resultado tiene una consecuencia que es la que interesa al criterio: el ruido relativo " +
  "disminuye al aumentar el volumen. En una serie de pocos casos semanales, la variación aleatoria " +
  "representa una fracción grande del nivel; en una serie de muchos casos, esa misma variación " +
  "aleatoria resulta proporcionalmente pequeña frente al nivel. Dicho de otro modo, el mismo " +
  "mecanismo de notificación produce series proporcionalmente más ruidosas cuanto menor es su " +
  "volumen."
));
c.push(nuevo(
  "La segunda expresión evalúa el ruido relativo en los dos umbrales que el criterio emplea. Con " +
  "diez casos semanales el ruido alcanza el 32 % del nivel de la serie: las variables de rezago " +
  "que alimentan a los modelos tabulares están entonces dominadas por la variabilidad de " +
  "notificación más que por la señal epidemiológica, y un modelo flexible ajustará ese ruido. Con " +
  "cuarenta casos semanales el ruido desciende al 16 %, proporción a partir de la cual la " +
  "estructura no lineal que el modelo flexible explota puede considerarse señal."
));
c.push(nota(
  "Conviene subrayar el origen de estos dos umbrales, porque es lo que los distingue de una " +
  "calibración sobre los datos. No se obtuvieron probando cortes y escogiendo el que mejor " +
  "resultado daba, sino evaluando una propiedad de la distribución de Poisson en dos niveles de " +
  "ruido relativo. Por esa razón los umbrales no dependen de las series de este estudio y pueden " +
  "aplicarse a cualquier otra serie de conteo.",
  "F1F6EE"));

c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h2("Fórmula 14 · Cobertura del criterio"));
c.push(eq("t_cobertura"));
c.push(rot("Donde"));
c.push(donde("C", "cobertura del criterio, expresada como fracción entre cero y uno"));
c.push(donde("N", "número de series sobre las que se evalúa el criterio"));
c.push(donde("s(vᵢ)", "familia que el criterio recomienda para la serie i a partir de su perfil"));
c.push(donde("ℰᵢ", "conjunto de modelos equivalentes al mejor en la serie i, según la fórmula 9"));
c.push(donde("1[·]", "función indicadora, que vale uno si la recomendación pertenece al conjunto"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La cobertura es la métrica principal con la que se evalúa el criterio, y su definición responde " +
  "directamente al hallazgo del apartado 3.8.3. Dado que en la mayoría de las series varias " +
  "familias resultan estadísticamente indistinguibles del mejor, exigir al criterio que acierte " +
  "exactamente cuál tiene el error mínimo no constituye un objetivo alcanzable: equivaldría a " +
  "pedirle que acertara el resultado de un sorteo."
));
c.push(nuevo(
  "Lo que la cobertura mide es si la recomendación cae dentro del conjunto de alternativas " +
  "defendibles. Se cuenta un acierto cuando la familia recomendada pertenece al conjunto de " +
  "modelos equivalentes al mejor, y la cobertura es la proporción de series en las que eso ocurre. " +
  "Una cobertura del cien por ciento significa que el criterio nunca recomendó un modelo " +
  "significativamente peor que el mejor disponible, aunque no necesariamente recomendó el de menor " +
  "error."
));
c.push(nuevo(
  "La métrica tiene una propiedad que conviene explicitar: depende del conjunto de equivalentes y, " +
  "por tanto, del poder del experimento. Un experimento con más orígenes de pronóstico produciría " +
  "conjuntos más pequeños y una cobertura más exigente. Por eso la cobertura debe reportarse " +
  "siempre junto al número de orígenes y al nivel de significación empleados."
));

c.push(h2("Fórmula 15 · Exceso de error de la recomendación"));
c.push(eq("t_regret"));
c.push(rot("Donde"));
c.push(donde("rᵢ", "exceso de error de la recomendación en la serie i, como fracción"));
c.push(donde("MAE_{s(vᵢ)}", "error absoluto medio de la familia recomendada por el criterio"));
c.push(donde("MAE_{m*ᵢ}", "error absoluto medio del mejor modelo de esa serie"));
c.push(donde("r̄", "exceso de error medio sobre las N series evaluadas"));
c.push(rot("Explicación"));
c.push(nuevo(
  "El exceso de error complementa a la cobertura midiendo no si la recomendación es aceptable sino " +
  "cuánto cuesta. Se calcula como la diferencia entre el error de la familia recomendada y el del " +
  "mejor modelo, expresada como fracción de este último. Un valor de cero indica que el criterio " +
  "recomendó precisamente el de menor error; un valor de 0,05 indica que la recomendación erró un " +
  "cinco por ciento más que la mejor alternativa."
));
c.push(nuevo(
  "La expresión en términos relativos y no absolutos responde a que el error se mide en casos y el " +
  "nivel de las series del estudio recorre dos órdenes de magnitud: un exceso de un caso semanal " +
  "es despreciable en Lima y considerable en Tacna. Dividir entre el error del mejor modelo hace " +
  "comparables los excesos entre series."
));
c.push(nota(
  "Esa misma normalización introduce, sin embargo, un efecto de escala que debe declararse. Cuando " +
  "el error de referencia es de aproximadamente un caso semanal, una diferencia absoluta pequeña " +
  "produce un exceso relativo grande. En Moquegua, el único caso en que el criterio falla fuera de " +
  "la muestra de construcción, el exceso del 23,3 % corresponde a una diferencia de 0,22 casos por " +
  "semana. Por esa razón el exceso relativo se reporta acompañado de su valor en casos.",
  "FDF2E0"));

c.push(h2("Fórmula 16 · Ahorro de cómputo"));
c.push(eq("t_ahorro"));
c.push(rot("Donde"));
c.push(donde("A", "ahorro de cómputo, expresado como fracción entre cero y uno"));
c.push(donde("c(m)", "costo de entrenar y validar la familia m bajo el esquema de origen móvil"));
c.push(donde("s(vᵢ)", "familia recomendada para la serie i"));
c.push(donde("ℳ", "conjunto de familias candidatas"));
c.push(rot("Explicación"));
c.push(nuevo(
  "El ahorro cuantifica la razón de ser del criterio. El numerador acumula el costo de entrenar " +
  "únicamente la familia recomendada en cada serie; el denominador, el de entrenar todas las " +
  "familias candidatas en todas las series, que es lo que exige el procedimiento de evaluación " +
  "exhaustiva. El cociente expresa qué fracción del costo total representa la estrategia del " +
  "criterio, y uno menos ese cociente es el ahorro."
));
c.push(nuevo(
  "El costo de cada familia se estima como la media de los tiempos de ejecución observados en la " +
  "corrida experimental, que son de 24,0 segundos para el modelo estadístico, 36,0 para los " +
  "árboles potenciados, 38,3 para el bosque aleatorio y 81,6 para la red recurrente. La diferencia " +
  "entre familias no es despreciable: la más costosa consume más del triple que la más económica."
));
c.push(nuevo(
  "El ahorro medido es del 81,9 % sobre la muestra de construcción y del 81,5 % sobre los " +
  "departamentos de validación. Su magnitud depende de la composición de " +
  "las recomendaciones: cuanto más frecuentemente el criterio recomiende la familia más económica, " +
  "mayor será el ahorro. Conviene por ello reportarlo junto a la cobertura y al exceso de error, " +
  "porque un criterio que recomendara siempre el modelo más barato maximizaría el ahorro a costa " +
  "de las otras dos métricas."
));

// ======================================================================
// 3.11.4
// ======================================================================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("3.11.4  Explicación local y explicación global"));

c.push(h2("Fórmula 17 · Importancia global de una variable"));
c.push(eq("t_shap_global"));
c.push(rot("Donde"));
c.push(donde("Φⱼ", "importancia global de la variable j"));
c.push(donde("φᵢⱼ", "atribución de la variable j en la observación i, según la fórmula del apartado 3.11.1"));
c.push(donde("n", "número de observaciones sobre las que se evalúa la atribución"));
c.push(donde("|·|", "valor absoluto"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La expresión agrega las atribuciones individuales en una medida única por variable. El paso " +
  "determinante es el valor absoluto: se promedia la magnitud de la contribución y no la " +
  "contribución con su signo."
));
c.push(nuevo(
  "La razón es que una variable puede empujar la predicción al alza en unas semanas y a la baja en " +
  "otras. La variable de calendario, por ejemplo, incrementa la predicción durante el invierno y " +
  "la reduce en verano. Si se promediaran las contribuciones con su signo, esas aportaciones " +
  "opuestas se cancelarían y la variable aparecería como irrelevante, cuando en realidad es " +
  "determinante en ambos períodos. Tomando el valor absoluto, lo que se mide es cuánto mueve la " +
  "variable la predicción, con independencia de la dirección."
));
c.push(nuevo(
  "La medida resultante se expresa en las mismas unidades que la predicción —casos semanales— y no " +
  "está normalizada: su magnitud depende del nivel de la serie. Para comparar entre departamentos " +
  "de distinto volumen es necesaria la normalización que introduce la fórmula siguiente."
));

c.push(h2("Fórmula 18 · Atribución agregada por familia de variables"));
c.push(eq("t_shap_familia"));
c.push(rot("Donde"));
c.push(donde("Φ_𝒢", "atribución agregada de la familia de variables 𝒢, como fracción del total"));
c.push(donde("𝒢", "familia de variables: inercia corta, memoria media, rezago anual, ventanas móviles, calendario o tendencia"));
c.push(donde("Φⱼ", "importancia global de la variable j, según la fórmula 17"));
c.push(donde("|F|", "número total de variables del modelo"));
c.push(rot("Explicación"));
c.push(nuevo(
  "La expresión suma las importancias de las variables que pertenecen a una misma familia y divide " +
  "entre la suma de todas las importancias. El resultado es la fracción de la atribución total que " +
  "recae sobre esa familia, de modo que las fracciones de todas las familias suman uno."
));
c.push(nuevo(
  "La agregación cumple dos funciones. La primera es de estabilidad: las variables de una misma " +
  "familia están fuertemente correlacionadas entre sí —los rezagos de una y dos semanas de una " +
  "serie epidemiológica se parecen mucho— y, ante variables correlacionadas, la atribución se " +
  "reparte entre ellas de forma sensible a variaciones menores del modelo. El reparto interno es " +
  "inestable, pero el total de la familia no lo es. La segunda es de comparabilidad: al expresar " +
  "la atribución como fracción del total, la medida deja de depender del nivel de la serie y " +
  "permite contrastar departamentos de volumen muy distinto."
));
c.push(nuevo(
  "Esa comparabilidad es la que hace posible el contraste que da sentido al apartado. Calculada la " +
  "atribución por familia en cada departamento, puede correlacionarse con las propiedades del " +
  "perfil y comprobarse si las reglas del criterio describen un mecanismo o solo una coincidencia " +
  "numérica. Los resultados de ese contraste se presentan en el apartado 4.5."
));

// ======================================================================
// 3.11.6
// ======================================================================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("3.11.6  Limitaciones del análisis — cierre solicitado"));
c.push(p(
  "El comentario de la revisión pide explicitar el alcance que estas limitaciones fijan. El texto " +
  "siguiente se añade como párrafo de cierre del apartado, después de la enumeración de las tres " +
  "limitaciones.", { italics: true }));
c.push(nuevo(
  "El alcance del análisis de atribución queda delimitado por esas tres limitaciones del modo " +
  "siguiente. El análisis se emplea en esta investigación para una única finalidad: verificar si " +
  "los modelos se apoyan en la estructura que el perfil de la serie declara, y por tanto si las " +
  "reglas del criterio de selección describen un mecanismo reconocible o únicamente una regularidad " +
  "numérica. Para esa finalidad las tres limitaciones no son impedimento, porque la pregunta que " +
  "se formula es sobre el comportamiento del modelo y es exactamente eso lo que la atribución mide."
));
c.push(nuevo(
  "Fuera de esa finalidad, el análisis no se emplea y sus resultados no deben leerse en otras " +
  "direcciones. En particular, no se utiliza para identificar factores de riesgo de la neumonía, " +
  "para cuantificar la influencia epidemiológica de ninguna variable ni para orientar " +
  "intervenciones sanitarias. Una atribución elevada del rezago de una semana indica que el modelo " +
  "se apoya en el nivel reciente de la serie, no que el número de casos de una semana cause el de " +
  "la siguiente."
));
c.push(nuevo(
  "Tampoco se emplea para comparar la importancia de variables individuales entre sí, dado que la " +
  "correlación entre rezagos consecutivos hace inestable ese reparto; por esa razón toda la " +
  "interpretación se realiza sobre familias de variables. Y no se formula ninguna afirmación " +
  "contrafáctica a partir de las atribuciones: el análisis no responde qué habría ocurrido con un " +
  "valor distinto de la variable, sino cómo responde la predicción del modelo ante ese cambio."
));
c.push(nuevo(
  "En consecuencia, el componente de explicabilidad de esta investigación tiene alcance de " +
  "instrumento de verificación metodológica y no de instrumento de inferencia epidemiológica. Su " +
  "resultado sostiene o refuta afirmaciones sobre el criterio de selección, que es lo que esta " +
  "tesis propone, y no afirmaciones sobre la enfermedad, que corresponden a un diseño de " +
  "investigación distinto del adoptado aquí."
));

const doc = new Document({
  creator: "Gersson André Huichi Santander",
  title: "Explicación detallada de las fórmulas nuevas",
  styles: {
    default: {
      document: { run: { font: "Arial", size: 21, color: "000000" } },
      heading1: { run: { font: "Arial", size: 24, bold: true } },
    },
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
                          margin: { top: 1300, right: 1300, bottom: 1300, left: 1500 } } },
    children: c,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(process.argv[2], buf);
  console.log("escrito:", process.argv[2], (buf.length / 1024).toFixed(0) + " KB");
});
