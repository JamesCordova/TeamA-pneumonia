const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, PageBreak,
} = require("docx");

const HERE = __dirname;
const REPO = "D:/Ingenieria de Sistemas/Neumonia repo/TeamA-pneumonia";
const EQS = JSON.parse(fs.readFileSync(
  path.join(REPO, "pruebasg/results/tesis/ecuaciones/manifiesto.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(
  path.join(REPO, "pruebasg/results/tesis/datos_capitulos.json"), "utf8"));
const LOC = JSON.parse(fs.readFileSync(
  path.join(REPO, "pruebasg/results/shap/explicaciones_locales.json"), "utf8"));
const FIG_T = path.join(REPO, "pruebasg/results/tesis/figuras");
const FIG_S = path.join(REPO, "pruebasg/results/shap/figuras");

function num(x, d = 2) {
  if (x === null || x === undefined) return "—";
  if (typeof x !== "number") return String(x);
  let [e, dec] = x.toFixed(d).split(".");
  const s = e.startsWith("-") ? "-" : "";
  e = e.replace("-", "").replace(/\B(?=(\d{3})+(?!\d))/g, "\u202F");
  if (dec) dec = dec.replace(/0+$/, "");
  return s + e + (dec ? "," + dec : "");
}
const cap = (s) => s.charAt(0) + s.slice(1).toLowerCase();

function p(t, o = {}) {
  return new Paragraph({
    alignment: o.align || AlignmentType.JUSTIFIED,
    spacing: { after: o.after ?? 130, line: 310 },
    children: [new TextRun({ text: t, bold: !!o.bold, italics: !!o.italics })],
  });
}
function h1(t) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 150 },
    children: [new TextRun({ text: t, bold: true, size: 26, font: "Arial", color: "000000" })],
  });
}
function h2(t) {
  return new Paragraph({
    spacing: { before: 250, after: 60 },
    shading: { type: ShadingType.CLEAR, fill: "1F4E79" },
    indent: { left: 110, right: 110 },
    children: [new TextRun({ text: "  " + t, bold: true, size: 22, color: "FFFFFF" })],
  });
}
function h3(t) {
  return new Paragraph({
    spacing: { before: 190, after: 80 },
    children: [new TextRun({ text: t, bold: true, size: 21, color: "1F4E79" })],
  });
}
function nuevo(t) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 80, after: 125, line: 310 },
    indent: { left: 400, right: 230 },
    shading: { type: ShadingType.CLEAR, fill: "F1F6EE" },
    children: [new TextRun({ text: t, size: 20 })],
  });
}
function bullet(t) {
  return new Paragraph({
    bullet: { level: 0 }, spacing: { after: 70, line: 300 },
    alignment: AlignmentType.JUSTIFIED, children: [new TextRun({ text: t })],
  });
}
function nota(t, fill) {
  return new Paragraph({
    spacing: { before: 120, after: 150 },
    shading: { type: ShadingType.CLEAR, fill: fill || "EEF3F8" },
    indent: { left: 170, right: 170 },
    children: [new TextRun({ text: t, size: 20 })],
  });
}
function eq(id) {
  const m = EQS[id];
  if (!m) throw new Error("falta " + id);
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 130, after: 130 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(m.file),
                              transformation: { width: m.w, height: m.h } })],
  });
}
function figura(dir, file, titulo, ancho = 500, ratio = 0.56) {
  const f = path.join(dir, file);
  if (!fs.existsSync(f)) return [];
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 140, after: 50 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(f),
                                transformation: { width: ancho, height: Math.round(ancho * ratio) } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 150 },
      children: [new TextRun({ text: titulo, size: 18, italics: true })],
    }),
  ];
}
function tabla(widths, enc, filas, titulo, fsz = 16) {
  const W = (i) => widths[i] ?? widths[widths.length - 1];
  const out = [];
  if (titulo) out.push(new Paragraph({
    spacing: { before: 150, after: 55 },
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
  out.push(new Paragraph({ spacing: { after: 110 }, children: [new TextRun({ text: "" })] }));
  return out;
}

const c = [];
const emp = D.empates, per = D.pertenencia, cfg = D.configuracion;
const cin = D.criterio_en_muestra, cout = D.criterio_fuera_muestra;
const eout = cout.estrategias[cout.estrategias.length - 1];
const mejorFija = cout.estrategias.filter((e) => e.estrategia !== "criterio propuesto")
  .sort((a, b) => a.regret_medio - b.regret_medio)[0];
const fallo = (D.fallos_fuera_muestra || [])[0];
const puno = LOC.find((l) => l.departamento === "PUNO") || LOC[0];

// ===================== PORTADA =====================
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 70 },
  children: [new TextRun({ text: "TEXTO PENDIENTE DE INCORPORAR", bold: true, size: 29 })] }));
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({
    text: "Secciones faltantes de los capítulos II, III y IV, más el avance de la semana 6",
    italics: true, size: 21 })] }));
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "Revisión del documento al 20 de octubre de 2026",
                           size: 18, color: "555555" })] }));
c.push(p(
  "Se revisó el documento completo. Lo ya incorporado queda fuera de este material; aquí está " +
  "únicamente lo que falta, en el orden en que debe insertarse, más el texto nuevo que corresponde " +
  "al avance de esta semana."
));
c.push(...tabla([1500, 3400, 3887],
  ["Ubicación", "Sección", "Situación"],
  [
    ["Cap. II", "Posicionamiento frente a la plataforma de referencia", "Nuevo (avance de la semana 6)"],
    ["3.6.4", "Perfil estadístico de la serie", "Falta: el Cap. IV ya lo usa"],
    ["3.7.1", "Codificación circular del calendario", "Falta el ejemplo; la fuga de información sí se incorporó"],
    ["3.8.3", "Conjunto de modelos estadísticamente equivalentes", "Falta: sigue el título antiguo"],
    ["3.9.2", "Visualización comparativa — versión 2.0", "Ampliación (avance de la semana 6)"],
    ["3.10", "Etapa 7: Criterio de selección automática", "Falta: es el aporte de la tesis"],
    ["3.11.4", "Explicación local", "Falta dentro de la sección de explicabilidad"],
    ["3.11.6", "Limitaciones del análisis", "Falta dentro de la sección de explicabilidad"],
    ["4.3", "Conjuntos de modelos equivalentes", "Falta: contiene el hallazgo principal"],
    ["4.4.2", "Validación fuera de la muestra", "Falta: es la evidencia independiente"],
    ["4.5", "Descomposición local de una predicción", "Ampliación (avance de la semana 6)"],
    ["4.6", "Síntesis de los resultados", "Falta"],
  ]));
c.push(nota(
  "Advertencia sobre la numeración: el documento numera como 3.10 la sección de explicabilidad. " +
  "Si se inserta la Etapa 7 antes de ella —que es donde corresponde, porque la explicabilidad " +
  "verifica el criterio y no al revés—, la explicabilidad pasa a 3.11 y sus subapartados se " +
  "renumeran. Este material ya emplea esa numeración.",
  "FDF2E0"));

// ===================== CAP II =====================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h2("CAPÍTULO II — Posicionamiento frente a la plataforma de referencia"));
c.push(p(
  "Texto nuevo, derivado del análisis de los repositorios de la colaboración realizado esta semana. " +
  "Se sugiere incorporarlo al final del Estado del Arte, antes del párrafo que formula la brecha."
));
c.push(nuevo(
  "El grupo de investigación en el que se inscribe esta tesis mantiene, junto con el Data Institute " +
  "for Societal Challenges de la Universidad de Oklahoma, una plataforma de vigilancia y pronóstico " +
  "de neumonía para el Perú. Esa plataforma constituye el antecedente más próximo al presente " +
  "trabajo y conviene delimitar con precisión qué aporta y qué deja abierto."
));
c.push(nuevo(
  "La plataforma implementa trece modelos de pronóstico más un ensamble por mediana, con " +
  "calibración de intervalos, sobre 106 series correspondientes a tres medidas, dos grupos de edad " +
  "y los niveles nacional y departamental. Para cada serie ejecuta todos los modelos elegibles y " +
  "señala como recomendado el de mejor desempeño en el conjunto de validación histórica. Incorpora " +
  "además contrastes de significación entre el modelo ganador y el inmediato siguiente."
));
c.push(nuevo(
  "Dos elementos de ese diseño son directamente pertinentes. El primero es que la plataforma asigna " +
  "a cada serie un nivel según su recuento medio semanal —regular, de recuento bajo o dispersa— y " +
  "ese nivel determina qué familias de modelo se ejecutan sobre ella: los modelos estadísticos y de " +
  "aprendizaje automático se reservan para las series de nivel regular, mientras que las series " +
  "dispersas reciben únicamente modelos de referencia. Es, por tanto, una regla de umbral sobre el " +
  "volumen que condiciona qué modelos se aplican, y constituye el antecedente directo de las reglas " +
  "de volumen del criterio propuesto en esta investigación."
));
c.push(nuevo(
  "El segundo es que, superado ese filtro, la selección del modelo se resuelve ejecutando todas las " +
  "alternativas y comparando su desempeño. El filtro responde a una cuestión de viabilidad —qué " +
  "modelos pueden ajustarse sobre una serie de ese tamaño— y no a una predicción de cuál resultará " +
  "adecuado. La pregunta de qué familia conviene emplear sigue resolviéndose por evaluación " +
  "exhaustiva."
));
c.push(nuevo(
  "La brecha que esta investigación aborda se delimita así con mayor precisión. No consiste en " +
  "introducir la idea de condicionar la elección del modelo a propiedades de la serie, que ya está " +
  "presente en la plataforma como criterio de viabilidad. Consiste en convertir esa heurística de " +
  "implementación en un criterio explícito: derivar los umbrales de una propiedad estadística de los " +
  "datos de conteo en lugar de fijarlos por observación, extender la decisión de la viabilidad a la " +
  "elección propiamente dicha, y someter el criterio resultante a una evaluación que mida su acierto " +
  "y su costo. Ninguno de esos tres elementos está presente en el antecedente."
));
c.push(...tabla([2100, 3300, 3387],
  ["", "Plataforma de referencia", "Criterio propuesto"],
  [
    ["Qué decide la regla de volumen", "Qué modelos es viable ejecutar", "Qué familia corresponde emplear"],
    ["Procedimiento posterior", "Ejecutar todos los elegibles y comparar", "Ejecutar solo la familia recomendada"],
    ["Origen de los umbrales", "Observación de los datos disponibles", "Ruido relativo de un conteo de Poisson"],
    ["Evaluación de la regla", "No se evalúa como criterio", "Cobertura, exceso de error y ahorro, dentro y fuera de la muestra"],
    ["Explicabilidad", "Registrada como pendiente, no implementada", "Atribución global y local implementada y usada como verificación"],
  ]));
c.push(p(
  "Nota para el autor, no para el documento: la última fila se verificó en el repositorio de la " +
  "plataforma, cuyo plan de trabajo registra la importancia de variables y SHAP con el estado «not " +
  "built yet; no feature importance is stored», y cuya lista de dependencias no incluye la " +
  "biblioteca correspondiente.", { italics: true }
));

// ===================== 3.6.4 =====================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h2("CAPÍTULO III — Secciones faltantes"));
c.push(h1("3.6.4  Perfil estadístico de la serie"));
c.push(p("Se inserta al final de la Etapa 3. Es la sección de la que depende todo el Capítulo IV.",
         { italics: true }));
c.push(nuevo(
  "El análisis estadístico descrito en los apartados anteriores se consolida en un descriptor único " +
  "de la serie, al que se denomina perfil. El perfil es un vector de metafeatures —medidas resumen " +
  "calculadas sobre la propia serie— que la caracteriza con independencia del departamento al que " +
  "corresponde y de la enfermedad que registra. Esta independencia es deliberada y constituye la " +
  "condición que permite que el criterio construido sobre el perfil sea aplicable a otras series de " +
  "notificación obligatoria."
));
c.push(eq("t_perfil"));
c.push(nuevo(
  "La construcción del perfil sigue la línea de caracterización de series temporales mediante " +
  "atributos estadísticos propuesta por Wang, Smith e Hyndman (2006), y adopta de ella las medidas " +
  "de fuerza de tendencia y de estacionalidad. A ese conjunto se añaden descriptores propios de los " +
  "datos de conteo con los que trabaja esta investigación —el índice de dispersión y la proporción " +
  "de semanas sin casos—, que no forman parte del conjunto estándar pero resultan determinantes en " +
  "series de vigilancia epidemiológica de bajo volumen."
));
c.push(h3("Fuerzas de tendencia y de estacionalidad"));
c.push(nuevo(
  "A partir de la descomposición descrita en el apartado 3.5.3 se cuantifica cuánta de la " +
  "variabilidad de la serie explica cada componente. Ambas medidas comparan la varianza del residuo " +
  "contra la varianza de la componente respectiva sumada al residuo, de modo que quedan acotadas " +
  "entre cero y uno: un valor próximo a cero indica que la componente no aporta información sobre " +
  "el residuo, y un valor próximo a uno que domina el comportamiento de la serie."
));
c.push(eq("t_ft"));
c.push(eq("t_fs"));
c.push(h3("Índice de dispersión"));
c.push(nuevo(
  "Las notificaciones semanales son conteos. Bajo el supuesto de un proceso de Poisson la varianza " +
  "sería igual a la media, de modo que el cociente entre ambas mide directamente el exceso de " +
  "variabilidad respecto de ese supuesto de referencia. Un valor igual a uno corresponde a " +
  "variabilidad puramente poissoniana; valores superiores indican sobredispersión, atribuible a " +
  "agrupamiento de casos, a brotes o a irregularidades de la notificación."
));
c.push(eq("t_disp"));
c.push(h3("Entropía espectral y proporción de semanas sin casos"));
c.push(nuevo(
  "La entropía espectral normalizada mide cuánta estructura aprovechable contiene la serie. Se " +
  "calcula sobre la densidad espectral normalizada a una distribución de probabilidad: un espectro " +
  "concentrado en pocas frecuencias, propio de una serie regular, produce entropía baja, mientras " +
  "que el ruido blanco produce un valor próximo a la unidad. La proporción de semanas sin casos " +
  "completa el perfil y resulta necesaria en las series de bajo volumen, donde condiciona tanto la " +
  "elección del modelo como la aplicabilidad de las métricas de error relativo."
));
c.push(eq("t_ent"));
c.push(eq("t_p0"));

// ===================== 3.7.1 =====================
c.push(h1("3.7.1  Complemento: codificación circular del calendario"));
c.push(p("Sustituye al párrafo que enuncia la codificación circular sin ejemplo.", { italics: true }));
c.push(nuevo(
  "La semana del año no puede incorporarse al modelo como un número ordinal. Los modelos tabulares " +
  "interpretan la cercanía entre valores como cercanía entre situaciones, y en esa escala la semana " +
  "52 y la semana 1 —que son consecutivas en el tiempo— quedan separadas por 51 unidades, la máxima " +
  "distancia posible. El modelo aprendería entonces una discontinuidad artificial en el cambio de " +
  "año, justamente en el período de mayor incidencia de la neumonía."
));
c.push(nuevo(
  "Para evitarlo, la semana del año se proyecta sobre una circunferencia mediante un par de " +
  "funciones seno y coseno, de modo que la distancia entre dos semanas refleje su proximidad real " +
  "en el ciclo anual. En esa representación la semana 52 dista de la semana 1 exactamente 0,1208, " +
  "el mismo valor que separa la semana 1 de la semana 2, mientras que la semana 26 —el punto " +
  "opuesto del año— alcanza 1,996, que es el diámetro de la circunferencia."
));
c.push(...tabla([1600, 1800, 1800, 2000, 1587],
  ["Semana", "sin(2πw/52)", "cos(2πw/52)", "Distancia a la semana 1", "Distancia ordinal"],
  [["51", "−0,1205", "0,9927", "0,2411", "50"],
   ["52", "0,0000", "1,0000", "0,1208", "51"],
   ["1", "0,1205", "0,9927", "0,0000", "0"],
   ["2", "0,2393", "0,9709", "0,1208", "1"],
   ["26", "−0,0000", "−1,0000", "1,9964", "25"]],
  "Tabla. Representación circular de la semana del año frente a la escala ordinal"));

// ===================== 3.8.3 =====================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("3.8.3  Determinación del conjunto de modelos estadísticamente equivalentes"));
c.push(p("Sustituye íntegramente al apartado titulado «Selección del modelo óptimo».", { italics: true }));
c.push(nuevo(
  "A partir de los resultados de la validación se identifica, para cada serie, el modelo de menor " +
  "error y el conjunto de modelos cuyo error no resulta estadísticamente distinguible del suyo. El " +
  "cambio respecto de una selección por valor puntual del error responde a una razón metodológica: " +
  "la evidencia obtenida muestra que en la mayoría de las series la diferencia entre los modelos " +
  "mejor situados es inferior al error del propio estimador, de modo que declarar un ganador " +
  "equivaldría a ordenar los modelos por ruido muestral."
));
c.push(nuevo(
  "La comparación se realiza de forma pareada por origen de pronóstico: ambos modelos se evalúan " +
  "sobre exactamente las mismas semanas, de manera que la dificultad intrínseca de cada semana se " +
  "cancela al tomar la diferencia."
));
c.push(eq("t_dif"));
c.push(nuevo(
  "Bajo la hipótesis nula de que ambos modelos tienen el mismo error esperado, el estadístico de la " +
  "prueba t para muestras pareadas sigue una distribución t de Student con K − 1 grados de " +
  "libertad, siendo K el número de orígenes. Con horizonte de una semana y desplazamiento de cuatro " +
  "semanas, las semanas evaluadas en orígenes sucesivos no se solapan, lo que respalda el " +
  "tratamiento de las diferencias como observaciones independientes."
));
c.push(eq("t_t"));
c.push(nuevo(
  "El conjunto de modelos estadísticamente equivalentes al mejor queda definido como aquellos cuya " +
  "comparación contra él no permite rechazar la hipótesis nula, junto con el propio mejor modelo. " +
  "La selección se realiza entonces dentro de ese conjunto, atendiendo a la consistencia del " +
  "desempeño entre horizontes, al costo computacional y a la interpretabilidad, y no al valor " +
  "puntual del error."
));
c.push(eq("t_empate"));

// ===================== 3.9.2 =====================
c.push(h1("3.9.2  Complemento: tablero de resultados, versión 2.0"));
c.push(p("Ampliación del apartado de visualización, correspondiente al avance de esta semana.",
         { italics: true }));
c.push(nuevo(
  "El conjunto de representaciones se integra en un tablero autónomo, construido como un único " +
  "archivo sin dependencias externas, que puede abrirse sin servidor ni conexión. Su segunda " +
  "versión incorpora, además del mapa coroplético y de la comparación de modelos ya descritos, dos " +
  "paneles que corresponden a los componentes propios de esta investigación."
));
c.push(nuevo(
  "El panel del criterio de selección sitúa cada departamento sobre el plano definido por el " +
  "volumen medio y la fuerza estacional, es decir por su perfil y no por su identificación " +
  "geográfica, y muestra las regiones de decisión de las reglas junto con la familia recomendada " +
  "para cada serie y si esa familia pertenece al conjunto de modelos estadísticamente equivalentes " +
  "al mejor. El panel de explicabilidad presenta la atribución global agregada por familia de " +
  "variables y la descomposición local de una predicción concreta."
));
c.push(nuevo(
  "El panel de modelado incorpora asimismo una banda de incertidumbre al ochenta por ciento. Debe " +
  "precisarse su naturaleza: no es un intervalo de predicción del modelo —los modelos de árboles no " +
  "lo producen— sino el rango central del ochenta por ciento de los errores que ese mismo modelo " +
  "cometió durante la validación. Es una medida de dispersión observada y así se declara en el " +
  "propio tablero."
));

// ===================== ETAPA 7 =====================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("3.10  Etapa 7: Criterio de selección automática de modelo"));
c.push(p(
  "Sección que falta por completo. Debe insertarse después de la Etapa 6 y antes de la " +
  "explicabilidad. Requiere además actualizar el apartado 3.1, que enuncia seis etapas, y la " +
  "Figura 2 del esquema.", { italics: true }
));
c.push(nuevo(
  "Las etapas anteriores producen, para cada serie, la comparación de todas las familias de modelo " +
  "bajo un mismo esquema de validación. Ese procedimiento resuelve la elección para las series " +
  "evaluadas, pero no es reutilizable: aplicado a una serie nueva obliga a repetir la comparación " +
  "completa, con un costo que crece linealmente con el número de series a pronosticar. La séptima " +
  "etapa aborda ese problema construyendo un criterio que determina la familia de modelo a partir " +
  "del perfil de la serie, sin evaluarlas todas."
));
c.push(h3("3.10.1  Formulación"));
c.push(nuevo(
  "El criterio se formaliza como una función que asigna a cada perfil una familia de modelo del " +
  "conjunto de familias disponibles."
));
c.push(eq("t_selector"));
c.push(nuevo(
  "Dos propiedades definen su alcance. La primera es que el dominio de la función es el perfil de " +
  "la serie y no su identificación: el criterio no recibe el nombre del departamento ni el de la " +
  "enfermedad, de modo que su aplicabilidad no queda restringida al caso de estudio. La segunda es " +
  "que el recorrido son familias de modelo y no configuraciones de hiperparámetros: el criterio " +
  "decide qué familia emplear, y el ajuste dentro de la familia elegida sigue el procedimiento " +
  "descrito en el apartado 3.7.5."
));
c.push(h3("3.10.2  Reglas de decisión"));
c.push(nuevo(
  "El criterio se implementa mediante un conjunto ordenado de reglas sobre el perfil, que se " +
  "evalúan en secuencia hasta que una se satisface. Antes de ellas opera una regla de descarte, que " +
  "elimina familias del conjunto de candidatas cuando el perfil permite anticipar que no son " +
  "aplicables. La regla de descarte compara el número de secuencias de entrenamiento que la ventana " +
  "permite construir contra el número de parámetros de la red recurrente."
));
c.push(eq("t_descarte"));
c.push(nuevo(
  "Con los valores de esta investigación —ventana de 520 semanas, retrospección de 52 y 17 076 " +
  "parámetros— la razón resulta de aproximadamente uno a treinta y siete en favor de los " +
  "parámetros, lo que sitúa a la red en régimen de sobreparametrización y justifica descartarla sin " +
  "entrenarla."
));
c.push(nuevo(
  "Las reglas de selección se apoyan en dos propiedades del perfil: el volumen medio de casos y la " +
  "fuerza estacional. Los umbrales de volumen no se ajustaron a los datos, sino que se derivan del " +
  "ruido relativo de un conteo de Poisson, que es el inverso de la raíz de su media."
));
c.push(eq("t_poisson"));
c.push(eq("t_umbral"));
c.push(nuevo(
  "El umbral inferior corresponde al volumen por debajo del cual la variabilidad de notificación " +
  "supera un tercio del nivel de la serie, situación en la que las variables de rezago que alimentan " +
  "a los modelos tabulares están dominadas por ruido. El umbral superior corresponde al volumen a " +
  "partir del cual el ruido relativo cae por debajo de la sexta parte del nivel, y la estructura no " +
  "lineal que los modelos flexibles explotan puede considerarse señal."
));
c.push(...tabla([700, 2500, 2100, 3487],
  ["Id", "Condición sobre el perfil", "Familia recomendada", "Fundamento"],
  [["R0", "n_sec < N_par", "descarta aprendizaje profundo",
    "Sobreparametrización verificable antes de entrenar."],
   ["R1", "ȳ ≥ 40 y F_S < 0,60", "Árboles potenciados",
    "Ruido relativo inferior al 16 %: la estructura no lineal es aprovechable."],
   ["R2", "F_S ≥ 0,60", "SARIMA con Fourier",
    "La componente estacional domina la varianza y se representa con 20 parámetros."],
   ["R3", "ȳ < 10", "SARIMA con Fourier",
    "Ruido relativo superior al 32 %: los rezagos no son informativos."],
   ["R4", "en otro caso", "Bosque aleatorio",
    "Alternativa de menor sensibilidad al ajuste de hiperparámetros."]],
  "Tabla. Reglas de decisión del criterio de selección"));
c.push(h3("3.10.3  Métricas de evaluación del criterio"));
c.push(nuevo(
  "El criterio se evalúa con métricas propias, distintas de las que evalúan a los modelos. La razón " +
  "es que, dado que en la mayoría de las series varias familias resultan estadísticamente " +
  "equivalentes, exigir al criterio que acierte el error mínimo exacto no constituye un objetivo " +
  "alcanzable ni informativo. La cobertura es la métrica principal: mide la proporción de series en " +
  "las que la familia recomendada pertenece al conjunto de modelos estadísticamente equivalentes al " +
  "mejor."
));
c.push(eq("t_cobertura"));
c.push(nuevo(
  "El exceso de error mide cuánto error adicional impone la recomendación respecto del mejor modelo " +
  "disponible, y el ahorro de cómputo la reducción del costo de entrenamiento que se obtiene al " +
  "entrenar únicamente la familia recomendada en lugar de evaluar todas las candidatas."
));
c.push(eq("t_regret"));
c.push(eq("t_ahorro"));

// ===================== 3.11.4 y 3.11.6 =====================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("3.11.4  Explicación local y explicación global"));
c.push(p("Falta dentro de la sección de explicabilidad. Corresponde al avance de esta semana.",
         { italics: true }));
c.push(nuevo(
  "La atribución se obtiene para cada observación por separado, lo que constituye una explicación " +
  "local: permite responder por qué el modelo pronosticó un valor determinado en una semana " +
  "determinada, que es la pregunta que formula quien recibe una alerta epidemiológica. Agregando el " +
  "valor absoluto de las atribuciones sobre el conjunto de observaciones se obtiene la importancia " +
  "global de cada variable, que responde a qué variables sostienen el comportamiento del modelo en " +
  "su conjunto. Ambas lecturas son necesarias y responden a preguntas distintas."
));
c.push(eq("t_shap_global"));
c.push(nuevo(
  "Dado que las variables construidas por la ingeniería de características se agrupan naturalmente " +
  "por su significado —rezagos de corto plazo, rezago anual, estadísticos móviles, variables de " +
  "calendario y tendencia—, la atribución se agrega además por familia de variables. Esta " +
  "agregación resulta más estable que la atribución variable por variable, porque las variables de " +
  "una misma familia están fuertemente correlacionadas entre sí y el reparto entre ellas es " +
  "sensible a variaciones menores del modelo, mientras que el total de la familia no lo es."
));
c.push(eq("t_shap_familia"));
c.push(nuevo(
  "La propiedad de eficiencia enunciada en el apartado 3.11.1 proporciona una verificación " +
  "inmediata de la implementación: la suma de las contribuciones de todas las variables debe " +
  "reproducir exactamente la diferencia entre la predicción y el valor base. Esa comprobación se " +
  "ejecuta en cada explicación generada."
));
c.push(h1("3.11.6  Limitaciones del análisis"));
c.push(nuevo(
  "Tres limitaciones deben declararse. La primera es que el análisis explica el modelo y no el " +
  "fenómeno: una atribución elevada indica que el modelo se apoya en esa variable, no que la " +
  "variable determine epidemiológicamente la incidencia. La segunda es que, ante variables " +
  "correlacionadas entre sí —condición habitual en rezagos consecutivos de una serie temporal—, la " +
  "atribución se reparte entre ellas de forma que puede subestimar la relevancia de cada una por " +
  "separado; ésta es la razón por la que el análisis se interpreta por familias y no por variables " +
  "individuales. La tercera es que la atribución no es causal: no responde qué ocurriría si la " +
  "variable tomara otro valor en la realidad, sino cómo cambia la predicción del modelo ante ese " +
  "cambio."
));

// ===================== CAP IV =====================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h2("CAPÍTULO IV — Secciones faltantes"));
c.push(h1("4.3  Conjuntos de modelos estadísticamente equivalentes"));
c.push(p("Falta. Contiene el hallazgo principal de la investigación.", { italics: true }));
c.push(nuevo(
  `La aplicación del procedimiento descrito en el apartado 3.8.3 modifica sustancialmente la ` +
  `lectura de los resultados anteriores. En ${emp.n_con_3_o_mas} de los ${emp.n_total} ` +
  `departamentos, tres o más familias de modelo producen errores que no son estadísticamente ` +
  `distinguibles entre sí al nivel de significación adoptado. El número medio de modelos ` +
  `equivalentes al mejor es de ${num(emp.media_empatados, 2)}, y únicamente en ` +
  `${emp.n_con_ganador_unico} departamento existe un ganador estadísticamente único: ` +
  `${emp.dept_ganador_unico.map(cap).join(", ")}.`
));
c.push(...figura(FIG_T, "cap4_conjunto_empatado.png",
  "Figura. Distribución del número de modelos estadísticamente indistinguibles del mejor.", 470, 0.56));
c.push(nota(
  "Este es el resultado que condiciona la interpretación de toda la comparación. El ordenamiento de " +
  "modelos por valor puntual del error describe en su mayor parte variación muestral y no " +
  "diferencias reales de capacidad predictiva. En consecuencia, la pregunta por cuál es el mejor " +
  "modelo no admite una respuesta estable, y la contribución de la investigación debe situarse en " +
  "las decisiones que sí resultan distinguibles."
));
c.push(...tabla([2600, 1900, 2100, 2187],
  ["Familia", "Pertenece al conjunto", "Proporción", "Lectura"],
  [["Árboles potenciados", `${per.XGBoost}/${emp.n_total}`,
    `${num(100 * per.XGBoost / emp.n_total, 1)} %`, "Nunca significativamente peor que el mejor"],
   ["Bosque aleatorio", `${per.RandomForest}/${emp.n_total}`,
    `${num(100 * per.RandomForest / emp.n_total, 1)} %`, "Equivalente al mejor en casi todas las series"],
   ["SARIMA con Fourier", `${per.SARIMA}/${emp.n_total}`,
    `${num(100 * per.SARIMA / emp.n_total, 1)} %`, "Suficiente en la mayoría de las series"],
   ["Red recurrente", `${per.LSTM}/${emp.n_total}`,
    `${num(100 * per.LSTM / emp.n_total, 1)} %`, "Significativamente peor casi siempre"]],
  "Tabla. Pertenencia de cada familia al conjunto de modelos equivalentes al mejor"));
c.push(nuevo(
  `De esta tabla se desprenden las dos decisiones que el criterio puede sostener. La primera es el ` +
  `descarte de la familia de aprendizaje profundo: queda fuera del conjunto de equivalentes en ` +
  `${emp.n_total - per.LSTM} de las ${emp.n_total} series, de modo que excluirla a partir del ` +
  `perfil, antes de entrenarla, evita la familia más costosa sin pérdida de exactitud. La segunda ` +
  `es que el modelo estadístico —el más económico de ajustar, el único que proporciona intervalos ` +
  `de predicción y el de parámetros interpretables— pertenece al conjunto de equivalentes en ` +
  `${per.SARIMA} de las ${emp.n_total} series, por lo que recomendarlo en ellas no supone pérdida ` +
  `medible de exactitud.`
));
c.push(nuevo(
  "Este hallazgo contrasta con lo que la literatura revisada en el Capítulo II permitiría anticipar. " +
  "Los trabajos que comparan familias de modelos sobre series epidemiológicas reportan " +
  "ordenamientos y concluyen la superioridad de una u otra, pero rara vez acompañan esa conclusión " +
  "de un contraste de significación sobre la diferencia observada. El presente resultado sugiere " +
  "que parte de esos ordenamientos podría no ser distinguible del ruido del propio estimador."
));

// ----- 4.4.2 -----
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("4.4.2  Validación fuera de la muestra de construcción"));
c.push(p("Falta. Es la única medición no contaminada por el diseño de las reglas.", { italics: true }));
c.push(nuevo(
  `El criterio se construyó observando los resultados de los ${cin.n} departamentos de la muestra ` +
  `estratificada. Se aplica ahora, sin modificación alguna, a los ${cout.n} departamentos ` +
  `restantes, cuyos resultados no intervinieron en el diseño de las reglas.`
));
c.push(...tabla([1700, 950, 850, 800, 1600, 1600, 700, 587],
  ["Departamento", "Casos/sem", "F_S", "Regla", "Recomendado", "Mejor observado", "En conj.", "Exceso"],
  cout.filas.map((r) => [cap(r.departamento), num(r.media_casos, 1), num(r.fuerza_estacional, 3),
                         r.regla, r.recomendado, r.mejor_real, r.dentro ? "sí" : "no",
                         `${num(r.regret_pct, 2)} %`]),
  "Tabla. Aplicación del criterio a los departamentos no empleados en su construcción", 15));
c.push(...tabla([2600, 1700, 2100, 2387],
  ["Estrategia", "Cobertura", "Exceso de error medio", "Exceso máximo"],
  cout.estrategias.map((e) => [e.estrategia, e.cobertura,
                               `${num(e.regret_medio, 2)} %`, `${num(e.regret_max, 2)} %`]),
  "Tabla. El criterio frente a estrategias de modelo fijo, fuera de la muestra de construcción"));
c.push(nota(
  `Fuera de la muestra de construcción el criterio alcanza una cobertura de ${eout.cobertura} ` +
  `(${num(eout.cobertura_pct, 1)} %) con un exceso de error medio de ` +
  `${num(eout.regret_medio, 2)} %. La estrategia consistente en emplear siempre la misma familia ` +
  `—${mejorFija.estrategia.replace("siempre ", "")}— obtiene sobre esas mismas series una ` +
  `cobertura de ${mejorFija.cobertura} y un exceso medio de ${num(mejorFija.regret_medio, 2)} %. ` +
  `El criterio de reglas discretas, en la formulación evaluada, no supera a una estrategia de ` +
  `modelo fijo en exactitud.`,
  "FDF2E0"));
if (fallo) {
  c.push(nuevo(
    `El fallo se concentra en una sola serie y una sola regla. En ${cap(fallo.departamento)}, con un ` +
    `volumen de ${num(fallo.media_casos, 1)} casos semanales, la regla R3 recomienda el modelo ` +
    `estadístico donde el bosque aleatorio resulta significativamente mejor. La misma regla acierta ` +
    `en las restantes series de volumen comparable del estudio —Tacna, Tumbes y Madre de Dios—, en ` +
    `las que el modelo estadístico sí pertenece al conjunto de equivalentes. El volumen por sí solo, ` +
    `por tanto, no discrimina el caso en que la regla falla, lo que indica que requiere una ` +
    `condición adicional que las metafeatures restantes del perfil podrían aportar.`
  ));
  c.push(nuevo(
    `Debe consignarse que el exceso del ${num(fallo.regret_pct, 2)} % en esa serie corresponde a una ` +
    `diferencia absoluta de ${num(fallo.regret_casos, 2)} casos por semana. El exceso relativo se ` +
    `amplifica cuando el error de referencia es del orden de un caso semanal, por el mismo efecto ` +
    `de escala que afecta al error porcentual absoluto medio. Esta observación no anula el fallo ` +
    `—la diferencia es estadísticamente significativa y la regla erró— pero indica que el exceso ` +
    `relativo no debe constituir la única métrica de evaluación del criterio.`
  ));
}
c.push(h3("4.4.3  Lo que la evidencia sostiene"));
c.push(nuevo(
  `De la evaluación anterior se desprende que la contribución del criterio no reside en una mejora ` +
  `de exactitud sobre una estrategia de modelo fijo, afirmación que la evidencia no respalda. ` +
  `Reside en tres resultados distintos: el descarte de la familia de aprendizaje profundo a partir ` +
  `del perfil, correcto en ${emp.n_total - per.LSTM} de las ${emp.n_total} series y que evita la ` +
  `familia de mayor costo de entrenamiento; la identificación de las ${per.SARIMA} series en las ` +
  `que el modelo interpretable resulta suficiente; y un ahorro de cómputo del ` +
  `${num(cout.ahorro_pct, 1)} % respecto de evaluar todas las familias.`
));

// ----- 4.5 local -----
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("4.5  Complemento: descomposición local de una predicción"));
c.push(p("Ampliación del apartado de verificación del mecanismo, avance de esta semana.",
         { italics: true }));
c.push(nuevo(
  "Al contraste global presentado en el apartado anterior se añade la lectura local, que descompone " +
  "una predicción concreta en las contribuciones de cada variable. Su valor no es ilustrativo sino " +
  "de verificación: permite comprobar que el comportamiento agregado que describen las " +
  "correlaciones se sostiene también en los episodios que más importan para la vigilancia, que son " +
  "los picos epidémicos."
));
c.push(nuevo(
  `La Figura siguiente presenta la semana de mayor incidencia de ${cap(puno.departamento)}, la del ` +
  `${puno.semana}, correspondiente al pico de invierno altiplánico. El modelo parte de un valor ` +
  `base de ${num(puno.valor_base, 2)} casos —su predicción media sobre el conjunto de ` +
  `entrenamiento— y lo eleva hasta ${num(puno.prediccion, 2)} mediante las contribuciones de las ` +
  `variables.`
));
c.push(...figura(FIG_S, `${puno.departamento.replace(/ /g, "_")}_RandomForest_local.png`,
  `Figura. Descomposición de la predicción de ${cap(puno.departamento)}, semana del ${puno.semana}.`,
  470, 0.68));
c.push(...tabla([2600, 1700, 1900, 2587],
  ["Variable", "Valor", "Contribución", "Familia"],
  puno.contribuciones.slice(0, 6).map((x) => [
    x.variable, String(x.valor_variable),
    (x.contribucion > 0 ? "+" : "") + num(x.contribucion, 2),
    x.familia.replace(/_/g, " ")]),
  "Tabla. Principales contribuciones a la predicción de la semana analizada"));
c.push(nuevo(
  "La lectura es directa y coherente con el mecanismo identificado en el contraste global: la " +
  "predicción elevada se explica sobre todo por la inercia reciente —los casos de las semanas " +
  "inmediatamente anteriores y su promedio móvil— y, de forma secundaria, por la variable de " +
  "calendario, que empuja al alza en julio por corresponder al invierno altiplánico. El modelo no " +
  "está reaccionando a un patrón espurio, sino a la combinación de nivel reciente y estacionalidad " +
  "que el perfil de la serie declara."
));
c.push(nuevo(
  "En cada explicación generada se verifica la propiedad de eficiencia: la suma de las " +
  "contribuciones reproduce exactamente la diferencia entre la predicción y el valor base. Se " +
  "produjeron las explicaciones locales de los once departamentos de la muestra estratificada, cada " +
  "una sobre su semana de mayor incidencia."
));

// ----- 4.6 -----
c.push(h1("4.6  Síntesis de los resultados"));
c.push(p("Falta. Cierra el capítulo.", { italics: true }));
c.push(nuevo("Los resultados obtenidos se resumen en cinco puntos."));
c.push(bullet(
  "Las series departamentales de neumonía presentan una heterogeneidad marcada: el volumen recorre " +
  "dos órdenes de magnitud y la fuerza estacional separa con claridad las regiones naturales, con " +
  "la sierra mostrando el comportamiento más regular."
));
c.push(bullet(
  `En ${emp.n_con_3_o_mas} de los ${emp.n_total} departamentos, tres o más familias de modelo ` +
  `producen errores estadísticamente indistinguibles entre sí. El ordenamiento por valor puntual ` +
  `del error describe en su mayor parte variación muestral.`
));
c.push(bullet(
  `La familia de aprendizaje profundo queda fuera del conjunto de modelos equivalentes al mejor en ` +
  `${emp.n_total - per.LSTM} de las ${emp.n_total} series, resultado que la longitud de serie ` +
  `disponible permite anticipar a partir del perfil.`
));
c.push(bullet(
  `El criterio de selección propuesto alcanza una cobertura del ${num(eout.cobertura_pct, 1)} % ` +
  `fuera de su muestra de construcción, con un ahorro de cómputo del ${num(cout.ahorro_pct, 1)} %, ` +
  `pero no supera en exactitud a una estrategia de modelo fijo. Su formulación mediante reglas ` +
  `discretas resulta insuficiente y requiere sustituirse por un clasificador sobre el vector ` +
  `completo de metafeatures.`
));
c.push(bullet(
  "El análisis de atribución verifica, en su lectura global y en la local, el mecanismo sobre el " +
  "que se apoyan las reglas de volumen, y no el de la regla de estacionalidad; revela además una " +
  "limitación corregible en el conjunto de características de los árboles potenciados."
));

const doc = new Document({
  creator: "Gersson André Huichi Santander",
  title: "Texto pendiente de incorporar",
  styles: {
    default: {
      document: { run: { font: "Arial", size: 21, color: "000000" } },
      heading1: { run: { font: "Arial", size: 26, bold: true, color: "000000" } },
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
