/*
 * Tablero v2.0 — lógica de la interfaz.
 *
 * Todo se dibuja a mano sobre SVG, sin biblioteca de gráficos ni de mapas. No
 * es una decisión estética: el prototipo tiene que abrirse desde el disco, sin
 * servidor ni conexión, para poder entregarse y grabarse; una dependencia
 * externa rompería eso y una incrustada multiplicaría por seis el peso del
 * archivo.
 *
 * Organización:
 *   1. utilidades            formato, fechas, colores
 *   2. estado                lo que el usuario ha elegido
 *   3. derivaciones          lo que se calcula a partir del estado
 *   4. motor de gráficos     ejes, líneas, bandas, lectura al pasar el ratón
 *   5. mapa
 *   6. cinta de titulares
 *   7. controles
 *   8. panel de pronóstico
 *   9. panel de contexto     seis pestañas, dos de ellas propias de la tesina
 *  10. arranque
 */
(function () {
'use strict';

const D = window.__DATOS__;
const NS = 'http://www.w3.org/2000/svg';

/* ───────────────────────── 1. utilidades ───────────────────────── */

/** Número con separador de millar y coma decimal, a la española. */
function fmt(v, dec) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  dec = dec === undefined ? 0 : dec;
  const s = Math.abs(v).toFixed(dec).split('.');
  s[0] = s[0].replace(/\B(?=(\d{3})+(?!\d))/g, '.');
  return (v < 0 ? '−' : '') + s.join(',');
}
function pct(v, dec) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  return (v > 0 ? '+' : '') + fmt(v, dec === undefined ? 0 : dec) + ' %';
}
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}
function svg(tag, attrs) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
  return n;
}
function $(sel) { return document.querySelector(sel); }
/**
 * Aplaza el dibujo hasta que el nodo esté en el documento y tenga medidas.
 *
 * Con requestAnimationFrame no bastaba: el navegador no lo dispara en una
 * pestaña oculta, de modo que un panel abierto en segundo plano se quedaba
 * en blanco hasta volver a él.
 */
function luego(fn) { setTimeout(fn, 0); }
function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
const MESES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic'];
function fechaCorta(iso) {
  const p = iso.split('-');
  return p[2] + ' ' + MESES[+p[1] - 1] + ' ' + p[0];
}
function ms(iso) { return Date.parse(iso + 'T00:00:00Z'); }

const COLOR = {
  SARIMA: '#c0504d', RandomForest: '#4f81bd', XGBoost: '#6f9c3d',
  LSTM: '#8064a2', SeasonalNaive: '#9a9a9a'
};
const NOMBRE_MODELO = {
  SARIMA: 'SARIMA + Fourier', RandomForest: 'Random Forest', XGBoost: 'XGBoost',
  LSTM: 'LSTM', SeasonalNaive: 'Estacional ingenuo'
};
const RAMPA = ['#eef5f5', '#cfe4e3', '#a9d0cf', '#7fb8b7', '#4f9a99', '#2c7675'];
const ZONA_COLOR = { exito: '#e2eddc', seguridad: '#f0cf55', alarma: '#ec9444', epidemica: '#c03f2c' };
const ZONA_CLARO = { exito: '#eff5ec', seguridad: '#f7e39a', alarma: '#f6c99c', epidemica: '#eeafa3' };
const ZONA_NOMBRE = { exito: 'ÉXITO', seguridad: 'SEGURIDAD', alarma: 'ALARMA', epidemica: 'EPIDEMIA' };

/* ───────────────────────── 2. estado ───────────────────────── */

const DEPTOS = D.mapa.deptos.map(d => d.id);
const NOMBRE = {};
D.mapa.deptos.forEach(d => { NOMBRE[d.id] = d.nombre; });

/** Última semana con cobertura nacional: la siguiente suele estar incompleta. */
function ultimaSemanaUtil() {
  const s = D.serie.under5;
  for (let i = D.semanas.length - 1; i >= 0; i--) {
    let n = 0;
    for (const d of DEPTOS) if (s[d] && s[d][i] !== null) n++;
    if (n >= DEPTOS.length - 2) return i;
  }
  return D.semanas.length - 1;
}

const S = {
  dept: 'LIMA',
  cohorte: 'under5',
  wk: ultimaSemanaUtil(),
  modo: 'magnitud',
  unidad: 'recuento',
  nombres: false,
  valores: false,
  modelo: null,          // null = el que recomienda el criterio
  comparar: false,
  rango: 1,              // años visibles en el gráfico de pronóstico
  tab: 'criterio',
  zoom: 1, panx: 0, pany: 0
};

/* ───────────────────────── 3. derivaciones ───────────────────────── */

function serie(dept, coh) { return (D.serie[coh || S.cohorte] || {})[dept] || []; }

function valorSemana(dept, i) {
  const s = serie(dept);
  return (i >= 0 && i < s.length) ? s[i] : null;
}

function poblacion(dept, anio) {
  const p = D.poblacion[dept] || {};
  if (p[anio] !== undefined) return p[anio];
  const anios = Object.keys(p).map(Number).sort((a, b) => a - b);
  if (!anios.length) return null;
  return p[anio < anios[0] ? anios[0] : anios[anios.length - 1]];
}

/** Valor de la semana en la unidad elegida en el mapa. */
function valorMapa(dept, i) {
  const v = valorSemana(dept, i);
  if (v === null) return null;
  if (S.unidad === 'recuento') return v;
  const pob = poblacion(dept, D.se[i][0]);
  return pob ? v * 100000 / pob : null;
}

/** Suma de las `n` semanas que terminan en `i` (inclusive). */
function suma(dept, i, n) {
  const s = serie(dept);
  let t = 0, hay = false;
  for (let k = i - n + 1; k <= i; k++) {
    if (k >= 0 && s[k] !== null && s[k] !== undefined) { t += s[k]; hay = true; }
  }
  return hay ? t : null;
}

/** Acumulado del año epidemiológico hasta la semana `i`. */
function acumuladoAnio(dept, i) {
  const anio = D.se[i][0];
  const s = serie(dept);
  let t = 0;
  for (let k = 0; k <= i; k++) if (D.se[k][0] === anio && s[k] !== null) t += s[k];
  return t;
}

/** Corte del canal endémico de la semana `i`. */
function corte(dept, i) {
  const c = ((D.canal[S.cohorte] || {})[dept] || {})[String(D.se[i][0])];
  if (!c) return null;
  const w = D.se[i][1] - 1;
  if (c[0][w] === null || c[0][w] === undefined) return null;
  return { lo: c[0][w], mid: c[1][w], hi: c[2][w] };
}

function zonaDe(v, c) {
  if (v === null || !c) return null;
  if (v <= c.lo) return 'exito';
  if (v <= c.mid) return 'seguridad';
  if (v <= c.hi) return 'alarma';
  return 'epidemica';
}

/** Familia recomendada por el criterio para la serie seleccionada. */
function recomendado(dept) {
  const c = (D.criterio[S.cohorte] || {})[dept];
  return c ? c.modelo : null;
}
function modeloActivo() { return S.modelo || recomendado(S.dept) || 'XGBoost'; }

/** Media robusta de una ventana corta: un solo atípico no fija la tendencia. */
function nivel(vs) {
  const v = vs.filter(x => x !== null && isFinite(x));
  if (!v.length) return NaN;
  if (v.length >= 4) {
    let peor = -1, exceso = 0, tope = 0;
    for (let i = 0; i < v.length; i++) {
      const otros = v.filter((_, j) => j !== i);
      const lo = Math.min.apply(null, otros), hi = Math.max.apply(null, otros);
      const amp = hi - lo;
      const fuera = v[i] > hi ? v[i] - hi : (v[i] < lo ? lo - v[i] : 0);
      if (fuera - amp > exceso + 1e-9) { peor = i; exceso = fuera - amp; tope = v[i] > hi ? hi + amp : lo - amp; }
    }
    if (peor >= 0) v[peor] = tope;
  }
  return v.reduce((a, b) => a + b, 0) / v.length;
}

/**
 * Dirección del pronóstico: la media de las cuatro semanas pronosticadas
 * frente al nivel de las cuatro observadas. ±8 % estable, ±25 % cambio grande.
 */
function direccion(dept) {
  const s = serie(dept);
  const obs = [];
  for (let k = S.wk - 3; k <= S.wk; k++) obs.push(k >= 0 ? s[k] : null);
  const fut = ((D.futuro[S.cohorte] || {})[dept] || {})[modeloActivo()];
  if (!fut) return null;
  const a = nivel(obs), b = fut.reduce((x, y) => x + y, 0) / fut.length;
  if (!isFinite(a) || !isFinite(b)) return null;
  const cambio = a === 0 ? (b === 0 ? 0 : Infinity) : (b - a) / a;
  let clase = 'estable';
  if (cambio > 0.25) clase = 'alza fuerte';
  else if (cambio > 0.08) clase = 'al alza';
  else if (cambio < -0.25) clase = 'baja fuerte';
  else if (cambio < -0.08) clase = 'a la baja';
  return { clase: clase, pct: cambio * 100, media: b, obs: a };
}

const FLECHA = { 'estable': '→', 'al alza': '↗', 'alza fuerte': '↑', 'a la baja': '↘', 'baja fuerte': '↓' };
const ETIQUETA_DIR = {
  'estable': 'Estable', 'al alza': 'Al alza', 'alza fuerte': 'Alza fuerte',
  'a la baja': 'A la baja', 'baja fuerte': 'Baja fuerte'
};

/* ───────────────────────── 4. motor de gráficos ───────────────────────── */

/** Cortes "redondos" para un eje. */
function ticks(min, max, n) {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min];
  const bruto = (max - min) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(bruto)));
  const norm = bruto / mag;
  const paso = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / paso) * paso; v <= max + 1e-9; v += paso) out.push(+v.toFixed(10));
  return out;
}

/**
 * Gráfico de líneas y bandas sobre un eje temporal.
 *
 * cfg = {
 *   x:       [ms],                              eje común
 *   bandas:  [{lo, hi, color, nombre}],         área entre dos series
 *   zonas:   [{hasta:[], clase}],               franjas apiladas (canal endémico)
 *   lineas:  [{nombre, color, v, dash, ancho, oculta}],
 *   vline:   {x, texto},
 *   yTitulo: str,
 *   leyenda: bool
 * }
 */
function chartLineas(host, cfg) {
  host.innerHTML = '';
  const W = host.clientWidth || 600, H = host.clientHeight || 260;
  if (W < 60 || H < 60) return;
  const M = { t: cfg.leyenda === false ? 10 : 24, r: 12, b: 26, l: 46 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;

  const visibles = (cfg.lineas || []).filter(l => !l.oculta);
  let ymin = 0, ymax = 0;
  const considerar = v => { if (v !== null && v !== undefined && isFinite(v)) { if (v > ymax) ymax = v; if (v < ymin) ymin = v; } };
  visibles.forEach(l => l.v.forEach(considerar));
  (cfg.bandas || []).forEach(b => { b.lo.forEach(considerar); b.hi.forEach(considerar); });
  (cfg.zonas || []).forEach(z => z.hasta.forEach(considerar));
  if (ymax === ymin) ymax = ymin + 1;
  ymax *= 1.08;

  const x0 = cfg.x[0], x1 = cfg.x[cfg.x.length - 1];
  const X = t => M.l + (x1 === x0 ? 0 : (t - x0) / (x1 - x0) * iw);
  const Y = v => M.t + ih - (v - ymin) / (ymax - ymin) * ih;

  const s = svg('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H });

  // rejilla y eje y
  const ty = ticks(ymin, ymax, 4);
  const gy = svg('g', { class: 'ax' });
  ty.forEach(v => {
    gy.appendChild(svg('line', { class: 'gridline', x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v) }));
    const t = svg('text', { x: M.l - 7, y: Y(v) + 3.5, 'text-anchor': 'end' });
    t.textContent = v >= 1000 ? fmt(v / 1000, v % 1000 ? 1 : 0) + ' mil' : fmt(v, v < 10 && v % 1 ? 1 : 0);
    gy.appendChild(t);
  });
  s.appendChild(gy);

  // eje x: un corte por año si el tramo es largo, por trimestre si es corto
  const gx = svg('g', { class: 'ax' });
  const anios = (x1 - x0) / (365.25 * 864e5);
  const paso = anios > 8 ? 24 : anios > 3 ? 12 : anios > 1.2 ? 6 : 3;
  const d0 = new Date(x0);
  let cur = Date.UTC(d0.getUTCFullYear(), Math.floor(d0.getUTCMonth() / paso) * paso, 1);
  while (cur <= x1) {
    if (cur >= x0) {
      const dd = new Date(cur);
      const t = svg('text', { x: X(cur), y: H - 8, 'text-anchor': 'middle' });
      t.textContent = paso >= 12 ? String(dd.getUTCFullYear())
        : dd.getUTCFullYear() + '-' + String(dd.getUTCMonth() + 1).padStart(2, '0');
      gx.appendChild(t);
      gx.appendChild(svg('line', { class: 'gridline', x1: X(cur), x2: X(cur), y1: M.t, y2: M.t + ih }));
    }
    const dd = new Date(cur);
    cur = Date.UTC(dd.getUTCFullYear(), dd.getUTCMonth() + paso, 1);
  }
  s.appendChild(gx);

  // zona sombreada: el tramo pronosticado, para que se distinga del observado
  if (cfg.sombra) {
    const xa = X(Math.max(cfg.sombra.desde, x0)), xb = X(Math.min(cfg.sombra.hasta, x1));
    if (xb > xa) {
      s.appendChild(svg('rect', { x: xa, y: M.t, width: xb - xa, height: ih, fill: '#12376b', opacity: .035 }));
    }
  }

  // franjas del canal endémico, de abajo hacia arriba
  let base = null;
  (cfg.zonas || []).forEach(z => {
    const arriba = [], abajo = [];
    for (let i = 0; i < cfg.x.length; i++) {
      if (z.hasta[i] === null || z.hasta[i] === undefined) continue;
      arriba.push(X(cfg.x[i]) + ',' + Y(z.hasta[i]));
      abajo.unshift(X(cfg.x[i]) + ',' + Y(base ? (base[i] || 0) : ymin));
    }
    if (arriba.length > 1) {
      s.appendChild(svg('polygon', {
        points: arriba.concat(abajo).join(' '), fill: ZONA_CLARO[z.clase],
        opacity: z.clase === 'epidemica' ? .55 : .85, stroke: 'none'
      }));
    }
    base = z.hasta;
  });

  // bandas de incertidumbre
  (cfg.bandas || []).forEach(b => {
    const arriba = [], abajo = [];
    for (let i = 0; i < cfg.x.length; i++) {
      if (b.hi[i] === null || b.hi[i] === undefined) continue;
      arriba.push(X(cfg.x[i]) + ',' + Y(b.hi[i]));
      abajo.unshift(X(cfg.x[i]) + ',' + Y(b.lo[i]));
    }
    if (arriba.length > 1) {
      s.appendChild(svg('polygon', { points: arriba.concat(abajo).join(' '), fill: b.color, stroke: 'none' }));
    }
  });

  // líneas
  visibles.forEach(l => {
    let d = '', abierto = false;
    for (let i = 0; i < cfg.x.length; i++) {
      const v = l.v[i];
      if (v === null || v === undefined || !isFinite(v)) { abierto = false; continue; }
      d += (abierto ? 'L' : 'M') + X(cfg.x[i]).toFixed(1) + ',' + Y(v).toFixed(1);
      abierto = true;
    }
    if (d) {
      s.appendChild(svg('path', {
        d: d, fill: 'none', stroke: l.color, 'stroke-width': l.ancho || 1.6,
        'stroke-dasharray': l.dash || null, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'
      }));
    }
    if (l.puntos) {
      for (let i = 0; i < cfg.x.length; i++) {
        const v = l.v[i];
        if (v === null || v === undefined || !isFinite(v)) continue;
        s.appendChild(svg('circle', {
          cx: X(cfg.x[i]), cy: Y(v), r: 2.8, fill: '#fff', stroke: l.color, 'stroke-width': 1.6
        }));
      }
    }
  });

  // regla vertical del origen del pronóstico
  if (cfg.vline && cfg.vline.x >= x0 && cfg.vline.x <= x1) {
    s.appendChild(svg('line', {
      x1: X(cfg.vline.x), x2: X(cfg.vline.x), y1: M.t, y2: M.t + ih,
      stroke: '#9aa1a9', 'stroke-width': 1, 'stroke-dasharray': '3 3'
    }));
    if (cfg.vline.texto) {
      const t = svg('text', { x: X(cfg.vline.x) + 4, y: M.t + 9, 'font-size': 9, fill: '#9aa1a9' });
      t.textContent = cfg.vline.texto;
      s.appendChild(t);
    }
  }

  // lectura al pasar el ratón
  const cursor = svg('line', { y1: M.t, y2: M.t + ih, stroke: '#12376b', 'stroke-width': 1, opacity: 0 });
  s.appendChild(cursor);
  const capa = svg('rect', { x: M.l, y: M.t, width: iw, height: ih, fill: 'transparent' });
  s.appendChild(capa);
  host.appendChild(s);

  const tip = el('div', 'crosstip');
  tip.style.display = 'none';
  host.appendChild(tip);

  capa.addEventListener('mousemove', ev => {
    const caja = s.getBoundingClientRect();
    const px = ev.clientX - caja.left;
    let mejor = 0, dmin = Infinity;
    for (let i = 0; i < cfg.x.length; i++) {
      const d = Math.abs(X(cfg.x[i]) - px);
      if (d < dmin) { dmin = d; mejor = i; }
    }
    cursor.setAttribute('x1', X(cfg.x[mejor]));
    cursor.setAttribute('x2', X(cfg.x[mejor]));
    cursor.setAttribute('opacity', 0.35);
    let html = '<b>' + (cfg.etiquetaX ? cfg.etiquetaX(mejor) : fechaCorta(new Date(cfg.x[mejor]).toISOString().slice(0, 10))) + '</b>';
    visibles.forEach(l => {
      const v = l.v[mejor];
      if (v === null || v === undefined || !isFinite(v)) return;
      html += '<div class="row"><i style="background:' + l.color + '"></i><span>' + esc(l.nombre) +
        '</span><b>' + fmt(v, cfg.dec === undefined ? 0 : cfg.dec) + '</b></div>';
    });
    (cfg.bandas || []).forEach(b => {
      if (b.lo[mejor] === null || b.lo[mejor] === undefined) return;
      html += '<div class="row"><i style="background:' + b.color + '"></i><span>' + esc(b.nombre) +
        '</span><b>' + fmt(b.lo[mejor], 0) + '–' + fmt(b.hi[mejor], 0) + '</b></div>';
    });
    tip.innerHTML = html;
    tip.style.display = 'block';
    const ancho = tip.offsetWidth;
    tip.style.left = Math.max(2, Math.min(W - ancho - 2, X(cfg.x[mejor]) - ancho / 2)) + 'px';
  });
  capa.addEventListener('mouseleave', () => { tip.style.display = 'none'; cursor.setAttribute('opacity', 0); });

  // leyenda, en la banda muerta de arriba
  if (cfg.leyenda !== false) {
    const lg = el('div', 'plot-legend');
    (cfg.bandas || []).forEach(b => {
      lg.appendChild(el('div', 'lgd band', '<i style="background:' + b.color + '"></i><span>' + esc(b.nombre) + '</span>'));
    });
    (cfg.lineas || []).forEach(l => {
      if (l.sinLeyenda) return;
      const chip = el('div', 'lgd' + (l.dash ? ' dash' : '') + (l.oculta ? ' off' : ''),
        '<i style="' + (l.dash ? 'border-color:' : 'background:') + l.color + '"></i><span>' + esc(l.nombre) + '</span>');
      chip.onclick = () => {
        const nuevo = !l.oculta;
        // el pronóstico de una familia se enciende y se apaga con su retrospectiva
        (cfg.lineas || []).forEach(o => { if (o.grupo && o.grupo === l.grupo) o.oculta = nuevo; });
        l.oculta = nuevo;
        chartLineas(host, cfg);
      };
      lg.appendChild(chip);
    });
    host.appendChild(lg);
  }
}

/** Barras horizontales (atribución SHAP, comparaciones cortas). */
function barras(host, filas, opciones) {
  const o = opciones || {};
  host.innerHTML = '';
  const max = Math.max.apply(null, filas.map(f => Math.abs(f.valor)).concat([1e-9]));
  filas.forEach(f => {
    const fila = el('div', 'barrow');
    fila.appendChild(el('span', null, esc(f.nombre)));
    const t = el('div', 'track');
    const fill = el('div', 'fill');
    fill.style.width = (100 * Math.abs(f.valor) / max) + '%';
    fill.style.background = f.color || '#4f81bd';
    t.appendChild(fill);
    fila.appendChild(t);
    fila.appendChild(el('b', null, o.formato ? o.formato(f.valor) : fmt(f.valor, 3)));
    host.appendChild(fila);
  });
}

/* ───────────────────────── 5. mapa ───────────────────────── */

let mapaSvg = null, mapaG = null;

function construirMapa() {
  const host = $('#mapa');
  host.innerHTML = '';
  const s = svg('svg', {
    viewBox: '0 0 ' + D.mapa.ancho + ' ' + D.mapa.alto,
    preserveAspectRatio: 'xMidYMid meet'
  });
  const defs = svg('defs');
  const pat = svg('pattern', { id: 'rayado', width: 6, height: 6, patternTransform: 'rotate(45)', patternUnits: 'userSpaceOnUse' });
  pat.appendChild(svg('rect', { width: 6, height: 6, fill: '#f1f3f5' }));
  pat.appendChild(svg('rect', { width: 2.5, height: 6, fill: '#b3bac2' }));
  defs.appendChild(pat);
  s.appendChild(defs);

  const g = svg('g', { id: 'capa-mapa' });
  D.mapa.deptos.forEach(d => {
    const p = svg('path', { class: 'dep', d: d.d, 'data-id': d.id });
    p.addEventListener('click', () => { S.dept = d.id; pintar(); });
    p.addEventListener('mouseenter', () => resaltar(d.id));
    p.addEventListener('mouseleave', () => resaltar(null));
    g.appendChild(p);
  });
  const etiquetas = svg('g', { id: 'capa-etiquetas' });
  g.appendChild(etiquetas);
  s.appendChild(g);
  host.appendChild(s);
  mapaSvg = s; mapaG = g;

  // desplazamiento con el ratón
  let arrastre = null;
  s.addEventListener('mousedown', ev => { arrastre = { x: ev.clientX, y: ev.clientY, px: S.panx, py: S.pany }; });
  window.addEventListener('mouseup', () => { arrastre = null; });
  window.addEventListener('mousemove', ev => {
    if (!arrastre) return;
    const k = D.mapa.ancho / s.getBoundingClientRect().width;
    S.panx = arrastre.px + (ev.clientX - arrastre.x) * k;
    S.pany = arrastre.py + (ev.clientY - arrastre.y) * k;
    aplicarZoom();
  });
}

/**
 * Cuántas unidades del lienzo entran en un píxel de pantalla.
 *
 * El SVG se ajusta con preserveAspectRatio, así que el texto dibujado dentro
 * encoge con el mapa. Las etiquetas tienen que medirse en píxeles de pantalla
 * —si no, en un panel estrecho quedan ilegibles—, de ahí esta conversión.
 */
function unidadesPorPixel() {
  if (!mapaSvg) return 1;
  const r = mapaSvg.getBoundingClientRect();
  if (!r.width || !r.height) return 1;
  const k = Math.min(r.width / D.mapa.ancho, r.height / D.mapa.alto);
  return k > 0 ? 1 / (k * S.zoom) : 1;
}

function aplicarZoom() {
  if (!mapaG) return;
  const cx = D.mapa.ancho / 2, cy = D.mapa.alto / 2;
  mapaG.setAttribute('transform',
    'translate(' + (cx + S.panx) + ',' + (cy + S.pany) + ') scale(' + S.zoom + ') translate(' + (-cx) + ',' + (-cy) + ')');
  const u = unidadesPorPixel();
  mapaG.querySelectorAll('path.dep').forEach(p => {
    p.style.strokeWidth = (p.classList.contains('sel') ? 2.4 : 0.9) * u;
  });
  const et = document.getElementById('capa-etiquetas');
  if (et) et.querySelectorAll('text').forEach(t => {
    t.setAttribute('font-size', (+t.dataset.base || 10) * u);
    t.setAttribute('stroke-width', 2.6 * u);
  });
}

function resaltar(id) {
  if (!mapaG) return;
  const base = 0.9 * unidadesPorPixel();
  mapaG.querySelectorAll('path.dep').forEach(p => {
    p.style.strokeWidth = (id && p.dataset.id === id) ? base * 2.4 : base;
  });
}

/** Cortes por cuantiles: seis clases sobre los valores de la semana. */
function cortesCuantiles(valores) {
  const v = valores.filter(x => x !== null && isFinite(x) && x > 0).sort((a, b) => a - b);
  if (!v.length) return null;
  const q = p => v[Math.min(v.length - 1, Math.max(0, Math.round(p * (v.length - 1))))];
  const c = [q(1 / 6), q(2 / 6), q(3 / 6), q(4 / 6), q(5 / 6)];
  return { cortes: c, min: v[0], max: v[v.length - 1] };
}

function pintarMapa() {
  const vals = {}, lista = [];
  DEPTOS.forEach(d => { const v = valorMapa(d, S.wk); vals[d] = v; if (v !== null) lista.push(v); });
  const esc6 = cortesCuantiles(lista);

  const color = d => {
    if (S.modo === 'zona') {
      const z = zonaDe(valorSemana(d, S.wk), corte(d, S.wk));
      return z ? ZONA_COLOR[z] : null;
    }
    const v = vals[d];
    if (v === null || !esc6) return null;
    let k = 0;
    while (k < 5 && v > esc6.cortes[k]) k++;
    return RAMPA[k];
  };

  mapaG.querySelectorAll('path.dep').forEach(p => {
    const c = color(p.dataset.id);
    p.setAttribute('fill', c || 'url(#rayado)');
    p.classList.toggle('sel', p.dataset.id === S.dept);
  });

  // etiquetas
  const capa = document.getElementById('capa-etiquetas');
  capa.innerHTML = '';
  if (S.nombres || S.valores) {
    const u = unidadesPorPixel();
    D.mapa.deptos.forEach(d => {
      const oscuro = S.modo === 'magnitud' && color(d.id) === RAMPA[5];
      const n = (S.nombres ? 1 : 0) + (S.valores ? 1 : 0);
      let dy = -((n - 1) * 6) * u;
      const poner = (txt, tam, peso) => {
        const t = svg('text', {
          x: d.ex, y: d.ey + dy, 'text-anchor': 'middle', 'font-size': tam * u,
          'font-weight': peso, fill: oscuro ? '#fff' : '#1b2a33',
          stroke: oscuro ? 'rgba(12,22,30,.65)' : 'rgba(255,255,255,.92)',
          'stroke-width': 2.6 * u, 'paint-order': 'stroke'
        });
        t.dataset.base = tam;
        t.textContent = txt;
        capa.appendChild(t);
        dy += (tam + 2.5) * u;
      };
      if (S.nombres) poner(d.nombre.toUpperCase(), 9, 600);
      if (S.valores) {
        const v = vals[d.id];
        poner(v === null ? '—' : fmt(v, S.unidad === 'tasa' ? 1 : 0), 10, 600);
      }
    });
  }

  pintarLeyenda(esc6);
  pintarFicha();
  aplicarZoom();
}

function pintarLeyenda(esc6) {
  const host = $('#leyenda');
  if (S.modo === 'zona') {
    let h = '<div class="lbl">Zona endémica · semana observada</div><div class="zl-bar">';
    ['exito', 'seguridad', 'alarma', 'epidemica'].forEach(z => {
      h += '<div class="zl-cell ' + z + '"><i style="background:' + ZONA_CLARO[z] + '"></i><span>' +
        ZONA_NOMBRE[z].charAt(0) + ZONA_NOMBRE[z].slice(1).toLowerCase() + '</span></div>';
    });
    h += '</div><div class="zl-ref">Cuartiles de las 5 temporadas previas, sin 2020–2021.</div>';
    h += '<div class="nodata-nota"><i></i> sin dato esa semana</div>';
    host.innerHTML = h;
    return;
  }
  let h = '<div class="seg"><button data-u="recuento" class="' + (S.unidad === 'recuento' ? 'on' : '') +
    '">Recuento</button><button data-u="tasa" class="' + (S.unidad === 'tasa' ? 'on' : '') + '">Tasa / 100k</button></div>';
  h += '<div class="lbl">' + (S.unidad === 'recuento' ? 'Casos' : 'Casos por 100 000') + ' · departamentos · 6 cuantiles</div>';
  h += '<div class="ramp">' + RAMPA.map(c => '<i style="background:' + c + '"></i>').join('') + '</div>';
  h += '<div class="ramp-scale"><span>mín ' + (esc6 ? fmt(esc6.min, S.unidad === 'tasa' ? 1 : 0) : '—') +
    '</span><span>máx ' + (esc6 ? fmt(esc6.max, S.unidad === 'tasa' ? 1 : 0) : '—') + '</span></div>';
  h += '<div class="nodata-nota"><i></i> sin dato esa semana</div>';
  host.innerHTML = h;
  host.querySelectorAll('[data-u]').forEach(b => {
    b.onclick = () => { S.unidad = b.dataset.u; pintar(); };
  });
}

function pintarFicha() {
  const d = S.dept, i = S.wk;
  const v = valorSemana(d, i);
  const c = corte(d, i);
  const z = zonaDe(v, c);
  const pob = poblacion(d, D.se[i][0]);
  const u4 = suma(d, i, 4), p4 = suma(d, i - 4, 4);
  const dir = direccion(d);
  const ev = D.evaluacion[d] || {};
  let h = '<div class="sc-n"><span>' + esc(NOMBRE[d]) + '</span>' +
    (z ? '<span class="zone-badge ' + z + '">' + ZONA_NOMBRE[z] + '</span>' : '') + '</div>';
  const fila = (k, val) => '<div class="sc-r"><span>' + k + '</span><b>' + val + '</b></div>';
  h += fila('Población', pob ? fmt(pob) : '—');
  h += fila('Casos · SE ' + D.se[i][1], v === null ? '—' : fmt(v));
  h += fila('Tasa / 100k', pob && v !== null ? fmt(v * 100000 / pob, 2) : '—');
  h += fila('Tendencia 4 sem', (u4 !== null && p4) ? pct(100 * (u4 - p4) / p4) : '—');
  h += fila('Habitual · umbral', c ? fmt(c.mid) + ' · ' + fmt(c.hi) : '—');
  h += fila('Pronóstico 4 sem', dir ? fmt(dir.media * 4) : '—');
  h += fila('Modelo ★', recomendado(d) ? NOMBRE_MODELO[recomendado(d)] : '—');
  if (ev.empate) h += fila('Empatados', ev.empate.length + ' de 4');
  $('#ficha').innerHTML = h;
}

/* ───────────────────────── 6. cinta de titulares ───────────────────────── */

function pintarTitulares() {
  const d = S.dept, i = S.wk;
  const v = valorSemana(d, i), prev = valorSemana(d, i - 1);
  const u4 = suma(d, i, 4), p4 = suma(d, i - 4, 4);
  const c = corte(d, i), z = zonaDe(v, c);
  const dir = direccion(d);
  const cohorte = D.meta.cohortes.filter(x => x.id === S.cohorte)[0].nombre;

  const cambio = (a, b) => (a !== null && b) ? pct(100 * (a - b) / b) : '';
  const clase = (a, b) => (a !== null && b) ? (a > b ? 'up' : a < b ? 'down' : '') : '';

  let h = '';
  h += '<div class="hl-scope"><div class="hl-what hl-lbl">Casos · ' + esc(cohorte.toLowerCase()) + '</div>' +
    '<div class="hl-area">' + esc(NOMBRE[d]) + '</div>' +
    '<div class="hl-step"><button id="wk-prev" ' + (i <= 0 ? 'disabled' : '') + '>‹</button>' +
    '<span class="hl-weeklabel">SE ' + D.se[i][1] + ' · ' + D.se[i][0] + '</span>' +
    '<button id="wk-next" ' + (i >= D.semanas.length - 1 ? 'disabled' : '') + '>›</button></div></div>';

  h += '<div class="hl-cell"><div class="hl-lbl">Esta semana</div><div class="hl-val">' +
    '<span class="hl-main">' + fmt(v) + '</span>' +
    '<span class="hl-pct ' + clase(v, prev) + '">' + cambio(v, prev) + '</span></div></div>';

  h += '<div class="hl-cell"><div class="hl-lbl">Últimas 4 semanas</div><div class="hl-val">' +
    '<span class="hl-main">' + fmt(u4) + '</span>' +
    '<span class="hl-pct ' + clase(u4, p4) + '">' + cambio(u4, p4) + '</span></div></div>';

  const exceso = (c && c.mid) ? 100 * (v - c.mid) / c.mid : null;
  h += '<div class="hl-cell"><div class="hl-lbl">Zona endémica</div><div class="hl-val">' +
    (z ? '<span class="hl-main badge zone-badge ' + z + '">' + ZONA_NOMBRE[z] + '</span>' : '<span class="hl-main">—</span>') +
    '<span class="hl-sub">' + (exceso === null ? '' : pct(exceso) + ' vs habitual') + '</span></div></div>';

  const fut = ((D.futuro[S.cohorte] || {})[d] || {})[modeloActivo()];
  const b = banda(d, modeloActivo());
  let rango = '—';
  if (fut && b && b[3]) rango = fmt(Math.max(0, fut[3] + b[3][0])) + '–' + fmt(fut[3] + b[3][1]);
  else if (fut) rango = fmt(fut[3]) + ' (puntual)';
  h += '<div class="hl-cell"><div class="hl-lbl">Próximas 4 semanas</div><div class="hl-val stack">' +
    '<span class="hl-main dir">' + (dir ? FLECHA[dir.clase] + ' ' + ETIQUETA_DIR[dir.clase] : '—') + '</span>' +
    '<span class="hl-sub">' + rango + ' en la SE ' + semanaFutura(3) + '</span></div></div>';

  h += '<div class="hl-cell"><div class="hl-lbl">' + D.se[i][0] + ' acumulado</div><div class="hl-val">' +
    '<span class="hl-main">' + fmt(acumuladoAnio(d, i)) + '</span></div></div>';

  $('#titulares').innerHTML = h;
  $('#wk-prev').onclick = () => { if (S.wk > 0) { S.wk--; pintar(); } };
  $('#wk-next').onclick = () => { if (S.wk < D.semanas.length - 1) { S.wk++; pintar(); } };
}

function semanaFutura(k) {
  if (!D.futuro_fechas.length) return '—';
  const f = new Date(Date.parse(D.futuro_fechas[k] + 'T00:00:00Z'));
  const ene4 = Date.UTC(f.getUTCFullYear(), 0, 4);
  const sem = Math.floor((f - ene4) / (7 * 864e5)) + 1;
  return sem + ' de ' + f.getUTCFullYear();
}

/**
 * Banda empírica del modelo, o null si no la hay para esta serie.
 *
 * Solo existe donde corrió el walk-forward, es decir en menores de 5 años.
 * Aplicar al grupo de 60 y más una banda calibrada con los errores de otra
 * población sería inventar una incertidumbre que nadie midió.
 */
function banda(dept, modelo) {
  if (S.cohorte !== 'under5') return null;
  return (D.banda[dept] || {})[modelo] || null;
}

/* ───────────────────────── 7. controles ───────────────────────── */

function desplegable(host, etiqueta, actual, opciones, alElegir, alinearDerecha) {
  const cell = el('div', 'cell');
  if (etiqueta) cell.appendChild(el('div', 'lbl', etiqueta));
  const btn = el('div', 'sel', '<span>' + esc(actual) + '</span><i class="caret">▼</i>');
  cell.appendChild(btn);
  btn.onclick = ev => {
    ev.stopPropagation();
    cerrarMenus();
    const m = el('div', 'menu' + (alinearDerecha ? ' right' : ''));
    opciones.forEach(o => {
      if (o.separador) { m.appendChild(el('div', 'msep')); return; }
      if (o.titulo) { m.appendChild(el('div', 'mhd', esc(o.titulo))); return; }
      const it = el('div', 'mi' + (o.valor === o.actual || o.sel ? ' on' : '') + (o.dis ? ' dis' : ''),
        (o.color ? '<i class="swatch" style="background:' + o.color + '"></i>' : '') +
        '<span>' + esc(o.nombre) + '</span>' + (o.extra ? '<span class="fam">' + esc(o.extra) + '</span>' : ''));
      if (!o.dis) it.onclick = () => { cerrarMenus(); alElegir(o.valor); };
      m.appendChild(it);
    });
    cell.appendChild(m);
    btn.classList.add('act');
  };
  host.appendChild(cell);
  return cell;
}

function cerrarMenus() {
  document.querySelectorAll('.menu').forEach(m => m.remove());
  document.querySelectorAll('.sel.act').forEach(s => s.classList.remove('act'));
}
document.addEventListener('click', cerrarMenus);

function pintarControles() {
  const host = $('#controles');
  host.innerHTML = '';
  const fila = el('div', 'ctl-row');

  const cd = el('div', 'c-dept');
  desplegable(cd, null, NOMBRE[S.dept],
    DEPTOS.map(d => ({ nombre: NOMBRE[d], valor: d, sel: d === S.dept })),
    v => { S.dept = v; pintar(); });
  fila.appendChild(cd);

  const cm = el('div', 'c-medida');
  desplegable(cm, null, 'Casos', [
    { nombre: 'Casos', valor: 'casos', sel: true },
    { nombre: 'Hospitalizaciones', valor: 'h', dis: true, extra: 'solo anual' },
    { nombre: 'Defunciones', valor: 'd', dis: true, extra: 'solo anual' }
  ], () => { });
  fila.appendChild(cm);

  const cc = el('div', 'c-cohorte');
  desplegable(cc, null, D.meta.cohortes.filter(c => c.id === S.cohorte)[0].nombre,
    D.meta.cohortes.map(c => ({
      nombre: c.nombre, valor: c.id, sel: c.id === S.cohorte,
      extra: c.id === 'under5' ? 'modelado' : 'solo vigilancia'
    })),
    v => { S.cohorte = v; pintar(); });
  fila.appendChild(cc);

  const ct = el('div', 'c-rango');
  const rangos = [{ n: '26 semanas', v: 0.5 }, { n: '1 año', v: 1 }, { n: '3 años', v: 3 },
  { n: '5 años', v: 5 }, { n: 'Todo', v: 99 }];
  desplegable(ct, null, rangos.filter(r => r.v === S.rango)[0].n,
    rangos.map(r => ({ nombre: r.n, valor: r.v, sel: r.v === S.rango })),
    v => { S.rango = v; pintar(); }, true);
  fila.appendChild(ct);

  const reset = el('div', 'c-reset');
  const b = el('button', 'iconbtn', '↺');
  b.title = 'Restablecer';
  b.onclick = () => {
    S.dept = 'LIMA'; S.cohorte = 'under5'; S.wk = ultimaSemanaUtil(); S.modelo = null;
    S.comparar = false; S.rango = 1; S.modo = 'magnitud'; S.unidad = 'recuento';
    S.zoom = 1; S.panx = 0; S.pany = 0; pintar();
  };
  reset.appendChild(b);
  fila.appendChild(reset);
  host.appendChild(fila);
}

/* ───────────────────────── 8. panel de pronóstico ───────────────────────── */

/** Índices del tramo visible del eje, según el rango elegido. */
function tramo() {
  const n = D.semanas.length;
  const desde = S.rango >= 99 ? 0 : Math.max(0, S.wk - Math.round(S.rango * 52.18));
  return { desde: desde, hasta: Math.min(n - 1, S.wk) };
}

function pintarPronostico() {
  const host = $('#panel-pronostico');
  const d = S.dept;
  const modelo = modeloActivo();
  const rec = recomendado(d);
  const ev = D.evaluacion[d] || {};
  const met = (D.metricas[d] || {})[modelo];
  const dir = direccion(d);
  const fut = ((D.futuro[S.cohorte] || {})[d] || {})[modelo];
  const bnd = banda(d, modelo);
  const cob = (D.cobertura[d] || {})[modelo];

  host.innerHTML = '';
  const head = el('div', 'phead');
  head.appendChild(el('div', 'ptitle', 'Pronóstico principal: <em>' + esc(NOMBRE[d]) + '</em>'));
  const hc = el('div', 'head-controls');
  const sel = el('div', 'cell');
  const opciones = [{ titulo: 'Recomendado por el criterio' },
  { nombre: NOMBRE_MODELO[rec] + ' ★', valor: null, sel: S.modelo === null, color: COLOR[rec] },
  { titulo: 'Otras familias' }];
  D.meta.modelos.forEach(m => {
    if (m === rec) return;
    const sinFuturo = D.meta.con_futuro.indexOf(m) < 0;
    opciones.push({
      nombre: NOMBRE_MODELO[m], valor: m, sel: S.modelo === m, color: COLOR[m],
      extra: sinFuturo ? 'descartado R0' : null, dis: sinFuturo
    });
  });
  desplegable(sel, null, NOMBRE_MODELO[modelo] + (modelo === rec ? ' ★' : ''), opciones,
    v => { S.modelo = v; pintar(); }, true);
  hc.appendChild(sel);
  const cmp = el('button', 'cmp-all' + (S.comparar ? ' on' : ''), 'Comparar las 5');
  cmp.onclick = () => { S.comparar = !S.comparar; pintar(); };
  hc.appendChild(cmp);
  head.appendChild(hc);
  host.appendChild(head);

  if (S.cohorte !== 'under5') {
    host.appendChild(el('div', 'aviso',
      'La validación retrospectiva y la explicabilidad de esta tesina se ejecutaron sobre ' +
      '<b>menores de 5 años</b>. Para 60 y más se muestra el pronóstico del modelo que el criterio ' +
      'deriva del perfil de esa serie, todavía sin contraste de error.'));
  }

  // resumen
  const resumen = el('div', 'fc-summary');
  const badge = el('div', 'dir-badge',
    '<span class="arrow">' + (dir ? FLECHA[dir.clase] : '·') + '</span>' +
    '<span class="dlabel">' + (dir ? ETIQUETA_DIR[dir.clase] : '—') + '</span>');
  resumen.appendChild(badge);
  const txt = el('div', 'fc-text');
  const conBanda = !!(fut && bnd && bnd[3]);
  let rango = '—';
  if (conBanda) rango = fmt(Math.max(0, fut[3] + bnd[3][0])) + '–' + fmt(fut[3] + bnd[3][1]);
  else if (fut) rango = fmt(fut[3]);
  txt.appendChild(el('div', 'fc-lead',
    (dir ? 'Se espera que ' + ({
      'estable': 'se mantenga', 'al alza': 'suba', 'alza fuerte': 'suba con fuerza',
      'a la baja': 'baje', 'baja fuerte': 'baje con fuerza'
    })[dir.clase] + '. ' : '') +
    '<b>' + rango + '</b> casos esperados en la SE ' + semanaFutura(3) +
    (conBanda ? ' (banda empírica del 80 %).' : ' (pronóstico puntual: esta serie no tiene banda validada).')));
  const dentro = ev.empate ? ev.empate.indexOf(modelo) >= 0 : null;
  txt.appendChild(el('div', 'fc-sub',
    (modelo === rec ? '<i class="star">★</i> ' : '') + esc(NOMBRE_MODELO[modelo]) +
    (ev.regla ? ' · regla ' + ev.regla : '') +
    (met ? ' · MAE ' + fmt(met.mae, 2) + ' casos en ' + fmt(met.n) + ' pasos retrospectivos' : '') +
    (dentro === null ? '' : dentro ? ' · dentro del conjunto empatado' : ' · fuera del conjunto empatado')));
  resumen.appendChild(txt);
  host.appendChild(resumen);

  const wrap = el('div', 'chart-wrap');
  const plot = el('div', 'plot');
  plot.id = 'plot-pronostico';
  wrap.appendChild(plot);
  host.appendChild(wrap);

  const foot = el('div', 'chart-foot');
  foot.appendChild(el('span', 'mono',
    'HORIZONTE ' + D.meta.horizonte + ' SEM · EMITIDO ' + fechaCorta(D.semanas[S.wk])));
  foot.appendChild(el('span', null, 'arrastre el mapa · clic en la leyenda para ocultar una serie'));
  host.appendChild(foot);

  const naive = (D.metricas[d] || {}).SeasonalNaive;
  let conf = 'Confiabilidad (' + (ev.n_origenes || '—') + ' orígenes de validación): ';
  conf += cob !== null && cob !== undefined
    ? 'la banda del 80 % contuvo el <b>' + fmt(cob, 1) + ' %</b> de las semanas del 30 % final, ' +
      'que no intervino en calcularla'
    : 'banda no disponible';
  if (met && naive && naive.mae) {
    conf += ' · MAE <b>' + fmt(100 * (naive.mae - met.mae) / naive.mae, 1) +
      ' %</b> mejor que repetir la misma semana del año anterior';
  }
  host.appendChild(el('div', 'reliability', conf));

  // Diferido: el gráfico se mide contra su contenedor, y el contenedor solo
  // tiene altura definitiva cuando el navegador ha resuelto el flex del panel.
  luego(dibujarPronostico);
}

function dibujarPronostico() {
  const plot = document.getElementById('plot-pronostico');
  if (!plot) return;
  const d = S.dept, t = tramo();
  const x = [], obs = [];
  for (let i = t.desde; i <= t.hasta; i++) { x.push(ms(D.semanas[i])); obs.push(valorSemana(d, i)); }
  const nObs = x.length;
  D.futuro_fechas.forEach(f => x.push(ms(f)));
  const rellenar = () => new Array(x.length).fill(null);

  const obsCompleto = obs.slice();
  while (obsCompleto.length < x.length) obsCompleto.push(null);
  const lineas = [{ nombre: 'Observado', color: '#1a1d21', ancho: 2, v: obsCompleto }];
  const bandas = [];
  const modelos = S.comparar ? D.meta.modelos.slice() : [modeloActivo()];
  const activo = modeloActivo();

  // Dos trazos por familia: la validación retrospectiva (punteada, es lo ya
  // ocurrido y contrastado) y el pronóstico (continua y con marcas, es lo que
  // todavía no ha pasado). Distinguirlos evita leer como pronóstico lo que en
  // realidad es un ajuste sobre datos conocidos.
  modelos.forEach(m => {
    const bt = (D.backtest[d] || {})[m];
    const v = rellenar();
    if (bt) {
      for (let k = 0; k < bt.i.length; k++) {
        const i = bt.i[k];
        if (i >= t.desde && i <= t.hasta) v[i - t.desde] = bt.v[k];
      }
    }
    lineas.push({
      nombre: NOMBRE_MODELO[m] + ' · retrospectivo', color: COLOR[m], grupo: m,
      ancho: m === activo ? 1.7 : 1.1, dash: '4 3', v: v
    });
    const fut = ((D.futuro[S.cohorte] || {})[d] || {})[m];
    if (fut) {
      const f = rellenar();
      f[nObs - 1] = obs[nObs - 1];
      for (let k = 0; k < fut.length; k++) f[nObs + k] = fut[k];
      lineas.push({
        nombre: NOMBRE_MODELO[m] + ' · pronóstico', color: COLOR[m], grupo: m,
        ancho: m === activo ? 2.4 : 1.5, puntos: m === activo, sinLeyenda: true, v: f
      });
    }
  });

  // banda empírica, solo sobre el tramo futuro del modelo activo
  const bnd = banda(d, modeloActivo());
  const fut = ((D.futuro[S.cohorte] || {})[d] || {})[modeloActivo()];
  if (bnd && fut && !S.comparar) {
    const lo = rellenar(), hi = rellenar();
    lo[nObs - 1] = hi[nObs - 1] = obs[nObs - 1];
    for (let k = 0; k < fut.length; k++) {
      const b = bnd[k];
      if (!b) continue;
      lo[nObs + k] = Math.max(0, fut[k] + b[0]);
      hi[nObs + k] = fut[k] + b[1];
    }
    bandas.push({ nombre: 'Banda 80 %', color: 'rgba(79,129,189,.20)', lo: lo, hi: hi });
  }

  // referencias del canal endémico
  const habitual = rellenar(), umbral = rellenar();
  for (let i = t.desde; i <= t.hasta; i++) {
    const c = corte(d, i);
    if (c) { habitual[i - t.desde] = c.mid; umbral[i - t.desde] = c.hi; }
  }
  lineas.push({ nombre: 'Habitual', color: '#b9a64a', dash: '5 4', ancho: 1.3, v: habitual });
  lineas.push({ nombre: 'Umbral epidémico', color: '#ec9444', dash: '5 4', ancho: 1.3, v: umbral });

  chartLineas(plot, {
    x: x, lineas: lineas, bandas: bandas,
    sombra: { desde: ms(D.semanas[t.hasta]), hasta: x[x.length - 1] },
    vline: { x: ms(D.semanas[t.hasta]), texto: 'pronóstico →' }
  });
}

/* ───────────────────────── 9. panel de contexto ───────────────────────── */

const PESTANAS = [
  { id: 'serie', nombre: 'Serie de tiempo' },
  { id: 'endemico', nombre: 'Canal endémico' },
  { id: 'criterio', nombre: 'Criterio de selección', nueva: true },
  { id: 'shap', nombre: 'SHAP', nueva: true },
  { id: 'metricas', nombre: 'Métricas' },
  { id: 'ranking', nombre: 'Ranking' }
];

function pintarContexto() {
  const host = $('#panel-contexto');
  host.innerHTML = '';
  const head = el('div', 'phead');
  const tabs = el('div', 'tabs');
  PESTANAS.forEach(p => {
    const b = el('button', S.tab === p.id ? 'on' : '',
      esc(p.nombre) + (p.nueva ? '<span class="newtag">tesina</span>' : ''));
    b.onclick = () => { S.tab = p.id; pintarContexto(); };
    tabs.appendChild(b);
  });
  head.appendChild(tabs);
  head.appendChild(el('div', 'psub', subtitulo()));
  host.appendChild(head);

  const cuerpo = el('div', 'ctx-body');
  cuerpo.id = 'cuerpo-contexto';
  host.appendChild(cuerpo);
  ({
    serie: tabSerie, endemico: tabEndemico, criterio: tabCriterio,
    shap: tabShap, metricas: tabMetricas, ranking: tabRanking
  })[S.tab](cuerpo);
}

function subtitulo() {
  const d = S.dept, ev = D.evaluacion[d] || {};
  switch (S.tab) {
    case 'serie': return 'Seis departamentos de mayor volumen';
    case 'endemico': return 'Año ' + D.se[S.wk][0] + ' · cuartiles de 5 temporadas';
    case 'criterio': return (ev.regla ? 'regla ' + ev.regla + ' · ' : '') +
      (ev.empate ? ev.empate.length + ' de 4 familias empatadas' : 'perfil de la serie');
    case 'shap': return D.meta.shap_disponible.indexOf(d) >= 0
      ? 'atribución de ' + NOMBRE_MODELO[modeloArbol(d)] : 'muestra estratificada (11 departamentos)';
    case 'metricas': return 'horizonte 1 · ' + (ev.n_origenes || '—') + ' orígenes';
    case 'ranking': return '25 departamentos · SE ' + D.se[S.wk][1] + ', ' + D.se[S.wk][0];
  }
  return '';
}

/* --- serie de tiempo --------------------------------------------------- */

function tabSerie(host) {
  const wrap = el('div', 'chart-wrap');
  const plot = el('div', 'plot');
  wrap.appendChild(plot);
  host.appendChild(wrap);

  const medias = DEPTOS.map(d => {
    const s = serie(d).filter(v => v !== null);
    return { d: d, m: s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0 };
  }).sort((a, b) => b.m - a.m);
  const top = medias.slice(0, 6).map(x => x.d);
  if (top.indexOf(S.dept) < 0) { top[5] = S.dept; }

  const t = tramo();
  const x = [];
  for (let i = t.desde; i <= t.hasta; i++) x.push(ms(D.semanas[i]));
  const paleta = ['#12376b', '#0f9c9c', '#c0504d', '#6f9c3d', '#8064a2', '#d99a3a'];
  const lineas = top.map((d, k) => {
    const v = [];
    for (let i = t.desde; i <= t.hasta; i++) v.push(valorSemana(d, i));
    return {
      nombre: NOMBRE[d], color: d === S.dept ? '#1a1d21' : paleta[k],
      ancho: d === S.dept ? 2.2 : 1.3, v: v
    };
  });
  luego(() => chartLineas(plot, { x: x, lineas: lineas }));
}

/* --- canal endémico ---------------------------------------------------- */

function tabEndemico(host) {
  const anio = D.se[S.wk][0];
  const canal = ((D.canal[S.cohorte] || {})[S.dept] || {})[String(anio)];
  const wrap = el('div', 'chart-wrap');
  const plot = el('div', 'plot');
  wrap.appendChild(plot);
  host.appendChild(wrap);
  host.appendChild(el('div', 'foot-note',
    'Zonas construidas con los cuartiles Q1, mediana y Q3 de las mismas semanas en las cinco ' +
    'temporadas previas, excluidas 2020 y 2021. Con solo cinco temporadas de referencia los tres ' +
    'cortes pueden casi coincidir en series de bajo volumen, y entonces la zona de alarma se ' +
    'estrecha y se entra en zona epidémica con facilidad. El canal es contexto de lectura del ' +
    'pronóstico: <b>no interviene en el criterio de selección de modelo</b>.'));

  if (!canal) {
    plot.appendChild(el('div', 'plot-note', 'Sin canal endémico para ' + anio + ' en ' + esc(NOMBRE[S.dept]) + '.'));
    return;
  }
  const x = [], obs = [], q1 = [], q2 = [], q3 = [], techo = [];
  let maxv = 0;
  for (let i = 0; i < D.semanas.length; i++) {
    if (D.se[i][0] !== anio) continue;
    const w = D.se[i][1] - 1;
    x.push(ms(D.semanas[i]));
    const v = valorSemana(S.dept, i);
    obs.push(v);
    if (v !== null && v > maxv) maxv = v;
    q1.push(canal[0][w]); q2.push(canal[1][w]); q3.push(canal[2][w]);
    if (canal[2][w] !== null && canal[2][w] > maxv) maxv = canal[2][w];
  }
  const tope = maxv * 1.1;
  for (let k = 0; k < x.length; k++) techo.push(tope);

  luego(() => chartLineas(plot, {
    x: x,
    zonas: [{ hasta: q1, clase: 'exito' }, { hasta: q2, clase: 'seguridad' },
    { hasta: q3, clase: 'alarma' }, { hasta: techo, clase: 'epidemica' }],
    lineas: [{ nombre: 'Observado ' + anio, color: '#1a1d21', ancho: 2.2, v: obs },
    { nombre: 'Habitual (mediana)', color: '#b9a64a', dash: '5 4', ancho: 1.4, v: q2 }]
  }));
}

/* --- criterio de selección (aportación de la tesina) -------------------- */

function tabCriterio(host) {
  host.classList.add('split');
  const d = S.dept;
  const p = (D.perfil[S.cohorte] || {})[d] || {};
  const c = (D.criterio[S.cohorte] || {})[d] || {};
  const ev = D.evaluacion[d] || {};

  const izq = el('div', 'izq');
  const regla = D.reglas.filter(r => r.id === c.regla)[0];
  if (regla) {
    izq.appendChild(el('div', 'regla-box',
      '<span class="rid">' + esc(regla.id) + '</span><span class="rnom">' + esc(regla.nombre) + '</span>' +
      '<p><b>Condición:</b> ' + esc(regla.condicion) + '<br><b>Recomienda:</b> ' + esc(regla.recomienda) +
      '<br><span class="dim">' + esc(regla.fundamento) + '</span></p>'));
  }

  const ficha = el('div', 'ficha', '<h4>Perfil de la serie</h4>');
  const kv = (k, v) => ficha.appendChild(el('div', 'kv', '<span>' + k + '</span><b>' + v + '</b>'));
  kv('Volumen medio', fmt(p.media_casos, 1) + ' casos/sem');
  kv('Ruido de Poisson', p.media_casos ? fmt(100 / Math.sqrt(p.media_casos), 0) + ' %' : '—');
  kv('Fuerza estacional F<sub>S</sub>', fmt(p.fuerza_estacional, 3));
  kv('Fuerza de tendencia F<sub>T</sub>', fmt(p.fuerza_tendencia, 3));
  kv('Índice de dispersión', fmt(p.indice_dispersion, 2));
  kv('Entropía espectral', fmt(p.entropia_espectral, 3));
  kv('Proporción de ceros', fmt(100 * (p.prop_ceros || 0), 1) + ' %');
  kv('ACF(1) · ACF(52)', fmt(p.acf_1, 2) + ' · ' + fmt(p.acf_52, 2));
  kv('Semanas observadas', fmt(p.n_semanas));
  kv('Estacionaria (ADF)', p.estacionaria ? 'sí' : 'no');
  izq.appendChild(ficha);

  const fam = el('div', 'ficha', '<h4>Familias candidatas</h4>');
  const chips = el('div', 'chips');
  D.meta.candidatos.forEach(m => {
    const descartada = c.descartados && c.descartados[m];
    const empatada = ev.empate && ev.empate.indexOf(m) >= 0;
    const chip = el('span', 'mchip' + (m === c.modelo ? ' rec' : '') + (descartada ? ' out' : ''),
      '<i style="background:' + COLOR[m] + '"></i>' + esc(NOMBRE_MODELO[m]) +
      (m === c.modelo ? ' ★' : '') + (empatada ? ' <span class="tag">empate</span>' : ''));
    if (descartada) chip.title = descartada;
    chips.appendChild(chip);
  });
  fam.appendChild(chips);
  if (c.descartados && Object.keys(c.descartados).length) {
    fam.appendChild(el('div', 'kv',
      '<span>Descartada antes de entrenar</span><b>' + Object.keys(c.descartados).join(', ') + '</b>'));
  }
  fam.appendChild(el('div', 'kv', '<span>Costo del criterio</span><b>' + fmt(c.costo_s, 1) + ' s</b>'));
  fam.appendChild(el('div', 'kv', '<span>Costo de evaluarlo todo</span><b>' + fmt(c.costo_todo_s, 1) + ' s</b>'));
  fam.appendChild(el('div', 'kv', '<span>Ahorro</span><b>' + fmt(c.ahorro_pct, 1) + ' %</b>'));
  izq.appendChild(fam);

  if (ev.empate) {
    const emp = el('div', 'ficha', '<h4>Conjunto estadísticamente equivalente</h4>');
    emp.appendChild(el('div', 'kv', '<span>Mejor observado</span><b>' + esc(NOMBRE_MODELO[ev.mejor] || '—') + '</b>'));
    emp.appendChild(el('div', 'kv', '<span>Indistinguibles de él</span><b>' + ev.empate.length + ' de 4</b>'));
    emp.appendChild(el('div', 'kv',
      '<span>El criterio acierta</span><b>' + (ev.dentro === undefined ? '—' : ev.dentro ? 'sí' : 'no') + '</b>'));
    if (ev.regret_pct !== undefined) {
      emp.appendChild(el('div', 'kv', '<span>Exceso de error</span><b>' + fmt(ev.regret_pct, 2) + ' %</b>'));
    }
    izq.appendChild(emp);
  }
  host.appendChild(izq);

  const der = el('div', 'der');
  const wrap = el('div', 'chart-wrap');
  const plot = el('div', 'plot');
  plot.id = 'plot-criterio';
  wrap.appendChild(plot);
  der.appendChild(wrap);
  der.appendChild(el('div', 'foot-note',
    'Cada punto es un departamento en el plano que define el criterio: volumen medio (escala ' +
    'logarítmica) frente a fuerza estacional. Las zonas sombreadas son las regiones donde se ' +
    'activa cada regla; el color del punto es la familia recomendada. Haga clic en un punto para ' +
    'cambiar de departamento.'));
  host.appendChild(der);
  luego(() => dibujarPlanoCriterio(plot));
}

/** Plano volumen × estacionalidad: la frontera que el criterio dibuja. */
function dibujarPlanoCriterio(host) {
  host.innerHTML = '';
  const W = host.clientWidth || 520, H = host.clientHeight || 260;
  if (W < 60 || H < 60) return;
  const M = { t: 12, r: 14, b: 34, l: 48 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const perfiles = D.perfil[S.cohorte] || {};
  const crit = D.criterio[S.cohorte] || {};

  const X = v => M.l + (Math.log10(Math.max(0.5, v)) - Math.log10(0.5)) / (Math.log10(400) - Math.log10(0.5)) * iw;
  const Y = v => M.t + ih - Math.max(0, Math.min(1, v)) * ih;

  const s = svg('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H });

  // regiones de las reglas
  s.appendChild(svg('rect', { x: X(40), y: M.t + ih * 0.4, width: X(400) - X(40), height: ih * 0.6, fill: COLOR.XGBoost, opacity: .07 }));
  s.appendChild(svg('rect', { x: M.l, y: M.t, width: iw, height: Y(0.60) - M.t, fill: COLOR.SARIMA, opacity: .07 }));
  s.appendChild(svg('rect', { x: M.l, y: M.t, width: X(10) - M.l, height: ih + M.t - M.t, fill: COLOR.SARIMA, opacity: .05 }));

  const g = svg('g', { class: 'ax' });
  [0, 0.2, 0.4, 0.6, 0.8, 1].forEach(v => {
    g.appendChild(svg('line', { class: 'gridline', x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v) }));
    const t = svg('text', { x: M.l - 7, y: Y(v) + 3.5, 'text-anchor': 'end' });
    t.textContent = v.toFixed(1).replace('.', ',');
    g.appendChild(t);
  });
  [1, 10, 40, 100, 400].forEach(v => {
    g.appendChild(svg('line', { class: 'gridline', x1: X(v), x2: X(v), y1: M.t, y2: M.t + ih }));
    const t = svg('text', { x: X(v), y: H - 16, 'text-anchor': 'middle' });
    t.textContent = fmt(v);
    g.appendChild(t);
  });
  let t = svg('text', { x: M.l + iw / 2, y: H - 3, 'text-anchor': 'middle', 'font-size': 9.5, fill: '#9aa1a9' });
  t.textContent = 'volumen medio (casos por semana, escala logarítmica)';
  g.appendChild(t);
  t = svg('text', { x: 11, y: M.t + ih / 2, 'text-anchor': 'middle', 'font-size': 9.5, fill: '#9aa1a9', transform: 'rotate(-90 11 ' + (M.t + ih / 2) + ')' });
  t.textContent = 'fuerza estacional F_S';
  g.appendChild(t);
  s.appendChild(g);

  // umbrales
  [[X(10), 'λ=10'], [X(40), 'λ=40']].forEach(u => {
    s.appendChild(svg('line', { x1: u[0], x2: u[0], y1: M.t, y2: M.t + ih, stroke: '#9aa1a9', 'stroke-dasharray': '4 3' }));
    const lb = svg('text', { x: u[0] + 3, y: M.t + 10, 'font-size': 9, fill: '#9aa1a9' });
    lb.textContent = u[1];
    s.appendChild(lb);
  });
  s.appendChild(svg('line', { x1: M.l, x2: W - M.r, y1: Y(0.60), y2: Y(0.60), stroke: '#9aa1a9', 'stroke-dasharray': '4 3' }));
  const lb = svg('text', { x: W - M.r - 4, y: Y(0.60) - 4, 'font-size': 9, fill: '#9aa1a9', 'text-anchor': 'end' });
  lb.textContent = 'F_S = 0,60';
  s.appendChild(lb);

  DEPTOS.forEach(d => {
    const p = perfiles[d], c = crit[d];
    if (!p || !c) return;
    const sel = d === S.dept;
    const punto = svg('circle', {
      cx: X(p.media_casos), cy: Y(p.fuerza_estacional), r: sel ? 7 : 5,
      fill: COLOR[c.modelo], stroke: sel ? '#1a1d21' : '#fff', 'stroke-width': sel ? 2 : 1.2,
      style: 'cursor:pointer'
    });
    const tt = svg('title');
    tt.textContent = NOMBRE[d] + ' · ' + fmt(p.media_casos, 1) + ' casos/sem · F_S ' +
      fmt(p.fuerza_estacional, 3) + ' · ' + c.regla + ' → ' + NOMBRE_MODELO[c.modelo];
    punto.appendChild(tt);
    punto.addEventListener('click', () => { S.dept = d; pintar(); });
    s.appendChild(punto);
    if (sel) {
      const nm = svg('text', {
        x: X(p.media_casos), y: Y(p.fuerza_estacional) - 11, 'text-anchor': 'middle',
        'font-size': 10, 'font-weight': 600, fill: '#1a1d21'
      });
      nm.textContent = NOMBRE[d];
      s.appendChild(nm);
    }
  });
  host.appendChild(s);

  const lg = el('div', 'plot-legend');
  ['SARIMA', 'RandomForest', 'XGBoost'].forEach(m => {
    lg.appendChild(el('div', 'lgd', '<i style="background:' + COLOR[m] + ';height:8px;border-radius:4px"></i><span>' +
      esc(NOMBRE_MODELO[m]) + '</span>'));
  });
  host.appendChild(lg);
}

/* --- explicabilidad SHAP (aportación de la tesina) ---------------------- */

function modeloArbol(d) {
  const rec = recomendado(d);
  if (rec === 'XGBoost' || rec === 'RandomForest') return rec;
  return 'RandomForest';
}

function tabShap(host) {
  host.classList.add('split');
  const d = S.dept;
  const nodo = S.cohorte === 'under5' ? (D.shap[d] || {})[modeloArbol(d)] : null;

  const izq = el('div', 'izq');
  if (!nodo) {
    izq.appendChild(el('div', 'aviso info',
      S.cohorte !== 'under5'
        ? '<b>SHAP v0 — prototipo.</b> La atribución se calculó sobre <b>menores de 5 años</b>, ' +
          'que es el grupo sobre el que corre la validación de esta tesina.'
        : '<b>SHAP v0 — prototipo.</b> La atribución se calculó sobre la muestra estratificada de ' +
          D.meta.shap_disponible.length + ' departamentos, que es el alcance declarado de esta etapa. ' +
          esc(NOMBRE[d]) + ' no está en ella.'));
    const chips = el('div', 'chips');
    D.meta.shap_disponible.forEach(x => {
      const c = el('span', 'mchip', esc(NOMBRE[x]));
      c.style.cursor = 'pointer';
      c.onclick = () => { S.dept = x; pintar(); };
      chips.appendChild(c);
    });
    izq.appendChild(el('div', 'ficha', '<h4>Departamentos con atribución calculada</h4>'));
    izq.appendChild(chips);
    host.appendChild(izq);
    const vacio = el('div', 'der');
    vacio.appendChild(el('div', 'plot-note',
      'Elija uno de los departamentos de la muestra para ver su descomposición.'));
    host.appendChild(vacio);
    return;
  }

  izq.appendChild(el('div', 'ficha', '<h4>Atribución global por familia</h4>'));
  const cajaFam = el('div');
  izq.appendChild(cajaFam);
  const ETIQ = {
    inercia_corta: 'Inercia corta', ventanas_moviles: 'Ventanas móviles',
    memoria_media: 'Memoria media', calendario_estacional: 'Calendario', rezago_anual: 'Rezago anual',
    tendencia: 'Tendencia'
  };
  const filas = Object.keys(nodo.familias)
    .map(k => ({ nombre: ETIQ[k] || k, valor: nodo.familias[k] || 0, color: '#4f81bd' }))
    .sort((a, b) => b.valor - a.valor);
  barras(cajaFam, filas, { formato: v => fmt(100 * v, 1) + ' %' });

  const p = (D.perfil[S.cohorte] || {})[d] || {};
  const corta = (nodo.familias.inercia_corta || 0) + (nodo.familias.ventanas_moviles || 0);
  const cal = (nodo.familias.calendario_estacional || 0) + (nodo.familias.rezago_anual || 0);
  const lectura = el('div', 'ficha', '<h4>Lectura</h4>');
  lectura.appendChild(el('div', 'kv', '<span>Estructura corta</span><b>' + fmt(100 * corta, 1) + ' %</b>'));
  lectura.appendChild(el('div', 'kv', '<span>Estructura estacional</span><b>' + fmt(100 * cal, 1) + ' %</b>'));
  lectura.appendChild(el('div', 'kv', '<span>F<sub>S</sub> de la serie</span><b>' + fmt(p.fuerza_estacional, 3) + '</b>'));
  lectura.appendChild(el('div', 'kv', '<span>Volumen</span><b>' + fmt(p.media_casos, 1) + '</b>'));
  izq.appendChild(lectura);

  // El rezago anual exactamente en cero no es un hallazgo sobre la serie sino
  // sobre la configuración: lag_52 no está entre las variables de XGBoost.
  if (modeloArbol(d) === 'XGBoost' && !nodo.familias.rezago_anual) {
    izq.appendChild(el('div', 'aviso',
      'La atribución al rezago anual es exactamente cero porque <b>lag_52 no forma parte de las ' +
      'variables de XGBoost</b> en esta corrida, no porque la serie carezca de memoria anual. Es ' +
      'una limitación de configuración, corregible, y queda declarada como tal.'));
  }

  const corr = D.shap_corr.filter(c => c.modelo === modeloArbol(d) && c.familia === 'inercia_corta')[0];
  if (corr) {
    izq.appendChild(el('div', 'aviso info',
      'Sobre los ' + D.meta.shap_disponible.length + ' departamentos de la muestra, la atribución a la ' +
      'inercia corta correlaciona <b>r = ' + fmt(corr.r_log_volumen, 3) + '</b> con el logaritmo del ' +
      'volumen y <b>r = ' + fmt(corr.r_fuerza_estacional, 3) + '</b> con la fuerza estacional. El ' +
      'mecanismo sigue al volumen, que es justamente la variable sobre la que decide la regla R1.'));
  }
  host.appendChild(izq);

  const der = el('div', 'der');
  const wrap = el('div', 'chart-wrap');
  const plot = el('div', 'plot');
  plot.id = 'plot-shap';
  wrap.appendChild(plot);
  der.appendChild(wrap);
  if (nodo.local) {
    der.appendChild(el('div', 'foot-note',
      'Descomposición de <b>una</b> predicción: la semana de mayor valor entre las evaluadas. ' +
      'Por la propiedad de eficiencia, las contribuciones suman exactamente la diferencia entre la ' +
      'predicción (' + fmt(nodo.local.prediccion, 1) + ') y el valor base (' + fmt(nodo.local.valor_base, 1) + ')' +
      (nodo.local.eficiencia_verificada ? ', y esa identidad se verifica en el código' : '') + '.'));
  }
  host.appendChild(der);
  luego(() => dibujarCascadaShap(plot, nodo));
}

/** Cascada: cómo cada variable mueve la predicción desde el valor base. */
function dibujarCascadaShap(host, nodo) {
  host.innerHTML = '';
  if (!nodo.local) {
    host.appendChild(el('div', 'plot-note', 'Sin explicación local almacenada para esta serie.'));
    return;
  }
  const loc = nodo.local;
  const items = loc.contribuciones.slice(0, 8).map(c => ({
    etiqueta: c.variable + ' = ' + fmt(c.valor_variable, 1), valor: c.contribucion
  }));
  if (loc.n_variables_restantes) {
    items.push({ etiqueta: 'otras ' + loc.n_variables_restantes + ' variables', valor: loc.resto_agrupado });
  }
  const W = host.clientWidth || 520, H = host.clientHeight || 260;
  if (W < 60 || H < 60) return;
  const M = { t: 16, r: 48, b: 26, l: 136 };
  const iw = W - M.l - M.r, alto = Math.max(13, Math.min(26, (H - M.t - M.b) / items.length - 3));

  let acum = loc.valor_base, lo = loc.valor_base, hi = loc.valor_base;
  const tramos = items.map(it => {
    const a = acum; acum += it.valor;
    lo = Math.min(lo, a, acum); hi = Math.max(hi, a, acum);
    return { desde: a, hasta: acum, etiqueta: it.etiqueta, valor: it.valor };
  });
  const pad = (hi - lo) * 0.12 + 1;
  const X = v => M.l + (v - (lo - pad)) / ((hi + pad) - (lo - pad)) * iw;

  const s = svg('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H });
  s.appendChild(svg('line', { x1: X(loc.valor_base), x2: X(loc.valor_base), y1: M.t - 4, y2: H - M.b, stroke: '#9aa1a9', 'stroke-dasharray': '3 3' }));
  let tb = svg('text', { x: X(loc.valor_base), y: H - M.b + 14, 'text-anchor': 'middle', 'font-size': 9.5, fill: '#6b7280' });
  tb.textContent = 'valor base ' + fmt(loc.valor_base, 1);
  s.appendChild(tb);
  s.appendChild(svg('line', { x1: X(loc.prediccion), x2: X(loc.prediccion), y1: M.t - 4, y2: H - M.b, stroke: '#1a1d21', 'stroke-width': 1.4 }));
  tb = svg('text', { x: X(loc.prediccion), y: M.t - 6, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 600 });
  tb.textContent = 'predicción ' + fmt(loc.prediccion, 1);
  s.appendChild(tb);

  tramos.forEach((t, k) => {
    const y = M.t + k * (alto + 3);
    const x1 = X(Math.min(t.desde, t.hasta)), x2 = X(Math.max(t.desde, t.hasta));
    s.appendChild(svg('rect', {
      x: x1, y: y, width: Math.max(1.5, x2 - x1), height: alto,
      fill: t.valor >= 0 ? '#c0504d' : '#4f81bd', rx: 2
    }));
    const et = svg('text', { x: M.l - 8, y: y + alto / 2 + 3.5, 'text-anchor': 'end', 'font-size': 10, fill: '#4b5158' });
    et.textContent = t.etiqueta;
    s.appendChild(et);
    const vv = svg('text', {
      x: (t.valor >= 0 ? x2 + 5 : x1 - 5), y: y + alto / 2 + 3.5,
      'text-anchor': t.valor >= 0 ? 'start' : 'end', 'font-size': 9.5, fill: '#4b5158'
    });
    vv.textContent = (t.valor >= 0 ? '+' : '−') + fmt(Math.abs(t.valor), 2);
    s.appendChild(vv);
  });
  host.appendChild(s);
  const cap = el('div', 'origin-tag');
  cap.style.left = '10px'; cap.style.top = '2px';
  cap.textContent = loc.departamento + ' · ' + NOMBRE_MODELO[loc.modelo] + ' · semana del ' + fechaCorta(loc.semana);
  host.appendChild(cap);
}

/* --- métricas ---------------------------------------------------------- */

function tabMetricas(host) {
  const d = S.dept;
  const met = S.cohorte === 'under5' ? (D.metricas[d] || {}) : {};
  const ev = D.evaluacion[d] || {};
  const crit = (D.criterio[S.cohorte] || {})[d] || {};
  const costos = (D.meta.configuracion.costo_s) || {};
  const naive = met.SeasonalNaive;

  if (!Object.keys(met).length) {
    host.appendChild(el('div', 'aviso',
      'La comparación retrospectiva de las cinco familias se ejecutó sobre <b>menores de 5 años</b> ' +
      '(125 corridas, 25 departamentos). Para 60 y más todavía no hay validación: el criterio sí ' +
      'puede recomendar, porque decide a partir del perfil de la serie y no de su historial de error, ' +
      'pero esa recomendación aún no está contrastada.'));
    return;
  }

  const wrap = el('div', 'table-wrap');
  let h = '<table><thead><tr><th>Familia</th><th>MAE</th><th>RMSE</th><th>MAPE</th>' +
    '<th>vs ingenuo</th><th>Costo</th><th>Estado</th></tr></thead><tbody>';
  D.meta.modelos.forEach(m => {
    const x = met[m];
    if (!x) return;
    const mejor = ev.mejor === m;
    const empatado = ev.empate && ev.empate.indexOf(m) >= 0;
    const descartada = crit.descartados && crit.descartados[m];
    let estado = '';
    if (m === crit.modelo) estado += '<span class="tag">★ recomendado</span> ';
    if (mejor) estado += '<span class="tag">mejor observado</span> ';
    else if (empatado) estado += '<span class="tag">empatado</span> ';
    if (descartada) estado += '<span class="tag warn">descartado R0</span>';
    if (m === 'SeasonalNaive') estado = '<span class="tag off">referencia</span>';
    h += '<tr class="' + (empatado ? 'act' : '') + (descartada ? ' mut' : '') + '">' +
      '<td><div class="mname"><i class="swatch" style="background:' + COLOR[m] + '"></i>' + esc(NOMBRE_MODELO[m]) + '</div></td>' +
      '<td>' + fmt(x.mae, 3) + '</td><td>' + fmt(x.rmse, 3) + '</td><td>' + fmt(x.mape, 2) + ' %</td>' +
      '<td>' + (naive && naive.mae && m !== 'SeasonalNaive' ? pct(100 * (naive.mae - x.mae) / naive.mae, 1) : '—') + '</td>' +
      '<td>' + fmt(costos[m], 1) + ' s</td><td style="text-align:left">' + estado + '</td></tr>';
  });
  h += '</tbody></table>';
  wrap.innerHTML = h;
  host.appendChild(wrap);
  host.appendChild(el('div', 'foot-note',
    'Horizonte 1 semana sobre ' + (ev.n_origenes || '—') + ' orígenes de validación con origen móvil. ' +
    'Las filas resaltadas forman el <b>conjunto estadísticamente equivalente</b> al mejor: una prueba t ' +
    'pareada sobre las diferencias de error por origen no las distingue de él al 5 %. En esta serie son ' +
    (ev.empate ? ev.empate.length : '—') + ' de 4; en 19 de los 25 departamentos son tres o más, y esa es ' +
    'la razón por la que el criterio no persigue el error mínimo sino una familia suficientemente buena.'));
}

/* --- ranking ----------------------------------------------------------- */

let ordenRanking = { col: 'casos', desc: true };

function tabRanking(host) {
  const i = S.wk;
  const filas = DEPTOS.map(d => {
    const v = valorSemana(d, i);
    const pob = poblacion(d, D.se[i][0]);
    const u4 = suma(d, i, 4), p4 = suma(d, i - 4, 4);
    const c = corte(d, i);
    const crit = (D.criterio[S.cohorte] || {})[d] || {};
    const ev = D.evaluacion[d] || {};
    return {
      d: d, casos: v, tasa: (pob && v !== null) ? v * 100000 / pob : null,
      tend: (u4 !== null && p4) ? 100 * (u4 - p4) / p4 : null,
      zona: zonaDe(v, c), regla: crit.regla, modelo: crit.modelo,
      empate: ev.empate ? ev.empate.length : null, dentro: ev.dentro
    };
  });
  const col = ordenRanking.col;
  filas.sort((a, b) => {
    const x = a[col], y = b[col];
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    if (typeof x === 'string') return (ordenRanking.desc ? -1 : 1) * x.localeCompare(y);
    return (ordenRanking.desc ? -1 : 1) * (x - y);
  });

  const wrap = el('div', 'table-wrap');
  const cols = [['d', 'Departamento'], ['zona', 'Zona'], ['casos', 'Casos'], ['tasa', 'Tasa/100k'],
  ['tend', 'Tend. 4 sem'], ['regla', 'Regla'], ['modelo', 'Recomendado ★'], ['empate', 'Empatados']];
  let h = '<table><thead><tr>';
  cols.forEach(c => { h += '<th data-c="' + c[0] + '" style="cursor:pointer">' + c[1] + (col === c[0] ? (ordenRanking.desc ? ' ▾' : ' ▴') : '') + '</th>'; });
  h += '</tr></thead><tbody>';
  filas.forEach(f => {
    h += '<tr class="clickable ' + (f.d === S.dept ? 'act' : '') + '" data-d="' + f.d + '">' +
      '<td>' + esc(NOMBRE[f.d]) + '</td>' +
      '<td style="text-align:left">' + (f.zona ? '<span class="zone-badge ' + f.zona + '">' + ZONA_NOMBRE[f.zona] + '</span>' : '—') + '</td>' +
      '<td>' + fmt(f.casos) + '</td><td>' + fmt(f.tasa, 2) + '</td>' +
      '<td class="' + (f.tend > 0 ? 'up' : f.tend < 0 ? 'down' : '') + '">' + (f.tend === null ? '—' : pct(f.tend)) + '</td>' +
      '<td>' + (f.regla || '—') + '</td>' +
      '<td style="text-align:left"><div class="mname"><i class="swatch" style="background:' + COLOR[f.modelo] + '"></i>' +
      esc(NOMBRE_MODELO[f.modelo] || '—') + (f.dentro === false ? ' <span class="tag warn">fuera</span>' : '') + '</div></td>' +
      '<td>' + (f.empate === null ? '—' : f.empate + ' de 4') + '</td></tr>';
  });
  h += '</tbody></table>';
  wrap.innerHTML = h;
  host.appendChild(wrap);
  wrap.querySelectorAll('th[data-c]').forEach(th => {
    th.onclick = () => {
      const c = th.dataset.c;
      ordenRanking = { col: c, desc: ordenRanking.col === c ? !ordenRanking.desc : true };
      pintarContexto();
    };
  });
  wrap.querySelectorAll('tr[data-d]').forEach(tr => {
    tr.onclick = () => { S.dept = tr.dataset.d; pintar(); };
  });
  host.appendChild(el('div', 'foot-note',
    'La columna <b>Recomendado ★</b> es la salida del criterio aplicado al perfil de cada serie, ' +
    'sin mirar su historial de error. «Fuera» marca el único departamento en el que la familia ' +
    'recomendada no pertenece al conjunto empatado.'));
}

/* ───────────────────────── 10. arranque ───────────────────────── */

function pintar() {
  pintarTitulares();
  pintarControles();
  pintarMapa();
  pintarPronostico();
  pintarContexto();
  sincronizarBotones();
}

function cabecera() {
  const i = ultimaSemanaUtil();
  $('#chip-datos').innerHTML = 'Datos históricos · <b>hasta la SE ' + D.se[i][1] + ', ' + D.se[i][0] + '</b>';
}

function montarMapaControles() {
  $('#mm-magnitud').onclick = () => { S.modo = 'magnitud'; pintar(); };
  $('#mm-zona').onclick = () => { S.modo = 'zona'; pintar(); };
  $('#mt-nombres').onclick = () => { S.nombres = !S.nombres; pintar(); };
  $('#mt-valores').onclick = () => { S.valores = !S.valores; pintar(); };
  $('#zoom-mas').onclick = () => { S.zoom = Math.min(6, S.zoom * 1.35); aplicarZoom(); };
  $('#zoom-menos').onclick = () => {
    S.zoom = Math.max(1, S.zoom / 1.35);
    if (S.zoom === 1) { S.panx = 0; S.pany = 0; }
    aplicarZoom();
  };
}

function sincronizarBotones() {
  $('#mm-magnitud').classList.toggle('on', S.modo === 'magnitud');
  $('#mm-zona').classList.toggle('on', S.modo === 'zona');
  $('#mt-nombres').classList.toggle('on', S.nombres);
  $('#mt-valores').classList.toggle('on', S.valores);
}

let temporizador = null;
window.addEventListener('resize', () => {
  clearTimeout(temporizador);
  temporizador = setTimeout(() => { pintarMapa(); dibujarPronostico(); pintarContexto(); }, 160);
});

cabecera();
construirMapa();
montarMapaControles();
pintar();

})();
