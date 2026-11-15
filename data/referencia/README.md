# data/referencia/

Datos de referencia que el modelo lee directamente. Sí se versionan.

## `igac/capacidad_uso_valle.geojson`

Capacidad de Uso de las Tierras del Valle del Cauca (IGAC, 2006, escala 1:100.000). 3.723 polígonos en EPSG:4326 con `CLASE`, `SUBCLASE`, `GRUPO_MANEJO`, `UCS`, `PRINCIPALES_LIMITANES`, `USOS_RECOMENDADOS`, `PRACTICAS_MANEJO` y `AREA` (ha).

- Se regenera con `python -m rotarapp.datos.igac_capacidad` (servicio público del IGAC).
- Geometría simplificada a ~11 m y redondeada a ~1 m. A esta escala la precisión posicional es de decenas de metros, así que no se pierde información.
- La unidad mínima cartografiable a 1:100.000 es de unas 25 ha: a una finca pequeña normalmente le cae una sola clase. Sirve como **filtro para toda la finca**, no para dividirla por dentro.
- Clases especiales: `CA` cuerpo de agua, `ZU` zona urbana, `REVISAR` (2 polígonos) y 1 polígono sin clase.

| Clase | Polígonos | Área (ha) | % |
|---|---|---|---|
| II | 427 | 85.196 | 4,1 |
| III | 527 | 110.892 | 5,3 |
| IV | 642 | 111.268 | 5,3 |
| V | 54 | 8.780 | 0,4 |
| VI | 690 | 315.216 | 15,1 |
| VII | 690 | 1.171.327 | 56,2 |
| VIII | 404 | 216.793 | 10,4 |
| CA, ZU, otros | 289 | 63.023 | 3,0 |

Ojo: el HTML informativo (estudio IGAC-CVC) reporta solo 5.686 ha de clase 2. Este mapa tiene 85.196 ha de clase II. Para el modelo manda la geometría de este mapa.

## `cultivos/`

| Archivo | Contenido |
|---|---|
| `Rangos_cultivos_Valle_del_Cauca_v1.xlsx` | Tabla original del equipo: 17 cultivos |
| `Rangos_cultivos_Valle_del_Cauca_v2.xlsx` | Ampliada: 24 cultivos (+7 leguminosas), atributos de rotación, parámetros numéricos, precios y plagas compartidas. La genera `scripts/ampliar_cultivos.py` |
| `cultivos_atributos.csv` | Hoja `Rangos` de la v2 |
| `cultivos_parametros.csv` | Rangos numéricos (T, agua, HR, pH, altitud), Kc, fija N y rol: **lo que lee el simulador** |
| `precios_sipsa_2026_resumen.csv` | Mediana, P10 y P90 de precio por cultivo |
| `plagas_compartidas.csv` | Matriz cultivo × cultivo: n.º de agentes (por género) en común |

Vacíos: los valores de las leguminosas agregadas, los Kc supuestos y los rendimientos típicos son conocimiento general no verificado. Faltan los **costos de producción** y los precios de caña, sorgo, algodón, cacao, café, panela, soya, maní, caupí y guandul.

## `precios/sipsa_valle_2026.csv`

Precios mayoristas SIPSA (DANE) en 8 mercados del Valle, semanales del 2 de enero al 13 de noviembre de 2026, en COP/kg. Son precios de mercado, **no costos de producción ni precio en finca**.

## `plagas/plagas_valle_del_cauca.json`

99 plagas y enfermedades de los 17 cultivos más la soya, con nivel de evidencia (A: Valle, B: Colombia, C: general), hospedantes compartidos y fuentes. Las leguminosas agregadas en la v2 (salvo la soya) no están incluidas.
