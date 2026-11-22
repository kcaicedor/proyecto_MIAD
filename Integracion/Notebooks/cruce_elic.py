"""Homologación y cruce de ELIC, usando un DataFrame completo ya cargado.

En tu notebook:
    from cruce_elic import preparar_elic, resumen_elic
    import pandas as pd
    archivo = 'anex-ELIC-SerieTipoBaseMun-ago2026 - muestra.xlsx'
    guia = pd.read_excel(archivo, sheet_name='guía', header=None)
    municipios = pd.read_excel(archivo, sheet_name='lista municipios', dtype=str)
    df_codigos, df_analisis, auditoria = preparar_elic(df_elic, guia, municipios)
    print(resumen_elic(df_analisis, ['ano', 'departamento', 'destino']))

No carga la hoja 'elic': df_elic debe contener tu base completa.
"""

import re
import unicodedata
import pandas as pd

CATEGORICAS = [
    'obj_tra', 'clase_suelo', 'modalidad', 'destino',
    'tipo_vivi', 'vis_novis', 'estrato', 'cobertura'
]
LLAVES = ['cod_depto', 'cod_muni']


def normalizar_nombre(valor):
    """Normaliza encabezados: 'año' -> 'ano', 'VIS_NVIS' -> 'vis_novis'."""
    texto = unicodedata.normalize('NFKD', str(valor).strip().lower())
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r'[^a-z0-9]+', '_', texto).strip('_')
    return {'vis_nvis': 'vis_novis'}.get(texto, texto)


def normalizar_codigo(valor, ancho=0):
    """Equivale 1, 1.0 y '1'; conserva el código especial '1000+'."""
    if pd.isna(valor) or str(valor).strip() == '':
        return pd.NA
    codigo = str(valor).strip()
    if re.fullmatch(r'\d+(?:\.0+)?', codigo):
        codigo = str(int(float(codigo)))
    return codigo.zfill(ancho) if ancho else codigo


def obtener_catalogos(guia):
    """En la guía los encabezados de sección están en B y sus valores en B:C."""
    if guia.shape[1] < 3:
        raise ValueError("Carga la hoja 'guía' con header=None y todas sus columnas.")
    catalogos = {variable: {} for variable in CATEGORICAS}
    variable_activa = None

    for valor, descripcion in guia.iloc[:, [1, 2]].itertuples(index=False, name=None):
        encabezado = normalizar_nombre(valor) if pd.notna(valor) else ''
        if encabezado in CATEGORICAS:
            variable_activa = encabezado
            continue
        if encabezado in {'ano', 'mes', 'cod_depto', 'cod_muni', 'area',
                          'unidades', 'licencias'}:
            variable_activa = None
            continue
        if variable_activa is None or pd.isna(descripcion):
            continue

        codigo = normalizar_codigo(valor)
        if pd.isna(codigo) or not re.fullmatch(r'\d+\+?', codigo):
            continue
        etiqueta = str(descripcion).strip()
        anterior = catalogos[variable_activa].get(codigo)
        if anterior is not None and anterior != etiqueta:
            raise ValueError(f'Código ambiguo: {variable_activa}={codigo}')
        catalogos[variable_activa][codigo] = etiqueta

    sin_catalogo = [v for v, dic in catalogos.items() if not dic]
    if sin_catalogo:
        raise ValueError(f'No se encontraron equivalencias para {sin_catalogo}')
    return catalogos


def preparar_elic(df_elic, df_guia, df_municipios):
    """Devuelve base con códigos, base etiquetada y auditoría de códigos no encontrados."""
    base = df_elic.copy()
    municipios = df_municipios.copy()
    base.columns = [normalizar_nombre(c) for c in base.columns]
    municipios.columns = [normalizar_nombre(c) for c in municipios.columns]

    requeridas = set(CATEGORICAS + LLAVES + ['ano', 'mes', 'area', 'unidades', 'licencias'])
    faltantes = requeridas.difference(base.columns)
    if faltantes:
        raise ValueError(f'Faltan columnas en tu base ELIC: {sorted(faltantes)}')
    faltantes_mun = (set(LLAVES + ['departamento', 'municipio', 'cobertura'])
                     .difference(municipios.columns))
    if faltantes_mun:
        raise ValueError(f'Faltan columnas en municipios: {sorted(faltantes_mun)}')

    for df in (base, municipios):
        df['cod_depto'] = df['cod_depto'].map(lambda x: normalizar_codigo(x, 2)).astype('string')
        df['cod_muni'] = df['cod_muni'].map(lambda x: normalizar_codigo(x, 3)).astype('string')

    if municipios.duplicated(LLAVES).any():
        raise ValueError('El catálogo de municipios tiene códigos DIVIPOLA duplicados.')

    # Renombrar cobertura para no sobreescribir la registrada en ELIC.
    mun_ref = municipios[LLAVES + ['departamento', 'municipio', 'cobertura']].rename(
        columns={'cobertura': 'cobertura_municipio'}
    )
    base = base.merge(mun_ref, on=LLAVES, how='left', validate='many_to_one', indicator=True)
    base['municipio_encontrado'] = base['_merge'].eq('both')
    base = base.drop(columns='_merge')
    base['divipola'] = base['cod_depto'] + base['cod_muni']

    for c in ('ano', 'mes', 'area', 'unidades', 'licencias'):
        base[c] = pd.to_numeric(base[c], errors='coerce')
    base['fecha'] = pd.to_datetime(
        {'year': base['ano'], 'month': base['mes'], 'day': 1}, errors='coerce'
    )
    base['cobertura_municipio'] = base['cobertura_municipio'].map(normalizar_codigo).astype('string')
    base['cobertura'] = base['cobertura'].map(normalizar_codigo).astype('string')
    base['cobertura_coincide'] = (
        base['cobertura'].eq(base['cobertura_municipio'])
        .where(base['municipio_encontrado'])
    )

    catalogos = obtener_catalogos(df_guia)
    auditoria_registros = []
    for variable, equivalencias in catalogos.items():
        codigos = base[variable].map(normalizar_codigo).astype('string')
        base[variable] = codigos
        descripcion = codigos.map(equivalencias).astype('string')

        # Nota de la guía: antes de 2013 no se identificaba VIP por separado.
        if variable == 'vis_novis':
            anteriores = (base['fecha'].lt('2013-01-01') & codigos.eq('1')).fillna(False)
            descripcion.loc[anteriores] = 'VIS (sin desagregación VIP, antes de 2013)'

        base[f'{variable}_desc'] = descripcion
        no_encontrados = codigos[codigos.notna() & descripcion.isna()]
        for codigo, cantidad in no_encontrados.value_counts().items():
            auditoria_registros.append({
                'variable': variable, 'codigo': codigo, 'registros': int(cantidad)
            })

    auditoria = pd.DataFrame(auditoria_registros, columns=['variable', 'codigo', 'registros'])

    # Conserva código original en *_codigo y reemplaza las variables por etiquetas.
    df_analisis = base.copy()
    for variable in CATEGORICAS:
        df_analisis[f'{variable}_codigo'] = df_analisis[variable]
        df_analisis[variable] = df_analisis[f'{variable}_desc']
    df_analisis = df_analisis.drop(columns=[f'{v}_desc' for v in CATEGORICAS])

    return base, df_analisis, auditoria


def resumen_elic(df_analisis, dimensiones):
    """Agrega indicadores sobre cualquier combinación de dimensiones."""
    return (
        df_analisis.groupby(dimensiones, dropna=False)
        .agg(registros=('area', 'size'), area_m2=('area', 'sum'),
             unidades=('unidades', 'sum'), licencias=('licencias', 'sum'))
        .reset_index()
        .sort_values('area_m2', ascending=False)
    )
