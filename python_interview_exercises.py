"""
=============================================================================
Python Interview Exercises for Data Engineers (Junior/Trainee)
=============================================================================
Enfoque: Manipular datos, moverlos de un lado a otro, hacerlo eficientemente.

NO buscan algoritmos de IA complejos.
SÍ buscan que sepas limpiar datos, transformarlos y elegir estructuras eficientes.

Cada ejercicio tiene:
1. Enunciado del problema
2. Espacio para tu solución
3. Solución explicada
4. Tips de entrevista
"""

# =============================================================================
# SECTION 1: Las 4 Estructuras de Datos "Sagradas"
# =============================================================================

print("=" * 70)
print("SECTION 1: Estructuras de Datos Fundamentales")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 LISTAS - El pan de cada día
# -----------------------------------------------------------------------------

print("\n--- 1.1 LISTAS ---")

# EJERCICIO 1.1.1: Invertir una lista sin usar reverse()
# Input: [1, 2, 3, 4, 5]
# Output: [5, 4, 3, 2, 1]

lista = [1, 2, 3, 4, 5]
# TU CÓDIGO:


# SOLUCIÓN:
lista_invertida = lista[::-1]
print(f"Lista invertida: {lista_invertida}")
# TIP: [::-1] es slicing con step negativo. Muy pythonic.


# EJERCICIO 1.1.2: Eliminar duplicados manteniendo el orden
# Input: [1, 2, 2, 3, 1, 4, 3, 5]
# Output: [1, 2, 3, 4, 5]

datos = [1, 2, 2, 3, 1, 4, 3, 5]
# TU CÓDIGO:


# SOLUCIÓN:
# Método 1: Con set (pierde orden en Python < 3.7)
sin_duplicados_set = list(set(datos))  # NO garantiza orden

# Método 2: Manteniendo orden (PREFERIDO en entrevistas)
vistos = set()
sin_duplicados = []
for item in datos:
    if item not in vistos:
        vistos.add(item)
        sin_duplicados.append(item)
print(f"Sin duplicados (ordenado): {sin_duplicados}")

# Método 3: One-liner con dict (Python 3.7+, dict mantiene orden)
sin_duplicados_dict = list(dict.fromkeys(datos))
print(f"Sin duplicados (dict): {sin_duplicados_dict}")
# TIP: Menciona la complejidad - set lookup es O(1)


# EJERCICIO 1.1.3: Encontrar el segundo elemento más grande
# Input: [5, 2, 8, 1, 9, 3, 9]
# Output: 8 (no el duplicado de 9)

numeros = [5, 2, 8, 1, 9, 3, 9]
# TU CÓDIGO:


# SOLUCIÓN:
# Método 1: Ordenar (O(n log n))
segundo_mayor = sorted(set(numeros), reverse=True)[1]
print(f"Segundo mayor: {segundo_mayor}")

# Método 2: Sin ordenar (O(n)) - MÁS EFICIENTE
def segundo_mayor_eficiente(nums):
    primero = segundo = float('-inf')
    for n in nums:
        if n > primero:
            segundo = primero
            primero = n
        elif n > segundo and n != primero:
            segundo = n
    return segundo

print(f"Segundo mayor (eficiente): {segundo_mayor_eficiente(numeros)}")
# TIP: Si mencionas la complejidad O(n) vs O(n log n), impresionas.


# EJERCICIO 1.1.4: Rotar una lista k posiciones a la derecha
# Input: [1, 2, 3, 4, 5], k=2
# Output: [4, 5, 1, 2, 3]

lista = [1, 2, 3, 4, 5]
k = 2
# TU CÓDIGO:


# SOLUCIÓN:
k = k % len(lista)  # Por si k > len(lista)
rotada = lista[-k:] + lista[:-k]
print(f"Lista rotada {k} posiciones: {rotada}")


# -----------------------------------------------------------------------------
# 1.2 DICCIONARIOS - LA MÁS IMPORTANTE PARA DATA ENGINEER
# -----------------------------------------------------------------------------

print("\n--- 1.2 DICCIONARIOS ---")

# EJERCICIO 1.2.1: Contador de frecuencias (CLÁSICO DE ENTREVISTA)
# Input: ['compra', 'venta', 'compra', 'reembolso', 'compra']
# Output: {'compra': 3, 'venta': 1, 'reembolso': 1}

transacciones = ['compra', 'venta', 'compra', 'reembolso', 'compra']
# TU CÓDIGO:


# SOLUCIÓN:
# Método 1: Manual con .get()
conteo = {}
for t in transacciones:
    conteo[t] = conteo.get(t, 0) + 1
print(f"Conteo manual: {conteo}")

# Método 2: Con defaultdict
from collections import defaultdict
conteo_dd = defaultdict(int)
for t in transacciones:
    conteo_dd[t] += 1
print(f"Conteo defaultdict: {dict(conteo_dd)}")

# Método 3: Con Counter (PREFERIDO - menciona que conoces la librería)
from collections import Counter
conteo_counter = Counter(transacciones)
print(f"Conteo Counter: {dict(conteo_counter)}")
# TIP: Counter tiene métodos útiles como .most_common(n)


# EJERCICIO 1.2.2: Agrupar elementos por una clave (GROUP BY manual)
# Input: [{'name': 'Alice', 'dept': 'IT'}, {'name': 'Bob', 'dept': 'HR'},
#         {'name': 'Charlie', 'dept': 'IT'}]
# Output: {'IT': ['Alice', 'Charlie'], 'HR': ['Bob']}

empleados = [
    {'name': 'Alice', 'dept': 'IT'},
    {'name': 'Bob', 'dept': 'HR'},
    {'name': 'Charlie', 'dept': 'IT'},
    {'name': 'Diana', 'dept': 'HR'}
]
# TU CÓDIGO:


# SOLUCIÓN:
por_depto = defaultdict(list)
for emp in empleados:
    por_depto[emp['dept']].append(emp['name'])
print(f"Agrupado por depto: {dict(por_depto)}")
# TIP: Esto es un GROUP BY - menciona que en producción usarías pandas/SQL


# EJERCICIO 1.2.3: Invertir un diccionario (value -> key)
# Input: {'a': 1, 'b': 2, 'c': 3}
# Output: {1: 'a', 2: 'b', 3: 'c'}

original = {'a': 1, 'b': 2, 'c': 3}
# TU CÓDIGO:


# SOLUCIÓN:
invertido = {v: k for k, v in original.items()}
print(f"Diccionario invertido: {invertido}")
# TIP: ¿Qué pasa si hay valores duplicados? El último gana.


# EJERCICIO 1.2.4: Merge de dos diccionarios con suma de valores
# Input: {'a': 1, 'b': 2}, {'b': 3, 'c': 4}
# Output: {'a': 1, 'b': 5, 'c': 4}

dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
# TU CÓDIGO:


# SOLUCIÓN:
# Método 1: Manual
merged = dict1.copy()
for k, v in dict2.items():
    merged[k] = merged.get(k, 0) + v
print(f"Merged (suma): {merged}")

# Método 2: Con Counter (muy elegante)
merged_counter = Counter(dict1) + Counter(dict2)
print(f"Merged (Counter): {dict(merged_counter)}")


# EJERCICIO 1.2.5: Two Sum - Encontrar dos números que sumen target
# Input: nums=[2, 7, 11, 15], target=9
# Output: (0, 1) porque nums[0] + nums[1] = 9

nums = [2, 7, 11, 15]
target = 9
# TU CÓDIGO:


# SOLUCIÓN:
def two_sum(nums, target):
    seen = {}  # valor -> índice
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return None

print(f"Two Sum: {two_sum(nums, target)}")
# TIP: Este es O(n) con diccionario. Sin dict sería O(n²) con dos loops.


# -----------------------------------------------------------------------------
# 1.3 SETS - Operaciones de conjuntos
# -----------------------------------------------------------------------------

print("\n--- 1.3 SETS ---")

# EJERCICIO 1.3.1: Encontrar elementos comunes entre dos listas
# Input: [1, 2, 3, 4], [3, 4, 5, 6]
# Output: {3, 4}

lista1 = [1, 2, 3, 4]
lista2 = [3, 4, 5, 6]
# TU CÓDIGO:


# SOLUCIÓN:
comunes = set(lista1) & set(lista2)
print(f"Elementos comunes: {comunes}")

# También: set(lista1).intersection(lista2)


# EJERCICIO 1.3.2: Encontrar elementos únicos en la primera lista
# Input: [1, 2, 3, 4], [3, 4, 5, 6]
# Output: {1, 2}

# TU CÓDIGO:


# SOLUCIÓN:
solo_en_primera = set(lista1) - set(lista2)
print(f"Solo en primera lista: {solo_en_primera}")


# EJERCICIO 1.3.3: Validar que todas las columnas requeridas existen
# (CASO REAL DE DATA ENGINEER)
# Input: required=['id', 'name', 'date'], actual=['id', 'name', 'value']
# Output: Missing: {'date'}, Extra: {'value'}

required_cols = {'id', 'name', 'date'}
actual_cols = {'id', 'name', 'value'}
# TU CÓDIGO:


# SOLUCIÓN:
missing = required_cols - actual_cols
extra = actual_cols - required_cols
print(f"Columnas faltantes: {missing}")
print(f"Columnas extra: {extra}")
# TIP: Este patrón se usa MUCHO en validación de schemas


# EJERCICIO 1.3.4: Encontrar usuarios que compraron ambos productos
# (SIMULA UN JOIN)

compraron_A = {'user1', 'user2', 'user3', 'user5'}
compraron_B = {'user2', 'user3', 'user4', 'user6'}
# TU CÓDIGO:


# SOLUCIÓN:
compraron_ambos = compraron_A & compraron_B
compraron_solo_A = compraron_A - compraron_B
compraron_alguno = compraron_A | compraron_B

print(f"Compraron ambos: {compraron_ambos}")
print(f"Compraron solo A: {compraron_solo_A}")
print(f"Compraron al menos uno: {compraron_alguno}")


# =============================================================================
# SECTION 2: Conceptos "Pythonic"
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Código Pythonic")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 LIST COMPREHENSIONS
# -----------------------------------------------------------------------------

print("\n--- 2.1 LIST COMPREHENSIONS ---")

# EJERCICIO 2.1.1: Convertir a cuadrados solo los números pares
# Input: [1, 2, 3, 4, 5, 6]
# Output: [4, 16, 36]

numeros = [1, 2, 3, 4, 5, 6]
# TU CÓDIGO:


# SOLUCIÓN:
cuadrados_pares = [x**2 for x in numeros if x % 2 == 0]
print(f"Cuadrados de pares: {cuadrados_pares}")


# EJERCICIO 2.1.2: Limpiar y filtrar precios
# Input: ['$100', '$20', 'N/A', '$5.50', '', None, '$30']
# Output: [100.0, 20.0, 5.5, 30.0]

precios_sucios = ['$100', '$20', 'N/A', '$5.50', '', None, '$30']
# TU CÓDIGO:


# SOLUCIÓN:
def limpiar_precio(p):
    if not p or p == 'N/A':
        return None
    try:
        return float(p.replace('$', '').replace(',', ''))
    except ValueError:
        return None

precios_limpios = [limpiar_precio(p) for p in precios_sucios]
precios_validos = [p for p in precios_limpios if p is not None]
print(f"Precios limpios: {precios_validos}")
print(f"Suma total: ${sum(precios_validos)}")


# EJERCICIO 2.1.3: Dict comprehension - Crear lookup table
# Input: [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
# Output: {1: 'Alice', 2: 'Bob'}

usuarios = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}, {'id': 3, 'name': 'Charlie'}]
# TU CÓDIGO:


# SOLUCIÓN:
lookup = {u['id']: u['name'] for u in usuarios}
print(f"Lookup table: {lookup}")
# TIP: Esto es muy útil para "joins" manuales en pipelines


# EJERCICIO 2.1.4: Flatten lista anidada
# Input: [[1, 2], [3, 4], [5]]
# Output: [1, 2, 3, 4, 5]

anidada = [[1, 2], [3, 4], [5]]
# TU CÓDIGO:


# SOLUCIÓN:
plana = [item for sublista in anidada for item in sublista]
print(f"Lista plana: {plana}")
# TIP: El orden se lee como bucles anidados de izquierda a derecha


# -----------------------------------------------------------------------------
# 2.2 GENERADORES (yield) - CRÍTICO PARA DATA ENGINEER
# -----------------------------------------------------------------------------

print("\n--- 2.2 GENERADORES ---")

# EJERCICIO 2.2.1: Leer archivo grande línea por línea
# (Simularemos con una lista, pero en producción sería un archivo)

def procesar_archivo_mal(lineas):
    """MAL: Carga todo en memoria"""
    todas = [linea.upper() for linea in lineas]
    return todas

def procesar_archivo_bien(lineas):
    """BIEN: Genera una línea a la vez"""
    for linea in lineas:
        yield linea.upper()

# Simulación
lineas_ejemplo = ['linea1', 'linea2', 'linea3']

# Uso del generador
for linea in procesar_archivo_bien(lineas_ejemplo):
    print(f"Procesando: {linea}")

# TIP: Con yield, puedes procesar archivos de 100GB en una laptop de 16GB RAM


# EJERCICIO 2.2.2: Crear un generador de batches
# Input: [1,2,3,4,5,6,7,8,9,10], batch_size=3
# Output: [1,2,3], [4,5,6], [7,8,9], [10]

def batch_generator(items, batch_size):
    # TU CÓDIGO:
    pass


# SOLUCIÓN:
def batch_generator(items, batch_size):
    """Genera batches de tamaño fijo"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

datos = list(range(1, 11))
print("\nBatches de 3:")
for batch in batch_generator(datos, 3):
    print(f"  Batch: {batch}")
# TIP: Útil para cargar datos en chunks a una base de datos


# EJERCICIO 2.2.3: Generador infinito con filtro
# Genera números naturales que son múltiplos de 3 o 5

def multiplos_3_o_5():
    # TU CÓDIGO:
    pass


# SOLUCIÓN:
def multiplos_3_o_5(limite=None):
    """Genera múltiplos de 3 o 5"""
    n = 1
    while limite is None or n <= limite:
        if n % 3 == 0 or n % 5 == 0:
            yield n
        n += 1

print("\nPrimeros 10 múltiplos de 3 o 5:")
gen = multiplos_3_o_5()
for _ in range(10):
    print(next(gen), end=' ')
print()


# -----------------------------------------------------------------------------
# 2.3 MANEJO DE ERRORES
# -----------------------------------------------------------------------------

print("\n--- 2.3 MANEJO DE ERRORES ---")

# EJERCICIO 2.3.1: Parser de datos robusto
# Los datos vienen sucios. No debe romperse el pipeline.

datos_sucios = [
    {'fecha': '2024-01-15', 'monto': '100.50'},
    {'fecha': '2024-01-16', 'monto': 'N/A'},
    {'fecha': 'fecha_invalida', 'monto': '200'},
    {'fecha': '2024-01-17'},  # Falta monto
    None,  # Registro nulo
]

def parsear_registro_seguro(registro):
    # TU CÓDIGO:
    pass


# SOLUCIÓN:
from datetime import datetime

def parsear_registro_seguro(registro):
    """Parsea un registro manejando todos los errores posibles"""
    if registro is None:
        return None

    resultado = {'fecha': None, 'monto': None, 'errores': []}

    # Parsear fecha
    try:
        fecha_str = registro.get('fecha')
        if fecha_str:
            resultado['fecha'] = datetime.strptime(fecha_str, '%Y-%m-%d')
    except (ValueError, AttributeError) as e:
        resultado['errores'].append(f"Error en fecha: {e}")

    # Parsear monto
    try:
        monto_str = registro.get('monto')
        if monto_str and monto_str != 'N/A':
            resultado['monto'] = float(monto_str)
    except (ValueError, AttributeError) as e:
        resultado['errores'].append(f"Error en monto: {e}")

    return resultado

print("\nRegistros parseados:")
for dato in datos_sucios:
    resultado = parsear_registro_seguro(dato)
    print(f"  Input: {dato}")
    print(f"  Output: {resultado}\n")


# =============================================================================
# SECTION 3: Manipulación de Strings (80% del trabajo de un DE)
# =============================================================================

print("=" * 70)
print("SECTION 3: Manipulación de Strings")
print("=" * 70)

# EJERCICIO 3.1: Limpiar y normalizar nombres
# Input: "  JUAN  pérez García  "
# Output: "Juan Perez Garcia"

nombre_sucio = "  JUAN  pérez García  "
# TU CÓDIGO:


# SOLUCIÓN:
import unicodedata

def normalizar_nombre(nombre):
    # Quitar espacios al inicio/final
    nombre = nombre.strip()
    # Normalizar múltiples espacios a uno
    nombre = ' '.join(nombre.split())
    # Title case
    nombre = nombre.title()
    # Quitar acentos (opcional, depende del caso)
    nombre = unicodedata.normalize('NFD', nombre)
    nombre = nombre.encode('ascii', 'ignore').decode('utf-8')
    return nombre

print(f"Nombre normalizado: '{normalizar_nombre(nombre_sucio)}'")


# EJERCICIO 3.2: Extraer información de un string estructurado
# Input: "ERROR|2024-01-15 10:30:45|user_123|Login failed"
# Output: {'level': 'ERROR', 'timestamp': '2024-01-15 10:30:45',
#          'user': 'user_123', 'message': 'Login failed'}

log_line = "ERROR|2024-01-15 10:30:45|user_123|Login failed"
# TU CÓDIGO:


# SOLUCIÓN:
def parsear_log(linea):
    partes = linea.split('|')
    if len(partes) >= 4:
        return {
            'level': partes[0],
            'timestamp': partes[1],
            'user': partes[2],
            'message': '|'.join(partes[3:])  # Por si el mensaje tiene '|'
        }
    return None

print(f"Log parseado: {parsear_log(log_line)}")


# EJERCICIO 3.3: Validar y extraer emails de texto
# Input: "Contactar a juan@empresa.com o soporte@empresa.com"
# Output: ['juan@empresa.com', 'soporte@empresa.com']

texto = "Contactar a juan@empresa.com o soporte@empresa.com para más info"
# TU CÓDIGO:


# SOLUCIÓN:
import re

def extraer_emails(texto):
    patron = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(patron, texto)

print(f"Emails encontrados: {extraer_emails(texto)}")


# EJERCICIO 3.4: Convertir snake_case a camelCase
# Input: "user_first_name"
# Output: "userFirstName"

snake = "user_first_name"
# TU CÓDIGO:


# SOLUCIÓN:
def snake_to_camel(s):
    components = s.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

print(f"Camel case: {snake_to_camel(snake)}")


# EJERCICIO 3.5: Formatear datos para SQL
# Input: {'name': "O'Brien", 'age': 30, 'active': True}
# Output: "INSERT INTO users (name, age, active) VALUES ('O''Brien', 30, TRUE)"

datos_sql = {'name': "O'Brien", 'age': 30, 'active': True}
# TU CÓDIGO:


# SOLUCIÓN:
def format_sql_value(val):
    if val is None:
        return 'NULL'
    elif isinstance(val, bool):
        return 'TRUE' if val else 'FALSE'
    elif isinstance(val, (int, float)):
        return str(val)
    else:
        # Escapar comillas simples
        escaped = str(val).replace("'", "''")
        return f"'{escaped}'"

def generar_insert(tabla, datos):
    columnas = ', '.join(datos.keys())
    valores = ', '.join(format_sql_value(v) for v in datos.values())
    return f"INSERT INTO {tabla} ({columnas}) VALUES ({valores})"

print(f"SQL: {generar_insert('users', datos_sql)}")
# TIP: En producción usarías parametrized queries para evitar SQL injection


# =============================================================================
# SECTION 4: Ejercicios de Entrevista Completos
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Ejercicios Típicos de Entrevista")
print("=" * 70)

# -----------------------------------------------------------------------------
# EJERCICIO 4.1: FizzBuzz (Filtro básico)
# -----------------------------------------------------------------------------
print("\n--- 4.1 FizzBuzz ---")

def fizzbuzz(n):
    # TU CÓDIGO:
    pass


# SOLUCIÓN:
def fizzbuzz(n):
    resultado = []
    for i in range(1, n + 1):
        if i % 15 == 0:  # Múltiplo de 3 Y 5
            resultado.append('FizzBuzz')
        elif i % 3 == 0:
            resultado.append('Fizz')
        elif i % 5 == 0:
            resultado.append('Buzz')
        else:
            resultado.append(str(i))
    return resultado

print(f"FizzBuzz(15): {fizzbuzz(15)}")


# -----------------------------------------------------------------------------
# EJERCICIO 4.2: Anagramas
# -----------------------------------------------------------------------------
print("\n--- 4.2 Verificar Anagramas ---")

# Input: "listen", "silent"
# Output: True

def son_anagramas(s1, s2):
    # TU CÓDIGO:
    pass


# SOLUCIÓN:
def son_anagramas(s1, s2):
    # Método 1: Ordenar
    # return sorted(s1.lower()) == sorted(s2.lower())

    # Método 2: Counter (más eficiente para strings largos)
    return Counter(s1.lower()) == Counter(s2.lower())

print(f"'listen' y 'silent' son anagramas: {son_anagramas('listen', 'silent')}")
print(f"'hello' y 'world' son anagramas: {son_anagramas('hello', 'world')}")


# -----------------------------------------------------------------------------
# EJERCICIO 4.3: Validar Paréntesis Balanceados
# -----------------------------------------------------------------------------
print("\n--- 4.3 Paréntesis Balanceados ---")

# Input: "({[]})"
# Output: True

def parentesis_validos(s):
    # TU CÓDIGO:
    pass


# SOLUCIÓN:
def parentesis_validos(s):
    stack = []
    pares = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack[-1] != pares[char]:
                return False
            stack.pop()

    return len(stack) == 0

print(f"'({[]})' es válido: {parentesis_validos('({[]})')}")
print(f"'({[}])' es válido: {parentesis_validos('({[}])')}")


# -----------------------------------------------------------------------------
# EJERCICIO 4.4: Encontrar el primer carácter único
# -----------------------------------------------------------------------------
print("\n--- 4.4 Primer Carácter Único ---")

# Input: "aabbcdd"
# Output: 'c'

def primer_unico(s):
    # TU CÓDIGO:
    pass


# SOLUCIÓN:
def primer_unico(s):
    conteo = Counter(s)
    for char in s:
        if conteo[char] == 1:
            return char
    return None

print(f"Primer único en 'aabbcdd': {primer_unico('aabbcdd')}")
print(f"Primer único en 'aabbcc': {primer_unico('aabbcc')}")


# -----------------------------------------------------------------------------
# EJERCICIO 4.5: Comprimir string (Run-Length Encoding)
# -----------------------------------------------------------------------------
print("\n--- 4.5 Compresión de String ---")

# Input: "aabbbcccc"
# Output: "a2b3c4"

def comprimir(s):
    # TU CÓDIGO:
    pass


# SOLUCIÓN:
def comprimir(s):
    if not s:
        return ""

    resultado = []
    char_actual = s[0]
    conteo = 1

    for char in s[1:]:
        if char == char_actual:
            conteo += 1
        else:
            resultado.append(f"{char_actual}{conteo}")
            char_actual = char
            conteo = 1

    resultado.append(f"{char_actual}{conteo}")
    return ''.join(resultado)

print(f"Comprimido 'aabbbcccc': {comprimir('aabbbcccc')}")


# =============================================================================
# SECTION 5: Ejercicios de Data Engineering Reales
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Casos Reales de Data Engineering")
print("=" * 70)

# -----------------------------------------------------------------------------
# EJERCICIO 5.1: JOIN manual entre dos datasets
# -----------------------------------------------------------------------------
print("\n--- 5.1 JOIN Manual ---")

usuarios = [
    {'user_id': 1, 'nombre': 'Alice'},
    {'user_id': 2, 'nombre': 'Bob'},
    {'user_id': 3, 'nombre': 'Charlie'},
]

compras = [
    {'user_id': 1, 'producto': 'Laptop', 'monto': 1000},
    {'user_id': 1, 'producto': 'Mouse', 'monto': 50},
    {'user_id': 2, 'producto': 'Teclado', 'monto': 100},
    {'user_id': 4, 'producto': 'Monitor', 'monto': 300},  # Usuario que no existe
]

# INNER JOIN
# TU CÓDIGO:


# SOLUCIÓN:
# Crear lookup table primero (O(1) búsqueda)
usuarios_lookup = {u['user_id']: u['nombre'] for u in usuarios}

# INNER JOIN
inner_join = []
for compra in compras:
    if compra['user_id'] in usuarios_lookup:
        inner_join.append({
            **compra,
            'nombre': usuarios_lookup[compra['user_id']]
        })

print("INNER JOIN:")
for row in inner_join:
    print(f"  {row}")

# LEFT JOIN (todas las compras, con nombre si existe)
left_join = []
for compra in compras:
    left_join.append({
        **compra,
        'nombre': usuarios_lookup.get(compra['user_id'], None)
    })

print("\nLEFT JOIN:")
for row in left_join:
    print(f"  {row}")


# -----------------------------------------------------------------------------
# EJERCICIO 5.2: Agregaciones tipo GROUP BY
# -----------------------------------------------------------------------------
print("\n--- 5.2 GROUP BY Manual ---")

ventas = [
    {'region': 'Norte', 'producto': 'A', 'cantidad': 100},
    {'region': 'Norte', 'producto': 'B', 'cantidad': 150},
    {'region': 'Sur', 'producto': 'A', 'cantidad': 200},
    {'region': 'Sur', 'producto': 'B', 'cantidad': 100},
    {'region': 'Norte', 'producto': 'A', 'cantidad': 50},
]

# Calcular: suma de cantidad por región
# TU CÓDIGO:


# SOLUCIÓN:
from collections import defaultdict

# GROUP BY región
por_region = defaultdict(int)
for venta in ventas:
    por_region[venta['region']] += venta['cantidad']

print("Ventas por región:")
for region, total in por_region.items():
    print(f"  {region}: {total}")

# GROUP BY región Y producto
por_region_producto = defaultdict(int)
for venta in ventas:
    key = (venta['region'], venta['producto'])
    por_region_producto[key] += venta['cantidad']

print("\nVentas por región y producto:")
for (region, producto), total in sorted(por_region_producto.items()):
    print(f"  {region} - {producto}: {total}")


# -----------------------------------------------------------------------------
# EJERCICIO 5.3: Deduplicación con regla de negocio
# -----------------------------------------------------------------------------
print("\n--- 5.3 Deduplicación Inteligente ---")

# Quedarse con el registro más reciente por user_id
registros = [
    {'user_id': 1, 'email': 'old@mail.com', 'updated': '2024-01-01'},
    {'user_id': 1, 'email': 'new@mail.com', 'updated': '2024-01-15'},
    {'user_id': 2, 'email': 'user2@mail.com', 'updated': '2024-01-10'},
    {'user_id': 1, 'email': 'newest@mail.com', 'updated': '2024-01-20'},
]

# TU CÓDIGO:


# SOLUCIÓN:
def deduplicar_por_mas_reciente(registros, key_field, date_field):
    """Mantiene el registro más reciente para cada key"""
    mas_recientes = {}

    for reg in registros:
        key = reg[key_field]
        if key not in mas_recientes or reg[date_field] > mas_recientes[key][date_field]:
            mas_recientes[key] = reg

    return list(mas_recientes.values())

deduplicados = deduplicar_por_mas_reciente(registros, 'user_id', 'updated')
print("Registros deduplicados (más reciente):")
for reg in deduplicados:
    print(f"  {reg}")


# -----------------------------------------------------------------------------
# EJERCICIO 5.4: Detección de anomalías simple
# -----------------------------------------------------------------------------
print("\n--- 5.4 Detección de Anomalías ---")

# Encontrar transacciones fuera de 2 desviaciones estándar
transacciones = [100, 105, 98, 102, 500, 99, 101, 97, 103, 1000]

# TU CÓDIGO:


# SOLUCIÓN:
import statistics

def detectar_anomalias(datos, num_std=2):
    """Detecta valores fuera de num_std desviaciones estándar"""
    media = statistics.mean(datos)
    std = statistics.stdev(datos)

    limite_inferior = media - (num_std * std)
    limite_superior = media + (num_std * std)

    anomalias = [(i, v) for i, v in enumerate(datos)
                 if v < limite_inferior or v > limite_superior]

    return {
        'media': round(media, 2),
        'std': round(std, 2),
        'limites': (round(limite_inferior, 2), round(limite_superior, 2)),
        'anomalias': anomalias
    }

resultado = detectar_anomalias(transacciones)
print(f"Análisis de anomalías:")
print(f"  Media: {resultado['media']}")
print(f"  Std: {resultado['std']}")
print(f"  Límites: {resultado['limites']}")
print(f"  Anomalías (índice, valor): {resultado['anomalias']}")


# -----------------------------------------------------------------------------
# EJERCICIO 5.5: Pipeline de transformación completo
# -----------------------------------------------------------------------------
print("\n--- 5.5 Mini Pipeline ETL ---")

# Datos crudos (simula lectura de CSV/API)
datos_crudos = [
    "id,nombre,email,monto",
    "1,  JUAN PÉREZ ,juan@email.com,$1,500.00",
    "2,María García,maria@email.com,$750.50",
    "3,Carlos López,,N/A",  # Email vacío, monto inválido
    "4,Ana Martín,ana@email.com,$2,000.00",
    "",  # Línea vacía
]

def pipeline_etl(lineas):
    # TU CÓDIGO:
    pass


# SOLUCIÓN:
def pipeline_etl(lineas):
    """Pipeline completo: Extract, Transform, Load"""

    # EXTRACT: Parsear CSV
    registros = []
    header = None

    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue

        partes = linea.split(',')
        if header is None:
            header = partes
            continue

        registro = dict(zip(header, partes))
        registros.append(registro)

    print(f"Extraídos: {len(registros)} registros")

    # TRANSFORM
    transformados = []
    errores = []

    for reg in registros:
        try:
            # Limpiar nombre
            nombre = ' '.join(reg.get('nombre', '').strip().split())
            nombre = nombre.title()

            # Validar email
            email = reg.get('email', '').strip()
            if not email or '@' not in email:
                email = None

            # Parsear monto
            monto_str = reg.get('monto', '').replace('$', '').replace(',', '')
            try:
                monto = float(monto_str) if monto_str and monto_str != 'N/A' else None
            except ValueError:
                monto = None

            transformados.append({
                'id': int(reg['id']),
                'nombre': nombre,
                'email': email,
                'monto': monto,
                'es_valido': email is not None and monto is not None
            })

        except Exception as e:
            errores.append({'registro': reg, 'error': str(e)})

    print(f"Transformados: {len(transformados)} registros")
    print(f"Errores: {len(errores)}")

    # LOAD: Separar válidos e inválidos
    validos = [r for r in transformados if r['es_valido']]
    invalidos = [r for r in transformados if not r['es_valido']]

    return {
        'validos': validos,
        'invalidos': invalidos,
        'errores': errores
    }

resultado = pipeline_etl(datos_crudos)
print("\nRegistros válidos:")
for r in resultado['validos']:
    print(f"  {r}")
print("\nRegistros inválidos:")
for r in resultado['invalidos']:
    print(f"  {r}")


# =============================================================================
# SECTION 6: Tips de Performance
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Tips de Performance (Para impresionar)")
print("=" * 70)

print("""
COMPLEJIDAD QUE DEBES CONOCER:
------------------------------

Estructura          | Búsqueda  | Inserción | Eliminación
--------------------|-----------|-----------|-------------
Lista               | O(n)      | O(1)*     | O(n)
Diccionario/Set     | O(1)      | O(1)      | O(1)
Lista ordenada      | O(log n)  | O(n)      | O(n)

* O(1) amortizado para append al final

REGLAS DE ORO:
--------------
1. Si buscas mucho → Usa diccionario/set
2. Si necesitas orden → Lista
3. Si necesitas únicos → Set
4. Si necesitas contar → Counter
5. Si agrupas → defaultdict(list)

ANTI-PATRONES A EVITAR:
-----------------------
❌ for x in lista: if x in otra_lista  → O(n²)
✅ for x in lista: if x in otro_set    → O(n)

❌ concatenar strings en loop con +
✅ usar ''.join(lista_de_strings)

❌ leer archivo completo con read()
✅ iterar línea por línea o usar generador

❌ modificar lista mientras iteras
✅ crear lista nueva o iterar sobre copia
""")


# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 70)
print("RESUMEN: ¿Qué evalúan en una entrevista de DE?")
print("=" * 70)

print("""
1. LÓGICA
   ¿Puedes resolver el problema sin googlear?

2. ESTRUCTURAS DE DATOS
   ¿Usaste diccionario para O(1) o dos for loops para O(n²)?

3. CÓDIGO LIMPIO
   ¿Es legible? ¿Usas nombres descriptivos?

4. MANEJO DE ERRORES
   ¿Tu código maneja None, strings vacíos, datos sucios?

5. COMUNICACIÓN
   ¿Explicas tu pensamiento mientras codificas?

TIPS FINALES:
-------------
- Pregunta aclaraciones antes de codificar
- Menciona la complejidad Big O
- Escribe tests simples
- Menciona qué harías diferente en producción
- Si no sabes algo, di "No lo sé, pero investigaría..."

¡Practica estos ejercicios hasta que los hagas sin pensar!
""")
