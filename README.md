# TSP_Classical_Solutions

Creación de algoritmos clásicos para la resolución del TSP 


Pasos para la preparación del entorno: 

1. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
2. ptyhon -m venv tsp-venv 
3. .\tsp-venv\Scripts\Activate 
4. pip install -r requirements.txt


**Descarga del TSP rl11849**

>
    url = 'http://softlib.rice.edu/pub/tsplib/tsp/rl11849.tsp.gz'
    urllib.request.urlretrieve(url, 'rl11849.tsp.gz')

    with gzip.open('rl11849.tsp.gz', 'rt', encoding='utf-8') as src, \
        open('rl11849.tsp',    'w', encoding='utf-8') as dst:
        dst.write(src.read())
>