# CSV Writers per VISTA Pipeline

Questo modulo fornisce le utilità necessarie per inizializzare i file CSV (con i relativi e corretti header) e per fare l'append di righe nei CSV prodotti dalla pipeline (MOT predictions e Tracks captions).

## Struttura
- `write_csv_skeletons.py`: Lo script principale contenente la funzione `create_csv_skeletons` per creare le tabelle vuote e la classe `CSVWriter` per inserire successivamente le predizioni in formato riga.

## Uso Rapido (Per integrazione con la Pipeline)

### 1. Inizializzare i CSV

Prima di eseguire la pipeline sui frame di un video, inizializza le directory e i relativi file, fornendo una directory di output:

```python
from scripts.write_csv_skeletons import create_csv_skeletons

# Crea i file CSV sotto out/run_001
create_csv_skeletons('out/run_001')