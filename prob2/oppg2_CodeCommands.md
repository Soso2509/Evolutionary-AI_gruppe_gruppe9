Sørg for at du er i rot-mappen\
*\Evolutionary-AI_gruppe_gruppe9*

1. Beregne månedlig avkastning og skrive ut til csv-fil "mnd_avk_aksjer.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.1beregn_mnd_avk.py
```

2. Beregne kovariansmatrise og skrive ut til csv-fil "mnd_avk_cov_matrix.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.2beregn_kovarians_matrix.py
```

3. Kjør bep og skrive ut til csv-fil "3.3bep.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.3bep.py
```

4. Kjør aep og skrive ut til csv-fil "3.4aep.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.4aep.py
```
5. Kjør bes og skrive ut til csv-fil "3.5bes.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.5bes.py
```
6. Kjør aep og skrive ut til csv-fil "3.6aes.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.6aes.py
```
7. Kjør aep og skrive ut til csv-fil "3.7m+l.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.7m+l.py
```

Generer html-fil med tabeller av csv-filene:
```bash
python prob2/4.prob2_visual/4.1visual.py
```
Start server og vis den genererte html filen
Naviger til riktig mappe:
```bash
cd prob2/4.prob2_visual/
```

Start server:
```bash
python -m http.server 8000
```

Åpne nettleser og gå til følgende lenke for å åpne den generert html-filen:
```bash
http://localhost:8000/aksje_avkastning_sorterbar.html
```

Avslutt server:
```bash
Ctrl + C
```