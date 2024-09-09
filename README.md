# Evolutionary-AI Portfolio

## Forutsetninger

Før du starter, sørg for at du har følgende installert:

- [Python 3.8+](https://www.python.org/downloads/)
- [Visual Studio Code](https://code.visualstudio.com/)
- Git for versjonskontroll

## Oppsett av utviklingsmiljø

For å sikre at alle på teamet bruker de samme avhengighetene og versjonene, følg disse trinnene for å sette opp utviklingsmiljøet:

### Trinn 1: Klon prosjektet

Hver teammedlem må klone prosjektet til sin lokale maskin:
```bash
git clone https://github.com/Soso2509/Evolutionary-AI_gruppe_gruppe9.git
cd Evolutionary-AI_gruppe_gruppe9
```

### Trinn 2: Opprett et virtuelt miljø

For å isolere prosjektets avhengigheter, opprett et virtuelt miljø.

#### På Windows:
```bash
python -m venv env
```

#### På macOS/Linux:
```bash
python3 -m venv env
```

### Trinn 3: Aktiver det virtuelle miljøet

#### På Windows:
```bash
.\env\Scripts\activate
```

#### På macOS/Linux:
```bash
source env/bin/activate
```

### Trinn 4: Innstallere avhengigheter fra `requirements.txt` etter git pull

Etter at det virtuelle miljøet er aktivert, installer de nødvendige pakkene fra `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Trinn 6: Kjøre kode
Når utviklingsmiljøet er satt opp, kan du kjøre kode ved å kjøre følgende kommando:

Problem 2: 
1. Naviger til riktig mappe der kodefilene ligger: 
```bash
cd prob2/2.prob2_kode
```

2. Beregne månedlig avkastning og skrive ut til csv-fil "mnd_avk_aksjer.csv" (overskriver hvis filen finnes i mappen fra før): 
```bash
python prob2/2.prob2_kode/2.1beregn_mnd_avk.py
```

3. Beregne kovariansmatrise og skrive ut til csv-fil "mnd_avk_cov_matrix.csv" (overskriver hvis filen finnes i mappen fra før): 
```bash
python prob2/2.prob2_kode/2.2beregn_kovarians_matrix.py
```

4. Kjør ep og skrive ut til csv-fil "3.3ep.csv" (overskriver hvis filen finnes i mappen fra før): 
```bash
python prob2/2.prob2_kode/2.3ep.py
```

5. Generer html-fil med tabeller av csv-filene: 
```bash
python prob2/4.prob2_visual/4.1visual.py
```

Naviger til riktig mappe: 
```bash
cd prob2/4.prob2_visual/output/
```

Start server: 
```bash
python -m http.server 8000                              
```

Avslutt server: 
```bash
Ctrl + C                              
```

### Trinn 5: Oppdatere avhengigheter `requirements.txt` før git push

For å sikre at alle bruker samme avhengigheter, kan du generere eller oppdatere `requirements.txt` før du pusher koden.
Husk å gå til rot-mappen før du kjører disse kommandoene. Det er der "requirements.txt" filen er lagret og bør lagres. 

Generer eller oppdater `requirements.txt` med pakker som er innstallert i nåværende virtuelle miljø:
```bash
pip freeze > requirements.txt
```

### Trinn 8: Deaktivere det virtuelle miljøet

Når du er ferdig med å jobbe, kan du deaktivere det virtuelle miljøet ved å skrive:
```bash
deactivate
```

