# Evolutionary-AI Portfolio
## Group 9

## Forutsetninger

Før du starter, sørg for at du har følgende installert:

- [Python 3.8+](https://www.python.org/downloads/)
- [Visual Studio Code](https://code.visualstudio.com/)
- Git for versjonskontroll (anbefalt, men ikke påkrevd)

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

### Trinn 4: Installer nødvendige pakker

Etter at det virtuelle miljøet er aktivert, installer de nødvendige pakkene fra `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Trinn 5: Generere eller oppdatere `requirements.txt`

For å sikre at alle bruker samme avhengigheter, kan du generere eller oppdatere `requirements.txt` før du pusher koden.
Husk å gå til rot-mappen før du kjører disse kommandoene. Det er det "requirements.txt" filen er lagret og bør lagres. 

1. Installer `pipreqs` for å generere filen automatisk:
   ```bash
   pip install pipreqs
   ```

2. Generer eller oppdater `requirements.txt` med pakker som kun brukes i nåværende kodefiler:
   ```bash
   pipreqs . --force
   ```

2. Generer eller oppdater `requirements.txt` med pakker som er innstallert i nåværende virtuelle miljø:
   ```bash
   pip freeze > requirements.txt
   ```

### Trinn 6: Kjøre analysen

Når utviklingsmiljøet er satt opp, kan du kjøre analysen ved å kjøre følgende kommando:

```bash
python analysis.py
```

(Erstatt `analysis.py` med navnet på din analyseskript hvis det er annerledes.)

### Trinn 7: Oppdatere avhengigheter etter git pull

Når du har hentet siste endringer fra GitHub og `requirements.txt` har blitt oppdatert, må du sørge for at dine lokale avhengigheter er oppdatert.

1. **Aktiver det virtuelle miljøet**:

   - **På Windows**:
     ```bash
     .\env\Scripts\activate
     ```

   - **På macOS/Linux**:
     ```bash
     source env/bin/activate
     ```

2. **Installer nye avhengigheter** etter å ha kjørt `git pull`:
   ```bash
   pip install -r requirements.txt
   ```

### Trinn 8: Deaktivere det virtuelle miljøet

Når du er ferdig med å jobbe, kan du deaktivere det virtuelle miljøet ved å skrive:

```bash
deactivate
```

