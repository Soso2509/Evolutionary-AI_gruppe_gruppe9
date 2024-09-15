# Evolutionary-AI Portfolio

## Forutsetninger

Før du starter, sørg for at du har følgende installert:

- [Python 3.8+](https://www.python.org/downloads/)
- [Visual Studio Code](https://code.visualstudio.com/)
- Git for versjonskontroll

## Oppsett av utviklingsmiljø første gang du starter prosjektet

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
Når utviklingsmiljøet er satt opp, kan koden kjøres:

[Problem 2](/prob2/oppg2_CodeCommands.md)


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
### Trinn 9: Oppdater din lokale `main`-gren og deretter din egen gren før du begynner å programmere

Følg disse trinnene

1. **Sjekk ut main-grenen og trekk ned de nyeste endringene:**

   ```bash
   git checkout main
   git pull origin main
   ```

2. **Sjekk ut din egen gren:**

   ```bash
   git checkout <din-gren>
   ```

3. **Oppdater din gren med de nyeste endringene fra main:**

   ```bash
   git merge main
   ```

4. **Start å kode på din egen gren!**

### Trinn 10: Oppdatere `main`-grenen med endringer fra en annen gren

Følg disse trinnene for å slå sammen endringene fra en vilkårlig gren inn i `main`-grenen:

1. **Sjekk ut den grenen du vil slå sammen, og push eventuelle endringer til det eksterne repoet:**

   ```bash
   git checkout <din-gren>
   git add .
   git commit -m "Din commit-melding her"
   git push origin <din-gren>
   ```

2. **Bytt til `main`-grenen:**

   ```bash
   git checkout main
   ```

3. **Trekk ned de nyeste endringene fra det eksterne repoet på `main`:**

   ```bash
   git pull origin main
   ```

4. **Slå sammen endringene fra din gren inn i `main`:**

   ```bash
   git merge <din-gren>
   ```

   Hvis det oppstår sammenslåingskonflikter, må du løse dem manuelt, legge til endringene, og deretter fortsette sammenslåingen.

5. **Push de oppdaterte endringene til `main`-grenen i det eksterne repoet:**

   ```bash
   git push origin main
   ```


