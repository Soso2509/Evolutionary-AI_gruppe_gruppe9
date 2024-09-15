# Evolutionary-AI Portfolio

## Forutsetninger

Før du starter, sørg for at du har følgende installert:

- [Python 3.8+](https://www.python.org/downloads/)
- [Visual Studio Code](https://code.visualstudio.com/)
- Git for versjonskontroll

## Oversikt
- [Oppsett av utviklingsmiljø](#oppsett-av-utviklingsmiljø-første-gang-du-starter-prosjektet)
- [Aktivering av virtuelt miljø og installering av avhengigheter](#aktivering-av-det-virtuelle-miljøet-og-installering-av-avhengigheter)
- [Kjøring av kode](#kjøre-kode)
   - [Problem 2](#problem-2)
   - [Problem 3](#problem-3)
   - [Problem 4](#problem-4)
- [Oppdatering av avhengigheter](#oppdatere-avhengigheter-requirementstxt-før-git-push)
- [Deaktiverer det virtuelle miljøet](#deaktivere-det-virtuelle-miljøet)
- [Kommandoer relatert til git](#some-useful-git-commands)

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

## Aktivering av det virtuelle miljøet og installering av avhengigheter
#### På Windows:
```bash
.\env\Scripts\activate
```

#### På macOS/Linux:
```bash
source env/bin/activate
```

### Innstallere avhengigheter fra `requirements.txt` etter git pull
Etter at det virtuelle miljøet er aktivert, installer de nødvendige pakkene fra `requirements.txt`:
```bash
pip install -r requirements.txt
```
## Kjøre kode
Når utviklingsmiljøet er satt opp, kan koden kjøres:
- [Problem 2](#problem-2)
- [Problem 3](#problem-3)
- [Problem 4](#problem-4)

### Problem 2
#### Sørg for at du er i rot-mappen *\Evolutionary-AI_gruppe_gruppe9*
####  1. Beregne månedlig avkastning og skrive ut til csv-fil "mnd_avk_aksjer.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.1beregn_mnd_avk.py
```
#### 2. Beregne kovariansmatrise og skrive ut til csv-fil "mnd_avk_cov_matrix.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.2beregn_kovarians_matrix.py
```

#### 3. Kjør bep og skrive ut til csv-fil "3.3bep.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.3bep.py
```

#### 4. Kjør aep og skrive ut til csv-fil "3.4aep.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.4aep.py
```
#### 5. Kjør bes og skrive ut til csv-fil "3.5bes.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.5bes.py
```
#### 6. Kjør aep og skrive ut til csv-fil "3.6aes.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.6aes.py
```
#### 7. Kjør aep og skrive ut til csv-fil "3.7m+l.csv" (overskriver hvis filen finnes i mappen fra før):
```bash
python prob2/2.prob2_kode/2.7m+l.py
```

#### Generer html-fil med tabeller av csv-filene:
```bash
python prob2/4.prob2_visual/4.1visual.py
```
#### Start server og vis den genererte html filen
##### Naviger til riktig mappe:
```bash
cd prob2/4.prob2_visual/
```

##### Start server:
```bash
python -m http.server 8000
```

##### Åpne nettleser og gå til følgende lenke for å åpne den generert html-filen:
```bash
http://localhost:8000/aksje_avkastning_sorterbar.html
```

##### Avslutt server:
```bash
Ctrl + C
```

### Problem 3
### Problem 4

## Oppdatere avhengigheter `requirements.txt` før git push

For å sikre at alle bruker samme avhengigheter, kan du generere eller oppdatere `requirements.txt` før du pusher koden.
Husk å gå til rot-mappen før du kjører disse kommandoene. Det er der "requirements.txt" filen er lagret og bør lagres.

Generer eller oppdater `requirements.txt` med pakker som er innstallert i nåværende virtuelle miljø:
```bash
pip freeze > requirements.txt
```

## Deaktivere det virtuelle miljøet

Når du er ferdig med å jobbe, kan du deaktivere det virtuelle miljøet ved å skrive:
```bash
deactivate
```

## Some useful git-commands
### Process of making changes
1. `git status` _See if changes done locally_\
2. `git pull` _Retrieving changes from remote_\
3. _Make changes_\
4. `git add <file1> <file2> ...` _adds file to ones tracked by git_\
OR\
`git add .` _adds all un-tracked files_
5. `git commit -m "message"` _commit all files and adds a message in terminal (can also be added in editor)_\
6. `git push` _Pushes all changes committed til remote_

### Branches
`git branch <new-branch-name>` _Creates a new local branch_, Remember this branch is local and is not able to be seen or interacted with until you push it to remote \
`git branch` _Shows list of branches, the one with a * is the branch you are currently "standing" in_\
`git checkout <branch or commit>` _Can be uses to "switch" to previous commits or switch branches_\
`git switch <branch>` _To switch between branches_\
`git push -u origin <branch>` _Pushes branch to remote_\
`git merge <branch>` _Merges changes from another branch into the branch you're in_

### Oppdater din lokale `main`-gren og deretter din egen gren før du begynner å programmere

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

### Oppdatere `main`-grenen med endringer fra en annen gren

Følg disse trinnene for å slå sammen endringene fra en vilkårlig gren inn i `main`-grenen:

1. **Sørg for at du "står" i riktig branch og har commit'et og pushet endringer gjort til remote-repo**

   ```bash
   git branch  // liste over brancher, den med * ved er branchen du står i
   git checkout <din-gren>    // om nødvendig bytt branch
   git add .   // gjør ALLE endringer gjort klare til å committes
   git commit -m "Din commit-melding her"    // commiter alle endringene med message
   git push origin <din-gren> // dytter commits til den
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


