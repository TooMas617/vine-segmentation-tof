# Note sulla pipeline — Segmentazione semantica di componenti di vite da dati ToF

Documento di lavoro: registra **cosa fa** ogni fase della pipeline, **quali scelte
implementative** sono state prese e **perché**. Serve come base per il capitolo
metodologico della tesi e come riferimento per chi riprenderà il codice.

Ultimo aggiornamento: pipeline completa — nove moduli e quattro notebook,
verificati prima degli esperimenti (sezione 11). Il dataset è nella sua forma
**definitiva: 100 immagini**, partizione 70/15/15 con impronta `62de0ec4`.
Tutti i valori riportati in queste note si riferiscono a questa versione.

Gli articoli di riferimento citati nelle sezioni 2 e 7 sono depositati in
`reference/`, cartella esclusa dal versionamento. Sono tutti preprint arXiv e
nelle note vengono citati per identificativo, così restano reperibili anche
senza i file.

---

## 0. Architettura del progetto

### Struttura

```
VineProject_ToF/
├── src/                    moduli Python condivisi
│   ├── config.py           configurazione centralizzata
│   ├── data_preparation.py logica di preparazione dei dati
│   └── exploration.py      analisi esplorativa e statistiche del dataset
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   └── 02_EDA.ipynb
├── data/
│   ├── raw/                intensity, depth, confidence, annotations
│   ├── processed/          label maps, skeleton, maschere colorate,
│   │                       class_weights.json, channel_stats.json
│   ├── splits/             splits.json
│   └── augmented/          train / val / test materializzati
├── experiments/            una cartella per run
└── results/                experiments.csv (indice dei run)
```

### Scelte

**Separazione configurazione / implementazione.** Tutti i parametri modificabili
stanno in `config.py`; i moduli contengono solo logica. Il criterio adottato: va
in `config.py` ciò che può cambiare tra un esperimento e l'altro o che serve a più
moduli; resta nel modulo ciò che è vero per costruzione (es. i tipi di forma
LabelMe gestiti).

*Motivazione.* Il progetto originale era un unico notebook in cui parametri e
logica erano intrecciati. La separazione permette a quattro notebook distinti
(preparazione, EDA, training, valutazione) di condividere la stessa
configurazione senza duplicazioni, eliminando il rischio che due notebook usino
valori divergenti.

**Percorsi dedotti, non cablati.** `BASE_PATH` è derivato dalla posizione di
`config.py` sul filesystem (`src/config.py` → radice del progetto), con
possibilità di override tramite variabile d'ambiente. Il progetto può quindi
essere spostato o clonato senza modificare percorsi a mano.

**Nessun effetto collaterale all'import.** `config.py` definisce costanti e
funzioni ma non esegue nulla: il seeding e la creazione delle cartelle sono
chiamate esplicite del notebook. Un modulo importato non deve modificare lo stato
globale all'insaputa di chi lo usa.

**Una cartella per run.** La struttura originale pre-creava l'albero combinatorio
architettura × encoder × loss × optimizer (oltre 60 cartelle, in gran parte
vuote) e non distingueva due run della stessa configurazione con learning rate
diversi. È stata sostituita da un identificativo generato automaticamente
(`build_run_id`) nella forma
`<timestamp>_<architettura>-<encoder>_<loss>_<optimizer>[_suffisso]`, con una
cartella per run e un `experiments.csv` come indice ordinabile per metrica.

---

## 1. Generazione delle label maps

**Input:** file JSON prodotti da LabelMe.
**Output:** label map multiclasse, maschera binaria del cordone, versioni colorate.

### Classi

| Indice | Classe | Colore RGB |
|---|---|---|
| 0 | Background | (0, 0, 0) |
| 1 | Cane | (0, 128, 0) |
| 2 | Cordon | (128, 0, 0) |
| 3 | Node | (128, 128, 0) |
| 4 | Other | (0, 0, 128) |

### Scelte

**Alias delle etichette.** Le annotazioni usano etichette italiane
(`tralcio`, `cordone`, `nodo`, `altro`) mappate sulle classi canoniche inglesi.
La mappatura è molti-a-uno: `tralcio` e `tralcio_secco` confluiscono entrambe in
`Cane`, `cordone` e `curva` in `Cordon`.

*Motivazione.* Consente agli annotatori di lavorare in italiano e di distinguere
sottocategorie in fase di annotazione, senza vincolare la tassonomia del modello.
Le sottocategorie restano recuperabili dai JSON originali se in futuro servisse
una granularità maggiore.

**Due convenzioni di annotazione coesistono.** La verifica eseguita in fase di
EDA ha mostrato che il dataset non usa un vocabolario unico: 39 file su 100
usano etichette italiane, 61 usano etichette inglesi.

| Convenzione | File | Vocabolario |
|---|---|---|
| Italiana | 39 | `tralcio`, `tralcio_secco`, `cordone`, `curva`, `nodo`, `altro`, `ferro` |
| Inglese | 61 | `cane`, `cordon`, `node`, `other` |

Le due convenzioni **non hanno la stessa granularità**: quella inglese non
distingue il tralcio secco da quello vivo, non separa la curva dal resto del
cordone e non annota il filo di ferro. Poiché `tralcio` e `tralcio_secco`
confluiscono comunque in `Cane`, `curva` e `cordone` in `Cordon`, e `ferro` è
fra le etichette ignorate, la mappatura assorbe integralmente la differenza: le
label map prodotte dalle due convenzioni sono equivalenti.

**Le due convenzioni si compensano.** La convenzione italiana è più ricca sul
piano della tassonomia, ma quella inglese è **più accurata sui margini**: i
contorni delle forme seguono le strutture più fedelmente. Il confronto non è
quindi fra un sottoinsieme migliore e uno peggiore, ma fra due qualità diverse —
granularità delle etichette contro precisione geometrica.

*Conseguenza da tenere presente.* Poiché tutte le sottocategorie confluiscono
comunque nelle cinque classi canoniche, la granularità aggiuntiva delle 39
immagini italiane **non viene sfruttata** dal modello attuale, mentre la
precisione dei margini incide direttamente su ogni metrica basata su
sovrapposizione di area. Sui dati come sono oggi, quindi, la convenzione inglese
è quella che contribuisce di più — ed è quella con cui sono state annotate le
immagini aggiunte per portare il dataset a 100, il che porta la sua quota da 53
file su 92 a 61 su 100. La granularità italiana tornerebbe rilevante solo se il
problema cambiasse — per esempio distinguendo il tralcio secco da quello vivo,
distinzione rilevante in potatura — e in quel caso il sottoinsieme utilizzabile
sarebbe poco più di un terzo del dataset.

*Implicazione sulle metriche.* Una differenza sistematica di precisione dei
margini fra due sottoinsiemi introduce rumore nell'etichettatura che il modello
non può apprendere. Se un giorno le metriche di test risultassero
inspiegabilmente peggiori del previsto, vale la pena verificare come i campioni
delle due convenzioni si distribuiscono fra le partizioni.

**Etichette escluse.** `ferro` e `bud` sono presenti nelle annotazioni ma escluse
dalle maschere. Vengono scartate silenziosamente, distinguendole dalle etichette
*sconosciute*, che generano invece un avviso: la differenza tra "escluso di
proposito" ed "errore di annotazione" deve restare visibile.

**Priorità di disegno.** Le forme vengono rasterizzate in ordine di priorità
crescente; l'ultima disegnata prevale sui pixel sovrapposti.

- Multiclasse: `cordon` → `node` → `other` → `cane` (vince **Cane**)
- Binaria: `cane` → `other` → `node` → `cordon` (vince **Cordon**)

*Motivazione.* Nella mappa multiclasse il tralcio prevale perché è ciò che è
effettivamente visibile nel punto di sovrapposizione: il cordone vi passa dietro.
La maschera binaria ha invece lo scopo opposto — rappresentare il cordone come
struttura continua — e usa quindi la priorità inversa. Le due maschere rispondono
a due domande diverse: *cosa si vede in questo pixel* e *dove passa il cordone*.

**Trattamento dei linestrip.** I tralci sono annotati come polilinee aperte
(`linestrip`), non come poligoni: sono strutture allungate e sottili, e chiuderle
in un poligono produrrebbe un'area priva di significato. Vengono disegnate come
linee di un pixel e successivamente dilatate con kernel 3×3 per acquisire
spessore. Le altre classi sono annotate come poligoni e riempite.

La dilatazione è applicata una classe alla volta e scritta **solo dove la maschera
è ancora sfondo**, così non può mai sovrascrivere un'altra classe.

### Annotazioni scartate

L'esecuzione sul dataset definitivo segnala **4 forme scartate** su 100 file,
tutte di classe `Cane` (`tralcio` e `tralcio_secco`), annotate come `polygon`
con soli 2 punti:

| File | Etichetta |
|---|---|
| `blaze1_20260321_160518` | `tralcio` |
| `blaze1_20260321_162920` | `tralcio_secco` (×2) |
| `blaze1_20260324_161940` | `tralcio` |

Un poligono con due vertici non racchiude area e non può essere riempito.
Si tratta con ogni probabilità di errori di annotazione (strumento poligono
selezionato al posto di polilinea) oppure di tralci molto piccoli.

*Decisione:* lasciarle scartate. L'incidenza è trascurabile — 4 forme su 100
immagini, ciascuna con decine di forme annotate — e la segnalazione esplicita a
schermo rende il fenomeno tracciabile. L'alternativa, interpretare i poligoni
degeneri delle classi filiformi come polilinee, introdurrebbe una regola
implicita a fronte di un beneficio minimo.

*Verificato* durante la revisione delle annotazioni: sono le uniche quattro
forme del dataset che vengono perse, e nessuna forma viene scartata per tipo
non gestito. Le geometrie a due punti di `ferro` e `bud` non rientrano nel
conteggio: quelle etichette sono in `LABELS_TO_IGNORE` e non vengono mai
disegnate, e in LabelMe un `rectangle` o una `line` a due punti sono comunque
geometrie valide.

### Correzioni rispetto alla versione iniziale

- Un errore nel nome di un parametro (`ignore_labels` invece di `ignored_labels`)
  rendeva impossibile generare la maschera binaria. Non era emerso perché la
  funzione usciva anticipatamente quando le cartelle di output erano già piene.
- La logica di disegno era duplicata (disegno principale e dilatazione); è stata
  unificata in un'unica funzione.
- La colorazione delle maschere usa ora una lookup table invece di un ciclo sulle
  classi: una sola indicizzazione NumPy al posto di N confronti sull'immagine.
- La configurazione viene validata prima di scrivere sul filesystem, invece di
  fallire a metà elaborazione.

---

## 2. Generazione delle skeleton maps

**Input:** label map multiclasse e binaria.
**Output:** skeleton map con indici di classe, overlay colorato per ispezione.

Le skeleton map servono alla metrica **clDice** e alla **Skeleton Recall Loss**,
entrambe basate sulla linea centrale delle strutture anziché sull'area.

### Scelte

**Precalcolo offline.** Gli scheletri sono calcolati una volta e salvati su disco,
non ricalcolati nel training loop.

*Motivazione.* L'algoritmo di thinning Zhang–Suen è iterativo e costoso; eseguirlo
a ogni epoca dominerebbe il tempo di caricamento dei dati. Il precalcolo rende
inoltre gli scheletri identici tra run diverse, condizione necessaria per
confrontare le metriche.

**Scheletro del cordone estratto dalla maschera unificata.** Il cordone è una
struttura continua ma parzialmente occlusa dai tralci che gli passano davanti.
Scheletrizzando i soli pixel visibili si otterrebbe una linea centrale spezzata in
corrispondenza di ogni occlusione. Si parte quindi dalla maschera binaria
unificata (dove il cordone è continuo) e si rimuovono **a posteriori** i pixel
occlusi.

*Conseguenza.* Lo scheletro del cordone risulta topologicamente corretto — segue
l'asse reale della struttura — ma è definito solo dove il cordone è visibile. I
pixel occlusi non vengono conteggiati come linea centrale mancata: il modello non
può ragionevolmente predire cordone dove il cordone non si vede.

**Tubatura con clamping.** La linea centrale a un pixel viene dilatata per
acquisire spessore, e il risultato è ritagliato sulla maschera di partenza.

*Motivazione.* La dilatazione è isotropa: senza il ritaglio il tubo sconfinerebbe
nelle classi adiacenti ovunque la struttura sia più sottile del kernel.

**Assenza di conflitti tra classi.** Dopo il mascheramento delle occlusioni, ogni
scheletro vive solo sui pixel in cui la label map multiclasse contiene la propria
classe. Gli scheletri delle diverse classi sono quindi **disgiunti per
costruzione** e la fusione finale non deve risolvere alcun conflitto: la priorità
decisa a monte nella label map viene ereditata automaticamente.

*Verifica.* Su dati sintetici di controllo: zero pixel di scheletro fuori dalla
propria classe, zero pixel di scheletro del cordone in zona occlusa.

### Dimensione del kernel di dilatazione — decisione motivata

`dilation_kernel_size` è mantenuto a **2**, nonostante sia un valore pari e
quindi asimmetrico (la dilatazione non è centrata e sposta il tubo di mezzo
pixel). La scelta è stata verificata quantitativamente sull'intero dataset.

**Criterio adottato.** Si è misurata la *copertura*: la frazione della maschera
di classe occupata dal tubo dello scheletro. Una copertura elevata indica che il
tubo ha inghiottito l'intera struttura, per cui il clDice cessa di misurare la
connettività della linea centrale e degenera in una misura di sovrapposizione di
area — perdendo esattamente la proprietà per cui viene adottato.

**Risultati sulla classe `Cane`** (100 immagini):

| | kernel 2×2 | kernel 3×3 |
|---|---|---|
| copertura media | 34,8 % | 53,6 % |
| copertura massima | 56,6 % | **83,7 %** |
| immagini oltre l'80 % | **0** | 3 |

**Motivazione.** Il confronto è tra due difetti di natura diversa. Il kernel 2
introduce un'asimmetria geometrica di mezzo pixel, ma il tubo resta contenuto
nella struttura (grazie al clamping) e la connettività della linea centrale è
intatta, poiché deriva dall'algoritmo di thinning e non dalla dilatazione: è
quindi un difetto che non tocca la proprietà topologica misurata. Il kernel 3
comporta invece la degenerazione della metrica su 3 immagini. Il primo difetto è
cosmetico, il secondo metodologico.

**Osservazione sulle immagini critiche.** Le immagini con copertura più alta
(stem `20105`, `20106`, `20109`, `20110` — consecutivi, quindi della stessa
sessione di acquisizione) ritraggono **vigneto giovane**, con tralci
sensibilmente più sottili. Il rischio di degenerazione non è quindi casuale ma
legato a una sottopopolazione identificabile del dataset: se in futuro la quota
di vigneto giovane aumentasse, la scelta del kernel andrebbe riconsiderata.

**Alternativa non adottata.** Un elemento strutturante a croce 3×3
(`cv2.MORPH_CROSS`) sarebbe centrato e simmetrico ma dilaterebbe solo nelle
quattro direzioni ortogonali (area 5 contro 9 del quadrato, contro 4 del 2×2),
collocandosi tra i due casi. Richiederebbe un parametro aggiuntivo in
`config.py` per risolvere un problema che nei dati attuali non si manifesta.

---

## 3. Suddivisione train / validation / test

**Output:** `data/splits/splits.json`, con metadati e impronta del contenuto.

### Scelte

**Suddivisione sugli stem originali, prima dell'augmentation.** La partizione è
calcolata sui nomi dei file originali; le copie aumentate ereditano la partizione
dell'immagine da cui derivano.

*Motivazione.* È la protezione contro il data leakage più elementare e più
importante: se le copie ribaltate finissero in partizioni diverse, il modello
valuterebbe in test immagini che ha già visto in training, e le metriche sarebbero
prive di significato.

**File unico condiviso.** Tutti i notebook leggono la stessa partizione tramite
`load_splits()`. La versione iniziale salvava gli split due volte (un file con
nome basato su hash e un `splits.json` che lo sovrascriveva sempre), rendendo
ambiguo quale fosse la partizione effettiva.

**Impronta del contenuto.** Un hash MD5 troncato della partizione è salvato nei
metadati. Due run con lo stesso `splits_hash` hanno usato dati identici; se
differisce, le metriche non sono confrontabili.

**Validazione automatica.** Prima del salvataggio si verifica che nessuna
partizione sia vuota, che nessuno stem compaia in due partizioni e che la somma
copra tutti i campioni disponibili. Il secondo controllo è il rilevamento
automatico del leakage.

Rapporti adottati: 70 % train, 15 % validation, 15 % test, seed 42.

### Indipendenza dei campioni — verificata

La suddivisione assegna i file in modo indipendente l'uno dall'altro. Tale
procedura è corretta solo se le immagini sono acquisizioni distinte: se
provenissero da sequenze video di una stessa vite, frame contigui sarebbero
quasi identici e uno split casuale collocherebbe quasi-duplicati in partizioni
diverse, gonfiando le metriche di test.

**Le immagini del dataset non provengono da sequenze video**, ma sono
acquisizioni indipendenti. Lo split casuale è quindi metodologicamente valido e
non è necessaria una suddivisione per gruppi.

### Ripartizione effettiva

Su **100 immagini**: 70 train (70 %), 15 validation (15 %), 15 test (15 %).
Impronta della partizione: **`62de0ec4`**.

### Come si è arrivati a questo dataset

Il dataset ha attraversato tre stati, e le impronte lo registrano. Vale la pena
tenerne traccia perché **metriche prodotte con impronte diverse non sono
confrontabili**, e i risultati precedenti al refactoring appartengono al primo.

| Immagini | Impronta | Ripartizione | Cosa era successo |
|---|---|---|---|
| 93 | `d57f79eb` | 65/14/14 | stato iniziale |
| 92 | `7afaa392` | 64/14/14 | escluso un campione con annotazione incompleta |
| **100** | **`62de0ec4`** | **70/15/15** | campione riannotato e reintegrato, più otto immagini nuove |

**Il campione escluso.** `blaze-101-25367037-20087` mostrava un cordone ben
visibile ma **non annotato**: nel JSON non compariva alcuna forma
`cordone`/`cordon`, e la label map risultava priva di pixel di quella classe.
Era l'unica immagine su 93 in questa condizione. È stato tenuto fuori dal
dataset finché l'annotazione non è stata completata; ora contiene 7 543 pixel
di cordone ed è rientrato.

*Meccanismo usato per l'esclusione.* I quattro file del campione furono spostati
in una cartella separata che rispecchiava la struttura di `data/raw/`, invece di
essere cancellati o filtrati da una lista nel codice. La pipeline parte dai JSON
presenti in `data/raw/annotations/`, quindi un campione spostato fuori sparisce
da solo da label map, split e dataset aumentato, **senza alcuna modifica al
codice** — e reintegrarlo è stato altrettanto semplice. La cartella è stata
rimossa una volta completata la riannotazione, non essendoci più campioni
esclusi.

> **Perché ogni cambiamento del dataset ridisegna l'intera partizione.**
> `train_test_split` mescola l'elenco completo degli stem: aggiungendone o
> togliendone anche uno solo, l'assegnazione di *tutti* gli altri cambia. Non
> esiste una partizione "quasi uguale" alla precedente, ed è la ragione per cui
> il dataset è stato portato alla sua forma definitiva **prima** di eseguire
> qualunque esperimento.

---

## 4. Augmentation offline

**Output:** `data/augmented/{train,val,test}/` con tutte le modalità.

### Scelte

**Flip applicati offline, non nel training loop.** Ogni immagine di training è
salvata insieme a tre copie ribaltate (orizzontale, verticale, entrambe).

*Motivazione.* Con i dati su Google Drive la lettura è il collo di bottiglia, e un
insieme aumentato fisso rende le run direttamente confrontabili: due
configurazioni diverse vedono esattamente gli stessi campioni.

**Solo trasformazioni geometriche di simmetria.** I flip sono scelti perché
preservano esattamente le etichette: nessuna interpolazione, nessun pixel di
bordo inventato, indici di classe intatti. Rotazioni o deformazioni elastiche
richiederebbero interpolazione nearest-neighbour sulle label e introdurrebbero
artefatti sulle strutture sottili, che sono proprio l'oggetto del problema.

**Validation e test scritti senza flip.** Le tre cartelle hanno la stessa
struttura, così la classe `Dataset` le tratta in modo identico; solo il contenuto
di train è aumentato.

Fino alla correzione dei canali queste due partizioni venivano copiate **byte per
byte** con `shutil.copy2`, che duplica il file senza mai aprirlo. Ora ogni file
viene **decodificato** — il PNG compresso viene letto e ricostruito come array di
pixel in memoria — e **ricodificato** da quell'array in un file nuovo.

*Cosa cambia davvero.* Sui valori dei pixel nulla: il PNG è un formato senza
perdite, quindi l'array riletto da disco è identico a quello scritto. Cambiano
solo tre cose, nessuna delle quali ha effetto sul training: il numero di canali,
che è lo scopo dell'operazione; i metadati accessori del PNG, che `imwrite`
rigenera da zero e quindi non conserva; la dimensione del file, perché il livello
di compressione in scrittura può differire da quello del sorgente.

*Perché la copia letterale non era più utilizzabile.* Il numero di canali di un
PNG non è modificabile senza decodificarlo — è una proprietà di come i pixel sono
compressi nel file. `copy2` non guarda dentro, quindi trasportava intatto il
formato del sorgente e lasciava validation e test con esattamente l'incoerenza
che `read_modality_image()` esiste per rimuovere. Il train ne era immune solo per
un motivo accidentale: veniva già decodificato per applicare i flip.

La ricodifica non è quindi un miglioramento in sé, è il prezzo da pagare per
poter intervenire sui canali. Verificato dopo il cambiamento: le copie ribaltate
coincidono ancora esattamente con `cv2.flip` applicato ai rispettivi originali,
per tutte e sei le modalità.

**Tutte le modalità sempre elaborate.** L'augmentation non tiene conto di
`ENABLED_MODALITIES`: il dataset aumentato descrive i *dati*, non l'*esperimento*.

*Motivazione.* Nella versione iniziale, disattivare una modalità produceva un
dataset privo di quel canale; riattivandola in seguito, il controllo "cartelle già
popolate" impediva la rigenerazione e il training procedeva su un dataset
incompleto senza alcun errore. La selezione dei canali spetta alla classe
`Dataset`, non alla preparazione dei dati.

**Lettura senza conversioni implicite.** Le modalità grezze sono lette con
`IMREAD_UNCHANGED`, che preserva profondità di bit e numero di canali. La versione
iniziale usava `cv2.imread()` senza flag, che converte sempre in BGR a 8 bit:
sui file ToF a 16 bit questo comportava una perdita di informazione silenziosa, e
salvava le immagini in scala di grigi su tre canali identici.

**Formato effettivo dei dati — conclusione corretta in fase di EDA.** Una nota
precedente affermava che i file di intensità fossero `uint8` con forma
`(480, 640)`, quindi a canale singolo. **L'affermazione era falsa per un quarto
del dataset.** L'ispezione sistematica di tutti i file, eseguita nella sezione 1
del notebook 02, ha mostrato due lotti di acquisizione con formati diversi:

| Lotto | File | `intensity` / `confidence` |
|---|---|---|
| `blaze-101-25367037-*` | 69 | 1 canale, `(480, 640)` |
| `blaze1_2026*` | 24 | 3 canali, `(480, 640, 3)` |

I 24 file a tre canali contengono **tre piani identici**: un'immagine in scala
di grigi salvata come RGB, verificata su tutti e 24 e per entrambe le modalità.
`depth` è invece genuinamente a tre canali su tutti i file (0 con piani
identici), quindi `MODALITY_CHANNELS['depth'] = 3` era ed è corretto.

*Perché l'errore era sfuggito.* La funzione originale accumulava i formati in un
insieme, segnalava con un avviso la presenza di più formati, e poi stampava
`next(iter(formats))` — **un formato arbitrario estratto dall'insieme**. La riga
di riepilogo mostrava quindi un solo formato e sembrava conclusiva, mentre
l'avviso passava inosservato. La funzione che l'ha sostituita stampa `MIXED (n)`
al posto di un rappresentante: un valore non uniforme deve essere visibile come
tale, non riassunto.

*Conseguenza se non corretto.* La classe `Dataset` concatena le modalità: avrebbe
prodotto tensori a 5 canali per 69 campioni e a 9 canali per 24. Il caso migliore
è un errore al momento di impilare il batch; il peggiore è un "aggiustamento" con
uno `squeeze` che disallinea silenziosamente i canali.

**Riconciliazione dei canali in lettura.** La regola è implementata in
`read_modality_image()`: si legge con `IMREAD_UNCHANGED`, e se il file ha più
canali di quanti `config.MODALITY_CHANNELS` ne dichiari, si verifica che i piani
siano identici e si tiene il primo. Se i piani differiscono la funzione
**solleva un'eccezione**, perché quella sarebbe una perdita di informazione e non
una ridondanza di formato.

*Motivazione della collocazione.* I dati grezzi restano intatti: sono il dato
consegnato e non vanno riscritti. La regola vive nel punto di lettura, quindi
esiste una sola volta ed è condivisa da tutti i consumatori — augmentation
offline, analisi esplorativa e la futura classe `Dataset` — invece di dover
essere ricordata in tre posti. Il punto aperto sull'adattamento della classe
`Dataset` all'intensità a canale singolo è così chiuso alla radice: il dato che
le arriva è già conforme alla configurazione.

**Verifica preventiva dei file.** L'esistenza di tutti i file sorgente è
controllata prima di scrivere qualsiasi output, per non lasciare le cartelle in
stato incoerente.

**Controllo di coerenza finale.** Al termine si verifica che tutte le modalità
abbiano lo stesso numero di file per partizione: una discrepanza significherebbe
immagini ed etichette non più appaiate, condizione che corromperebbe il training
in modo silenzioso.

### Verifiche eseguite

Su dati sintetici di controllo:

- profondità a 16 bit preservata tra originale e copia
- copia ribaltata identica a `cv2.flip` applicato all'originale, per immagini
  ed etichette — quindi allineamento geometrico esatto
- insieme degli indici di classe invariato dopo il ribaltamento
- nessuna copia aumentata nelle partizioni di validation e test
- nessuna sovrapposizione tra gli stem di train e quelli di test

---

## 5. Analisi esplorativa (notebook 02)

**Input:** dataset preparato dal notebook 01.
**Output:** `class_weights.json`, `channel_stats.json`, figure di controllo.

### Scelte

**Statistiche descrittive su tutti gli split, parametri dal solo training set.**
Distribuzione delle classi e statistiche di canale vengono calcolate e riportate
per train, validation, test e per il dataset intero. Solo i valori derivati dal
**training set** vengono però salvati su disco e usati durante l'addestramento.

*Motivazione.* Pesi delle classi e statistiche di normalizzazione entrano nel
modello: calcolarli sull'intero dataset farebbe filtrare nell'addestramento
informazione proveniente da validation e test. È una forma lieve di leakage, ma
verificabile e facilmente contestabile in sede di discussione. Le stesse
statistiche calcolate per partizione restano invece utili come **controllo
descrittivo**: mostrano se lo split casuale ha prodotto partizioni comparabili e
segnalerebbero un eventuale scostamento sistematico fra i lotti di acquisizione.

*Cautela nella lettura.* Con 14 immagini in validation e in test la numerosità
campionaria effettiva è il numero di **immagini**, non di pixel: i pixel dentro
una stessa immagine sono fortemente correlati, perché una classe o è presente
nella scena o non lo è. Le proporzioni per partizione vanno quindi presentate
come confronto di plausibilità, non come prova di equivalenza statistica.

**Presenza delle classi come controllo separato dalla quota di pixel.** Oltre
alla frazione di pixel si conta in **quante immagini** ciascuna classe compare.

*Motivazione.* Una classe assente da una partizione rende il suo IoU indefinito
su quella partizione, e di conseguenza la mean IoU non confrontabile fra run —
proprio la metrica che guida `MONITOR_METRIC` e la tabella comparativa finale.
La quota di pixel non rileva questa condizione: una classe può avere una
frazione piccola ma diffusa, oppure grande ma concentrata in poche immagini.

**Statistiche misurate sulle immagini originali, non su quelle aumentate.** Le
label map vengono lette da `data/processed/`, filtrate secondo lo split, invece
che dalle cartelle di `data/augmented/`.

*Motivazione.* Un flip è una permutazione di pixel: non altera l'istogramma dei
valori. Le copie ribaltate portano quindi esattamente gli stessi conteggi
dell'originale, quattro volte. Leggerle darebbe frequenze, medie e deviazioni
standard identiche al costo di quattro volte il tempo.

**Firma di provenienza nei valori salvati.** `class_weights.json` e
`channel_stats.json` registrano lo `splits_hash` della partizione su cui sono
stati misurati; al caricamento il valore viene confrontato con quello corrente e
una discrepanza produce un avviso.

*Motivazione.* Senza questa firma, ricalcolare lo split lascerebbe su disco pesi
e statistiche relativi alla partizione precedente. Il training proseguirebbe con
valori sbagliati senza alcun errore. Il confronto è a carico del chiamante:
`config.py` non può leggere lo split da sé, perché importare `data_preparation`
creerebbe un ciclo di import.

**Ordine dei canali dichiarato esplicitamente.** `MODALITY_ORDER` è derivato
dall'ordine di dichiarazione di `MODALITY_CHANNELS`, e `build_channel_names()`
assegna un nome a ciascuno dei cinque canali.

*Motivazione.* L'EDA misura cinque medie e cinque deviazioni standard in un certo
ordine; la classe `Dataset` le applicherà impilando le modalità nel proprio
ordine. Se i due divergessero, ogni canale verrebbe normalizzato con le
statistiche di un altro — un errore che non produce eccezioni e non è visibile
nelle metriche. I nomi dei canali sono salvati nel file e verificati in lettura.

**Rimozione del metodo `effective_num` per i pesi delle classi.** Dei cinque
metodi previsti ne restano quattro: `sqrt_inverse`, `inverse`, `median_freq`,
`none`.

*Motivazione.* La formula del numero effettivo di campioni, `(1 - β) / (1 - βⁿ)`
con β = 0,9999, è pensata per conteggi di **campioni**, dell'ordine delle
migliaia. Applicata a conteggi di **pixel**, dell'ordine dei milioni, `βⁿ` è
indistinguibile da zero per ogni classe: i pesi collassano tutti sullo stesso
valore e, dopo la normalizzazione, il metodo restituisce pesi uniformi. Non
avrebbe quindi avuto alcun effetto. L'implementazione precedente mascherava il
problema introducendo un fattore di scala arbitrario (`total_samples = 1000`) che
rendeva il risultato dipendente da un numero inventato.

### Distribuzione misurata

Su 100 immagini, partizione `62de0ec4`. Quota di pixel per classe:

| Classe | train | val | test | totale |
|---|---|---|---|---|
| Background | 83,55 % | 86,89 % | 83,55 % | 84,05 % |
| Cane | 11,57 % | 8,33 % | 11,50 % | 11,07 % |
| Cordon | 3,55 % | 3,74 % | 3,64 % | 3,59 % |
| Node | 0,27 % | 0,22 % | 0,34 % | 0,27 % |
| Other | 1,06 % | 0,83 % | 0,97 % | 1,01 % |

Numero di immagini in cui ciascuna classe compare:

| Classe | train | val | test | totale |
|---|---|---|---|---|
| Background | 70/70 | 15/15 | 15/15 | 100/100 |
| Cane | 70/70 | 15/15 | 15/15 | 100/100 |
| Cordon | 70/70 | 15/15 | 15/15 | 100/100 |
| Node | 52/70 | 12/15 | 12/15 | 76/100 |
| Other | 67/70 | 14/15 | 14/15 | 95/100 |

**Lo sbilanciamento è forte ma non estremo.** Background è circa 313 volte più
frequente di `Node`. Con `sqrt_inverse` questo si traduce in un peso di 17,7
volte quello dello sfondo — elevato ma trattabile, ed è esattamente la ragione
per cui quel metodo è preferito a `inverse`, che produrrebbe un fattore 313 e
renderebbe l'addestramento instabile sulle classi rare.

**`Node` non è solo raro, è spesso assente.** Compare in 76 immagini su 100:
manca del tutto da un quarto del dataset. È un'informazione che la sola quota di
pixel nascondeva, e ha due conseguenze — le metriche calcolate per immagine su
`Node` sono indefinite in 24 casi, e la varianza del suo IoU sarà alta. È il
dato che motiva la valutazione riportata più sotto.

**Il controllo di presenza passa.** Ogni classe compare in ogni partizione,
quindi la mean IoU resta ben definita ovunque. Il margine più stretto è `Node`,
presente in 12 immagini su 15 sia in validation sia in test.

**Le partizioni sono comparabili, con un'avvertenza.** Lo scostamento maggiore è
su `Cane`: 8,33 % in validation contro 11,57 % nel train, cioè circa un quarto
in meno. Su 15 immagini rientra nella variabilità attesa — la numerosità
efficace è il numero di immagini, non di pixel — ma è uno scarto da tenere
presente, perché la validation è ciò su cui si decidono checkpoint ed early
stopping. Se il comportamento in validation risultasse anomalo rispetto al test,
è il primo posto dove guardare.

### Statistiche di canale

Calcolate in una sola passata con somme incrementali in `float64` — somma,
somma dei quadrati, istogramma a 256 bin per canale — invece di accumulare i
valori dei pixel. L'implementazione originale raccoglieva ogni pixel in liste
Python: su questo dataset significa oltre cento milioni di oggetti e diversi
gigabyte di memoria, per produrre due numeri per canale.

La varianza è ottenuta come `E[x²] - E[x]²`, che richiede una passata sola. La
cancellazione numerica che rende rischiosa questa forma si manifesta quando la
deviazione standard è minuscola rispetto alla media; con valori a 8 bit
accumulati in `float64` il margine è di dieci ordini di grandezza.

**Ordine dei canali in memoria.** OpenCV decodifica le immagini a colori in
ordine **BGR**. I suffissi in `config` sono stati corretti di conseguenza
(`depth_B`, `depth_G`, `depth_R`): chiamare `depth_R` il canale 0 sarebbe stata
documentazione falsa, e avrebbe invitato ad aggiungere una conversione di
colore che avrebbe applicato a ogni canale le statistiche di un altro.

**Valori misurati sul training set** (70 immagini, partizione `62de0ec4`):

| Canale | Media | Dev. std |
|---|---|---|
| intensity | 82,27 | 27,76 |
| depth_B | 83,87 | 118,71 |
| depth_G | 62,07 | 94,87 |
| depth_R | 34,43 | 77,72 |
| confidence | 10,46 | 18,86 |

### Pesi delle classi

Calcolati sulle sole frequenze del training set. Con `sqrt_inverse` e
normalizzazione al minimo:

| Classe | Quota pixel (train) | Peso |
|---|---|---|
| Background | 83,55 % | 1,00 |
| Cane | 11,57 % | 2,69 |
| Cordon | 3,55 % | 4,85 |
| Node | 0,27 % | 17,69 |
| Other | 1,06 % | 8,88 |

Il rapporto fra la classe più pesante e la più leggera è **17,7×**. Con
`inverse` sarebbe **313×**: un fattore di quell'ordine rende l'addestramento
instabile sulle classi rare, perché ogni pixel di `Node` sbagliato pesa quanto
313 pixel di sfondo e il gradiente viene dominato da poche decine di pixel per
immagine. È la ragione per cui `sqrt_inverse` resta il metodo predefinito.

*Riferimento nel dominio.* Majeed et al. usano `median_freq` non normalizzato su
un dataset ancora più sbilanciato — 0,3 % di filo di ferro contro il nostro
0,27 % di `Node` — ottenendo un rapporto di circa 307× e un'accuratezza di 0,97
proprio sulla classe più sottile. Il fattore aggressivo è quindi praticabile in
questo dominio, e `inverse` merita di essere provato come seconda
configurazione invece di essere scartato a priori.

**`median_freq` non è un metodo distinto quando la normalizzazione è attiva.**
Normalizzando si divide ogni peso per il minimo, e il fattore mediana — che è
una costante — si semplifica: `(med/fᵢ)/(med/f_max) = f_max/fᵢ`, cioè
esattamente ciò che restituisce `inverse`. I due metodi differiscono solo con
`normalize=False`. Il metodo resta disponibile ma è escluso dal grafico di
confronto, dove avrebbe prodotto due serie identiche.

### Natura del canale depth — ritrovamento

Le deviazioni standard dei canali depth sono anomale: 118 su un intervallo
0–255. Gli istogrammi mostrano il motivo, e l'analisi che ne è seguita ha
chiarito la natura del dato.

**Il depth è una colorazione JET di uno scalare.** I file contengono circa 1 000
colori distinti che seguono il percorso blu → ciano → verde → giallo → rosso.
I tre canali trasportano quindi l'informazione di **una sola grandezza**, e in
forma non monotona: in JET nessun singolo canale cresce monotonamente con la
profondità. La rete deve di fatto invertire una colormap prima di poter usare
la distanza.

**Il 57 % dei pixel non ha profondità valida.** `depth == (0,0,0)` e
`confidence == 0` identificano gli stessi pixel, con un accordo del 99,97 %:
sono i punti in cui il sensore ToF non ha ricevuto ritorno. Ne consegue che il
canale `confidence` è in larga parte **ridondante**, perché la sua informazione
di validità coincide con quella già presente nel depth; aggiunge qualcosa solo
attraverso i valori non nulli, la cui mediana è 2 su 255.

**L'invalidità è concentrata sullo sfondo**, il che rende il segnale ToF
comunque utile:

| Classe | Pixel senza profondità valida |
|---|---|
| Background | 68,3 % |
| Cane | 11,3 % |
| Cordon | 2,1 % |
| Node | 0,8 % |
| Other | 10,8 % |

Il ToF misura quasi ovunque ci siano le strutture da segmentare e tace dove c'è
sfondo: la presenza stessa di una profondità valida è un indizio di foreground.

**Decisione: la rappresentazione dei dati resta invariata.** Le statistiche di
normalizzazione continuano a essere calcolate su tutti i pixel, inclusi gli
zeri dei punti invalidi — sono ciò che il modello riceve in ingresso. I dati
grezzi in millimetri non sono disponibili, e a un mese dalla consegna il
percorso che porta a esperimenti conclusi ha priorità su una rappresentazione
migliore ma da validare.

*Sviluppi possibili, se il tempo lo permettesse:* ricostruire lo scalare
invertendo la colormap e sostituire i tre canali con profondità più maschera di
validità — l'ingresso passerebbe da 5 a 3 canali **guadagnando** informazione;
oppure verificare quanto `confidence` contribuisca davvero, data la ridondanza
misurata.

### Copertura degli scheletri — analisi resa riproducibile

Il confronto fra kernel 2 e kernel 3 documentato nella sezione 2 era stato
eseguito una volta in modo estemporaneo. È ora una funzione del modulo, che
ricostruisce gli scheletri **in memoria** per ciascun kernel tramite
`build_skeleton_map()`: i file scritti dal notebook 01 non vengono né letti né
sovrascritti. Eseguita sulle 100 immagini del dataset definitivo.

| | kernel 2×2 | kernel 3×3 |
|---|---|---|
| copertura media `Cane` | 34,8 % | 53,6 % |
| copertura massima `Cane` | 56,6 % | **83,7 %** |
| immagini `Cane` oltre l'80 % | **0** | 3 |
| copertura media `Cordon` | 13,2 % | 20,2 % |
| copertura massima `Cordon` | 33,2 % | 50,2 % |

Il `Cordon` non si avvicina alla soglia con nessuno dei due kernel: è una
struttura più spessa dei tralci, quindi il tubo dello scheletro ne occupa una
frazione minore. Il vincolo viene interamente dalla classe `Cane`.

Il grafico mostra la distribuzione completa per immagine invece di due barre
riassuntive, perché la decisione dipende dalla **coda superiore**: un kernel è
inutilizzabile se alcune immagini superano la soglia, per quanto bene si
comporti sulle altre.

> **Come riprodurre i valori del cordone.** Per `Cane` il numeratore è il tubo
> dello scheletro della maschera visibile. Per `Cordon` **no**: lo scheletro
> viene estratto dalla maschera binaria unificata e le occlusioni sono rimosse
> dopo (sezione 2), mentre il denominatore resta il cordone *visibile*.
> Scheletrizzando direttamente il cordone visibile si ottiene il 19,0 % invece
> del 13,2 %: sono due grandezze diverse, e i valori qui riportati sono quelli
> prodotti da `exploration.compute_skeleton_coverage()`, cioè quelli relativi
> agli scheletri effettivamente scritti su disco.

### Verifiche visive

Due controlli che nessuna metrica sostituisce.

**Campioni con maschera sovrapposta all'immagine.** È l'unica vista in cui un
disallineamento fra immagine e label map sarebbe visibile a colpo d'occhio.

**Augmentation offline.** Ogni campione di training è mostrato accanto alle sue
copie ribaltate, immagine sopra ed etichetta sotto, lette da
`data/augmented/train/`. Un flip applicato all'immagine ma non alla sua
etichetta sarebbe evidente qui e invisibile in qualunque metrica finché il
modello non fallisse ad apprendere.

La visualizzazione dell'augmentation **online** non appartiene a questo
notebook: richiede la classe `Dataset` e la pipeline `albumentations`, che non
è installabile in locale. Verrà fatta nel notebook 03.

### Sviluppo futuro: eliminazione della classe `Node`

`Node` è la classe più problematica: circa lo 0,3 % dei pixel, e assente da un
quarto delle immagini. Si è quindi valutata l'ipotesi di eliminarla.

**L'ipotesi iniziale era di assorbirla in `Cordon`**, sul presupposto che i nodi
si trovino sul cordone. Le misure non la sostengono:

| Misura | Valore |
|---|---|
| Pixel di `Node` interni alla maschera del cordone | 54,0 % |
| Bordo dei pixel di `Node` su `Cordon` | 38,7 % |
| Bordo dei pixel di `Node` su `Cane` | 32,0 % |
| Bordo dei pixel di `Node` su `Background` | 28,7 % |

Quasi metà dei nodi **non** si trova sul cordone, coerentemente con il fatto che
un nodo è tale su qualsiasi germoglio. Il caso limite lo confermava: il campione
descritto in sezione 3 conteneva due nodi annotati e, prima della
riannotazione, nessun cordone. Una fusione
uniforme verso `Cordon` etichetterebbe come cordone circa il 45 % di pixel che
stanno sui tralci, corrompendo entrambe le classi.

**Meccanismo corretto, se si decidesse di procedere:** aggiungere `nodo`/`node`
a `config.LABELS_TO_IGNORE`. I poligoni dei nodi non verrebbero rasterizzati e
quei pixel conserverebbero la struttura sottostante — cordone dove il nodo sta
sul cordone, tralcio dove sta sul tralcio. È una fusione corretta pixel per
pixel invece che una regola uniforme sbagliata in metà dei casi, e costa una
riga di configurazione.

*Da verificare prima di adottarlo:* quanti pixel di `Node` finirebbero su
`Background`. Il 28,7 % di bordo su sfondo suggerisce che una parte dei poligoni
deborda oltre le strutture annotate.

---

## 6. Dataset, augmentation online e dataloader (`dataset.py`)

**Input:** `data/augmented/`, `channel_stats.json`.
**Output:** batch `(immagine, label, skeleton)` pronti per il training.

### Scelte

**Lettura tramite `read_modality_image()`.** La classe `Dataset` non legge i file
da sé: passa dalla stessa funzione usata dall'augmentation offline e dall'EDA.
Riceve quindi sempre il numero di canali dichiarato in `MODALITY_CHANNELS`, e la
questione aperta sull'adattamento all'intensità a canale singolo non si pone.

**Impilamento secondo `MODALITY_ORDER`.** Lo stesso ordine in cui l'EDA ha
misurato le statistiche. La selezione delle statistiche dei soli canali attivi
avviene **per nome** e non per posizione: disattivando una modalità intermedia,
un'indicizzazione posizionale sposterebbe ogni canale successivo sulle
statistiche sbagliate senza sollevare nulla.

**Nessuna trasformazione fotometrica.** Decisione metodologica: alterare
luminosità, contrasto o rumore cambierebbe il significato di un valore di
profondità e di una misura di affidabilità. Applicarle al solo `intensity`
romperebbe inoltre la coerenza fisica fra canali che descrivono lo stesso pixel.
La variabilità di aspetto viene quindi dai soli flip offline.

*Nota sul codice originale.* La pipeline cromatica era definita e configurata,
ma la classe `Dataset` scavalcava l'oggetto `MultiModalTransform` e ne invocava
a mano le fasi `geometric` e `normalizer`, saltando `color`. Il risultato era
coerente con la decisione presa, ma per una via accidentale: il codice
dichiarava un comportamento e ne eseguiva un altro. L'infrastruttura è stata
rimossa, così la decisione è leggibile invece che dedotta.

### Ingrandimento: perché il pre-ingrandimento e come sono stati scelti i valori

Il ramo di training ingrandisce l'immagine, applica la geometria casuale e
ritaglia il centro alla dimensione finale. Il pre-ingrandimento serve a far
cadere il ritaglio su **pixel reali**: rotazione e distorsione spingono dentro
la finestra contenuto proveniente da fuori, invece di rifletterci un bordo.

Il ramo di validation ridimensiona e basta, perché i suoi campioni devono
restare identici fra epoche e fra run.

**Formula dell'ingrandimento.** Detto `f` il fattore di pre-ingrandimento e `s`
la scala casuale, un campione di training risulta ingrandito di

```
ingrandimento = (dimensione_target / larghezza_sorgente) · f · s = 0,8 · f · s
```

dove `0,8 = 512 / 640` è fisso, dati `IMG_SIZE = (512, 512)` e sorgenti
480×640. Validation e test corrispondono al caso `f · s = 1`, cioè 0,80.

*(I conti sono sulla larghezza. Sull'altezza il fattore fisso sarebbe
`512/480 = 1,067`, perché l'immagine viene deformata da 3:4 a quadrata. La
deformazione è però identica nei due rami, quindi il confronto train/val non
dipende dall'asse scelto.)*

**Situazione con i valori originali** (`f = 1,5`, `s ∈ [0,70 ; 1,30]`):

| | `f · s` | Ingrandimento | Area visibile |
|---|---|---|---|
| val/test | 1,00 | **0,80** | 100 % |
| train, `s = 0,70` | 1,05 | 0,84 | 91 % |
| train, `s = 1,00` | 1,50 | 1,20 | 44 % |
| train, `s = 1,30` | 1,95 | 1,56 | 26 % |

Il modello viene quindi valutato a un ingrandimento **inferiore a qualunque
cosa abbia visto in addestramento**: 0,80 contro un minimo di 0,84.

**Il vincolo vero non è il ritaglio, è il bordo riflesso.** `CenterCrop`
richiede solo `f · s ≥ 1`, ma quella condizione garantisce che il ritaglio *sia
possibile*, non che cada su pixel reali. Ruotando di 15° un quadrato di lato
`L`, il quadrato inscritto fatto di soli pixel reali ha lato
`L / (cos 15° + sin 15°) = L / 1,2247`. Il ritaglio resta dentro finché

```
f · s ≥ cos 15° + sin 15° = 1,2247
```

che con `f = 1,5` significa **`s ≥ 0,8165`**, ben più severo di `s ≥ 2/3`.

**Difetto trovato con questa misura.** Il bound precedente, `s ∈ [0,68 ; 1,30]`,
violava la condizione nel 22 % dei casi in cui `RandomScale` si attivava. Misura
con sentinella su `cv2.warpAffine` — immagine costante, bordo riempito a zero,
si contano i pixel del ritaglio che non valgono la costante:

| `s` | lato intermedio | pixel riflessi nel ritaglio |
|---|---|---|
| 0,68 | 522 | **8,58 %** |
| 0,80 | 614 | 0,15 % |
| 0,8165 | 627 | 0,002 % |
| **0,82** | 629 | **0,000 %** |

Il pre-ingrandimento esisteva proprio per evitare i pixel riflessi, e il bound
scelto per avvicinarsi alla scala di valutazione li stava reintroducendo.

Da notare che alla soglia analitica esatta resta lo 0,002 %: l'interpolazione
bilineare campiona un intorno 2×2 e sborda di un pixel oltre il limite
geometrico. La soglia calcolata non basta da sola; serve la misura.

**La soglia analitica non basta: la catena non è la rotazione.** Fissato
`s = 0,82`, la cella 3.4 del notebook 00 ha rilevato contaminazione su
**una estrazione su 400**, per lo 0,12 % di quel ritaglio. Il fenomeno è stato
riprodotto in locale ricostruendo la catena con `cv2` — replica validata contro
il dato di Colab, 0,10 % di estrazioni contaminate contro 0,25 % osservato,
compatibili dato che l'osservazione poggia su un solo evento.

Isolando i transform a `s = 0,82`:

| combinazione | contaminazione |
|---|---|
| scala + rotazione | 0,0000 % |
| scala + distorsione | 0,0000 % |
| **scala + rotazione + distorsione** | **0,7721 %** |

Ognuno da solo è pulito; è la **composizione** a rompere, e solo per `k > 0`,
quando la distorsione campiona verso l'esterno e pesca nella fascia di bordo
che la rotazione ha appena prodotto. La causa è l'assenza di margine: a
`s = 0,82` il quadrato ruotato del ritaglio arriva a ±313,5 px su un'immagine
di semilato 315, cioè **1,5 px di slack**. La soglia `0,8165` descriveva la
rotazione *da sola* ed è stata applicata a una catena di tre trasformazioni.

**Due rimedi misurati e scartati.** Invertire l'ordine (distorsione prima della
rotazione) dà 54 combinazioni contaminate su 8125 contro 56: con il ritaglio
così stretto ciascuna delle due può spingere l'altra oltre il bordo. Ridurre
`DISTORTION_LIMIT` a ±0,02, cioè quasi spegnerla, lascia ancora 0,0141 %. La
sola manopola efficace è il margine, cioè un pavimento più alto su `s`.

**Valore adottato.** `scale_limit = (-0,10 ; 0,30)`, cioè `s ∈ [0,90 ; 1,30]`.
Sulla catena completa la contaminazione si azzera a `s = 0,87`; si adotta 0,90
per stare sopra la soglia e non appoggiati. Verificato a 0,0000 % su una
griglia esaustiva di 13 671 combinazioni di scala, angolo e distorsione e su
20 000 estrazioni Monte Carlo che riproducono i `p = 0,5` di ogni transform.

**Costo.** Il pavimento di ingrandimento passa da 1,23 a 1,35, ma è la *coda*
della distribuzione, non il centro: `RandomScale` si attiva metà delle volte,
quindi il grosso del ramo geometrico resta a `s = 1`. L'ingrandimento medio
complessivo passa da 1,272 a 1,288, cioè dell'1,2 %.

**Alternativa scartata.** Con `ROTATION_LIMIT = 10°` la soglia scende a
`s = 0,83` e il pavimento a 1,25: si recupera 0,10 di ingrandimento dimezzando
il range di rotazione. Non conviene.

**Il pavimento geometrico.** Un campione ruotato di ±15° e ritagliato senza
pixel inventati è ingrandito di **almeno 1,2247** rispetto alla valutazione, e
questo *non dipende da `f`*: abbassare il pre-ingrandimento alza in proporzione
la soglia su `s`. L'unica manopola che sposta il pavimento è `ROTATION_LIMIT`
(a 10° diventa 1,158, a 5° 1,083). Ingrandisci-e-ritaglia non può quindi
raggiungere la scala di valutazione, contrariamente a quanto affermava la
versione precedente di questa sezione: quella conclusione considerava solo la
fattibilità del ritaglio.

**Ramo identità.** Poiché il pavimento è invalicabile, la scala di valutazione
viene coperta da un secondo ramo, **alternativo ed esclusivo**: con
`IDENTITY_PROBABILITY = 0,5` il campione salta del tutto la geometria e riceve
il solo ridimensionamento, cioè esattamente la pipeline di validation. Un
campione è o la vista di valutazione o una vista perturbata, mai un compromesso.

| ramo | `f · s` | ingrandimento | area visibile |
|---|---|---|---|
| val/test | 1,00 | 0,80 | 100 % |
| train, identità (50 %) | 1,00 | 0,80 | 100 % |
| train, geometrico, `s = 0,90` | 1,35 | 1,08 | 69 % |
| train, geometrico, `s = 1,30` | 1,95 | 1,56 | 26 % |

La distribuzione di scala vista dal modello è quindi **bimodale**: 50 % dei
campioni esattamente alla scala di valutazione, 31 % fra 1,35 e 1,50, 19 % fra
1,50 e 1,95. Resta scoperto l'intervallo fra 1,00 e 1,35, che nessun campione
occupa.

Non è un problema — la scala su cui il modello è valutato è coperta
*esattamente* dal ramo identità — ma è una caratteristica dichiarata della
distribuzione, non un effetto collaterale ignorato. Il vuoto discende dal
pavimento geometrico della rotazione: chiuderlo richiederebbe di condizionare
il range di scala al fatto che la rotazione sia scattata o meno, cosa non
esprimibile dentro una `Compose` di `albumentations` senza codice dedicato.

**Perché scala e non ritaglio casuale.** Una traslazione casuale simulerebbe
una telecamera che inquadra altrove, cosa che in vigneto non accade: la
telecamera guarda sempre la pianta, al più da distanza leggermente diversa. Una
variazione di scala *è* un campo visivo diverso; un ritaglio casuale sarebbe una
scena diversa. Per lo stesso motivo si è escluso lo specchiamento come rimedio
al bordo: un vigneto riflesso non è una struttura che il modello incontrerà mai.

**Perché la verifica sta nel notebook e non solo qui.** La taratura di
`RANDOM_SCALE_LIMIT` viene da una ricostruzione della catena in `cv2`, non dal
codice di `albumentations`. La replica è stata validata contro il fallimento
osservato su Colab, ma resta una replica. La cella 3.4 del notebook 00 fa
quindi passare un'immagine costante con bordo a sentinella per la catena vera,
conta i pixel provenienti da fuori e ha un `assert` a zero: è quella, non
questa sezione, l'autorità sull'affermazione «nessun pixel inventato raggiunge
il modello».

L'episodio è anche il motivo per cui la cella esiste. Il difetto era invisibile
a occhio — un campione su 400, lo 0,1 % dei suoi pixel — e nessuna metrica di
training lo avrebbe segnalato.

### Interpolazione bilineare su un depth colorato — postilla

Il ridimensionamento usa `INTER_LINEAR`, che assegna a ogni pixel di output la
media pesata dei quattro pixel di input più vicini, **canale per canale**.
Poiché JET associa uno scalare a un punto su una *curva* nello spazio RGB,
mediare due colori equivale a prendere un punto sul segmento che li congiunge:
se la curva non è dritta, il risultato ne esce.

Misurato su un'immagine reale: dopo un `Resize` a 512×512 con `INTER_LINEAR`,
il **7,4 %** dei pixel ha un colore **assente dalla palette originale**. Con
`INTER_NEAREST` sarebbe lo 0 %. Il caso peggiore è il bordo fra pixel valido e
pixel invalido `(0,0,0)`: l'invalidità è binaria, ma l'interpolazione la rende
continua e genera pixel che sembrano profondità valide mai misurate.

**Nessuna modifica.** Alla rete i tre canali arrivano come tre grandezze
correlate: non deve invertire alcuna colormap, quindi un colore interpolato è
per lei un blend liscio come in qualsiasi immagine RGB ridimensionata. Il
problema diventerebbe di primo ordine solo adottando l'inversione della
colormap descritta nella sezione 5: in quel caso andrebbe interpolato lo
scalare, non il colore, e servirebbe `INTER_NEAREST` sulla maschera di validità.

### Dataloader

Solo il loader di training mescola i campioni; validation e test li presentano
sempre nello stesso ordine, altrimenti i numeri per batch non sarebbero
confrontabili fra run. Il generatore è inizializzato con `RANDOM_SEED` e i
worker con `config.seed_worker`, così l'augmentation casuale è riproducibile.

`albumentations` è importato in modo pigro: non è installabile in locale, e
tutto tranne `build_transforms()` funziona senza. Passando `transform=None` la
classe `Dataset` normalizza e converte in tensori senza ridimensionare, il che
permette di verificarla localmente prima di portare il codice su Colab.

---

## 7. Loss e metriche (`losses.py`, `metrics.py`)

**Input:** logit del modello, label map, scheletro tubato precalcolato.
**Output:** valore di loss per l'ottimizzatore, metriche per epoca e per run.

Le formule sono state verificate direttamente sui tre articoli di riferimento,
depositati in `reference/`.

### 7.1 Formule di riferimento

**clDice** (Shit et al., CVPR 2021, eq. 1 e 2). Dati la maschera vera `V_L`, la
predetta `V_P` e i rispettivi scheletri `S_L` e `S_P`:

```
Tprec(S_P, V_L) = |S_P ∩ V_L| / |S_P|      precisione topologica
Tsens(S_L, V_P) = |S_L ∩ V_P| / |S_L|      sensibilità topologica
clDice          = 2 · Tprec · Tsens / (Tprec + Tsens)
```

La prima è sensibile ai falsi positivi, la seconda ai falsi negativi; clDice è
la loro media armonica. Gli scheletri sono **morfologici, a un pixel**.

**Skeleton Recall Loss** (Kirchhoff et al., ECCV 2024, eq. 1):

```
L_SkelRecall = −(1/|C|) · Σ_c [ Σ_i Y_skel,i,c · Ŷ_i,c ] / [ Σ_i Y_skel,i,c ]
```

dove `Y_skel` è lo scheletro **tubato** del ground truth e `Ŷ` è la
**probabilità predetta**. La combinazione con la loss principale è
`L = L_generic + w · L_SkelRecall`, con `L_generic = CE + Dice` e, negli
esperimenti dell'articolo, `w ∈ {0,1 ; 1,0}`.

I due lavori usano quindi **due scheletri diversi per due scopi diversi**: il
tubo per la loss, perché i pixel in più stabilizzano il segnale; quello esatto
per la metrica, perché è la definizione. L'articolo sulla SRL lo dice
esplicitamente in §4.2, dove usa clDice *as a metric*.

### 7.2 Difetti corretti nel codice originale

**`FocalLoss` non era mai stata eseguita.** Passava `weight=self.weights`,
attributo inesistente: `AttributeError` alla prima chiamata. Nella riscrittura
la probabilità della classe vera è letta dalla softmax invece che ricavata come
`exp(−CE)`: con i pesi di classe attivi quest'ultima non è una probabilità, e
`gamma` avrebbe agito con intensità diversa da una classe all'altra.

**`SkeletonRecallLoss` non produceva gradiente.** Ricavava le classi predette
con `torch.argmax`, funzione a gradino la cui derivata è nulla ovunque. Il
tensore risultante è staccato dal grafo — verificato: `requires_grad=False`,
nessun `grad_fn`, e `backward()` solleva un errore. Sommato alla loss
principale non falliva, ma il suo contributo al gradiente era **esattamente
zero**: la loss compariva nel valore stampato e non ha mai influito
sull'addestramento. Corretta usando la probabilità softmax, che è anche la
formulazione dell'articolo.

**Le metriche erano mediate sui batch.** Ogni batch produceva IoU, Dice e
accuratezza, e a fine epoca si applicava `np.nanmean` sull'elenco. Un batch con
50 pixel di una classe pesava quanto uno con 5 000, e i batch privi della
classe davano `NaN` ed erano esclusi, per cui la media risultava condizionata
ai soli batch in cui la classe compariva. Sostituite da una **matrice di
confusione** accumulata sullo split, da cui ogni metrica di sovrapposizione si
deriva alla fine.

**Le metriche sklearn erano ridondanti.** Su classificazione per pixel,
`f1_score(average='macro')` è identico alla media del Dice per classe, già
calcolata. Veniva però ricalcolato trasferendo circa 1,3 milioni di pixel per
batch da GPU a CPU. Rimosse: la matrice di confusione dà precisione e recall
senza costo aggiuntivo.

### 7.3 Scostamenti dai metodi di riferimento

Ogni scostamento è deliberato e misurato.

**Scheletrizzazione per classe invece che dell'unione.** L'Algoritmo 1
dell'articolo sulla SRL binarizza *tutto* il foreground insieme, scheletrizza
l'unione, dilata, e riassegna le classi moltiplicando per il ground truth. Il
progetto scheletrizza invece **ogni classe separatamente**.

*Motivazione.* Il loro schema è pensato per classi che sono segmenti adiacenti
di uno stesso albero (sottotipi di vaso), non per strutture che si occludono a
vicenda. Nel vigneto il tralcio passa **davanti** al cordone: l'unione dei due
è localmente una macchia spessa il cui asse mediano devia dall'asse del
cordone, e la successiva moltiplicazione per la label map assegna quei pixel a
`Cane`, che ha priorità. Lo scheletro del cordone si spezza a ogni incrocio.

*Verifica* sulle 100 immagini, applicando il loro algoritmo alla lettera:

| | per classe (nostro) | unione poi divisione |
|---|---|---|
| pixel di scheletro del cordone | 1 373 | 1 676 |
| **componenti connesse** | **9,13** | **13,97** |
| immagini con più frammenti | — | **93 / 100** |

L'unione produce più pixel ma il 53 % di frammenti in più, e peggiora in 93
immagini su 100.

**Kernel di dilatazione quadrato 2×2 invece del diamante di raggio 2.**
L'articolo dilata con un diamante di raggio 2, cioè 13 pixel.

*Motivazione.* Non è una questione di sconfinamento: entrambe le
implementazioni ritagliano il tubo sulla maschera — il progetto con
`tubed & binary_mask`, l'articolo con la moltiplicazione `Y_skel × Y`. Il
problema è la **copertura**:

| Kernel | Pixel | Copertura media di `Cane` | Massima | Immagini oltre l'80 % |
|---|---|---|---|---|
| quadrato 2×2 | 4 | 34,8 % | 56,6 % | 0 / 100 |
| quadrato 3×3 | 9 | 53,6 % | 83,7 % | 3 / 100 |
| diamante r2 | 13 | 65,7 % | 92,8 % | 7 / 100 |

Col diamante il tubo occupa due terzi del tralcio: smette di essere una linea
centrale e diventa la maschera, e la SRL degenera in un normale termine di
recall sulla classe. L'articolo può permetterselo perché vasi e strade sono
spessi rispetto alla risoluzione; i tralci di vite no.

**clDice sullo scheletro esatto.** Il codice originale passava a clDice lo
scheletro tubato, conflando i due usi. Corretto: la metrica usa lo scheletro
morfologico, estratto al momento della valutazione sia per la predizione sia
per il ground truth, così i due lati del confronto sono costruiti allo stesso
modo. Lo scheletro tubato su disco resta e alimenta la loss.

*Conseguenza sulla scelta del kernel.* L'analisi di copertura della sezione 2
nasceva dal timore che il tubo facesse degenerare clDice. Con la metrica
riportata alla sua definizione quel rischio non esiste più, e la copertura
resta rilevante solo per la loss — dove misura quanto il termine si allontani
dalla connettività verso il puro recall di area.

**Forma della loss: `1 − recall` invece di `−recall`.** Differiscono di una
costante additiva, quindi hanno **gradiente identico**. La forma adottata
mantiene il valore in `[0, 1]`, leggibile accanto alle altre loss.

**Media sulle sole classi presenti nel batch.** L'articolo divide per `|C|`
fisso, che è indefinito quando una classe non compare. Le classi assenti sono
escluse dalla media invece di essere contate come recall nullo.

**Scala della combinazione.** L'articolo somma `CE + Dice` con coefficiente 1
ciascuno, scala circa 2; la `CombinedLoss` del progetto usa `0,7·CE + 0,3·Dice`,
combinazione convessa di scala circa 1. Un peso `w` preso dall'articolo va
quindi **dimezzato** per avere lo stesso effetto relativo:

| peso nell'articolo | equivalente qui |
|---|---|
| 0,1 (testato dagli autori) | 0,05 |
| 1,0 (testato dagli autori) | **0,5** |

**Valore adottato: `weight = 0,5`.** Il valore precedente, 0,2, corrispondeva a
circa 0,4 nell'articolo: un punto intermedio che non coincide con nessuna delle
due configurazioni pubblicate e che sarebbe stato difendibile solo come
taratura propria.

La scelta del più forte dei due non è un giudizio su quale funzioni meglio, ma
discende dallo scopo dell'ablazione, che è **falsificare la previsione della
sezione 7.6** e non cercare il valore ottimale. A 0,05 un risultato nullo
sarebbe ambiguo — indistinguibile fra «l'effetto previsto non c'è» e «il
termine era applicato troppo debolmente» — mentre a 0,5 un risultato nullo è
esso stesso un'informazione, perché varrebbe alla forza che gli autori hanno
pubblicato. Cercare il valore ottimale richiederebbe inoltre una serie di run
di taratura che il tempo disponibile non consente, e che comunque risponderebbe
a una domanda diversa da quella posta qui.

### 7.4 Cosa fa davvero la Skeleton Recall Loss

Arora et al. (arXiv:2508.11374) contestano i risultati dell'articolo originale
con un'analisi dei gradienti, mostrando che il gradiente della SRL rispetto
alla predizione è **costante**, e concludendo che *"a constant gradient keeps
pushing the parameters further away from the already achieved optimal values"*.

**L'osservazione è corretta, la conclusione no.** Il gradiente è costante
rispetto alla *probabilità*, ma le reti si ottimizzano attraverso i *logit*, e
attraverso la jacobiana della softmax diventa `−p(1−p)`:

| p(classe) | ∂L/∂p | ∂L/∂logit |
|---|---|---|
| 0,018 | −1,00 | −0,018 |
| 0,500 | −1,00 | −0,250 |
| 0,982 | −1,00 | −0,018 |
| 0,998 | −1,00 | −0,002 |

La loss **smette da sola di spingere** quando il modello è già sicuro:
l'obiezione principale non regge al livello a cui avviene l'ottimizzazione.

La stessa curva rivela però un limite che gli autori non nominano, e che è più
serio: il gradiente si annulla **anche a `p → 0`**, cioè sui pixel di scheletro
che il modello sbaglia con sicurezza — proprio quelli che la loss dovrebbe
recuperare. La cross-entropy lì ha gradiente massimo (`p − 1`). La Skeleton
Recall Loss è quindi un **affinatore attorno alla soglia decisionale**, non un
recuperatore di strutture perse.

**Resta valido** il loro rilievo che la SRL non penalizza i falsi positivi: la
loss vive solo sui pixel di scheletro, altrove il gradiente è nullo. È un
termine di puro recall senza contraltare di precisione, per costruzione.

**Restano validi i loro risultati sperimentali**: su dataset tubulari, 3
punteggi significativamente migliori, 3 significativamente peggiori e 6
marginali su 15; su dataset non tubulari, 1 migliore contro 7 peggiori.

### 7.5 clDice non vede il gonfiamento

Verificato su geometrie sintetiche, con un tralcio orizzontale di 4 pixel di
spessore:

| Predizione | IoU | Precisione | clDice |
|---|---|---|---|
| perfetta | 1,000 | 1,000 | 1,000 |
| **gonfiata** (spessore 4 → 10) | **0,400** | **0,400** | **1,000** |
| interrotta (buco di 4 px su 50) | 0,920 | 1,000 | 0,956 |
| gonfiata e interrotta | 0,387 | 0,400 | 0,956 |

**clDice è cieco al gonfiamento del raggio.** Non è un difetto: l'articolo
originale dichiara esplicitamente che un errore di raggio è tollerabile mentre
una rottura di connettività no. Le due famiglie di metriche misurano proprietà
ortogonali, e vanno riportate insieme.

*Rilevanza per questo progetto.* Il modello mostra già una tendenza a
sovrasegmentare i tralci rispetto all'annotazione. Quel difetto si manifesta su
IoU e precisione e **non comparirà in clDice**.

**Ed è proprio questa la ragione per cui clDice serve.** Le due metriche vanno
riportate e lette **insieme**: nessuna delle due, da sola, distingue un errore
di *topologia* da un errore di *raggio*.

Un IoU basso accompagnato da un clDice alto non è un risultato contraddittorio
né un fallimento: dice che il modello ha **individuato correttamente la
struttura** e sbaglia sui bordi. È un'informazione diagnostica, non una
scusante — ma permette di non scartare una configurazione che sul compito reale
funziona.

*Perché l'IoU ha un tetto su questo problema.* Tre fattori lo comprimono
indipendentemente dalla qualità del modello:

- le strutture sono **sottili e numerose**, quindi la frazione di pixel di
  bordo sul totale della classe è alta: uno scarto di un pixel su un tralcio
  largo dieci costa il 10 % di IoU, contro il 2 % su una struttura larga
  cinquanta
- le **occlusioni** frammentano ciò che è visibile, e la label map registra
  solo la parte visibile
- l'**annotazione manuale è essa stessa incerta sui bordi**: collocare il
  contorno di un tralcio sottile a livello di pixel è difficile anche per un
  operatore umano, come conferma la differenza di precisione dei margini fra le
  due convenzioni di annotazione documentata nella sezione 1

Il terzo punto è il più rilevante da dichiarare in tesi: parte dello scarto di
IoU non misura l'errore del modello ma il **rumore dell'etichetta**. clDice, che
dipende dalla linea centrale e non dal contorno, è molto meno sensibile a
quell'incertezza, ed è quindi la metrica su cui il confronto fra
configurazioni resta più stabile.

*Rilevanza per l'applicazione.* L'obiettivo finale è la potatura automatica: un
tralcio non riconosciuto porta a un taglio mancato o sbagliato, un tralcio con
margini leggermente gonfiati è un'imprecisione geometrica assorbibile a valle.
Ne segue che l'ipotesi della sezione 7.6 — se confermata — descriverebbe un
compromesso **accettabile** per questo caso d'uso, non un peggioramento.

*Conseguenza sul confronto fra run.* `MONITOR_METRIC` è `mean_iou_no_bg`. La
scelta va mantenuta o cambiata consapevolmente, non per inerzia: la tabella
comparativa finale deve riportare IoU, clDice, precisione e recall per classe,
e il criterio con cui le run vengono ordinate va dichiarato esplicitamente.

### 7.6 Previsione da verificare sperimentalmente

Mettendo insieme 7.4 e 7.5 si ottiene un'ipotesi falsificabile: poiché la SRL
premia il recall senza penalizzare i falsi positivi, e il modello già gonfia i
tralci, ci si attende che attivarla **peggiori la precisione su `Cane` e
`Cordon` lasciando stabili o in miglioramento IoU e clDice**.

L'esperimento è un'ablazione con e senza il termine, che `SKELETON_LOSS_CONFIG`
già consente con un flag: due run identiche in tutto il resto. Guardare la sola
mean IoU, o il solo clDice, non basterebbe a rilevarlo — per questo
`ConfusionMatrix` riporta precisione e recall per classe, e `plot_history` li
mette in grafico accanto alla loss.

Confermare o smentire l'ipotesi sui propri dati vale più di una citazione: una
critica pubblicata alla loss adottata esiste, e va affrontata con una misura.

### 7.7 Esito dell'ablazione: la previsione era sbagliata sul piano giusto

Tre seed per braccio (42, 123, 456), `UNet` + `ce`, **baseline senza pesi di
classe** — la configurazione adottata in sezione 12 — budget 70, tutto il resto
invariato. La SRL con `weight = 0.5`, kernel 2, `skeleton_classes = [1, 2]`.
Test t di Welch.

**Metriche di area:**

| metrica | senza SRL | con SRL | differenza | p |
|---|---|---|---|---|
| IoU media | 0,6674 ± 0,0075 | 0,6243 ± 0,0089 | **−0,0431** | **0,003** |
| precisione | 0,8500 ± 0,0132 | 0,8359 ± 0,0148 | −0,0141 | 0,286 |
| recall | 0,8120 ± 0,0030 | 0,7765 ± 0,0176 | −0,0356 | 0,069 |
| clDice medio | 0,9087 ± 0,0056 | 0,9012 ± 0,0035 | −0,0075 | 0,135 |
| IoU `Cane` | 0,7481 ± 0,0067 | 0,7358 ± 0,0047 | −0,0122 | 0,068 |
| IoU `Cordon` | 0,8009 ± 0,0068 | 0,7863 ± 0,0022 | −0,0146 | 0,054 |
| IoU `Node` | 0,4356 ± 0,0147 | 0,3370 ± 0,0584 | −0,0986 | 0,092 |
| IoU `Other` | 0,6849 ± 0,0070 | 0,6380 ± 0,0198 | **−0,0469** | **0,042** |

**Metriche topologiche — dove la loss opera per costruzione:**

| metrica | senza SRL | con SRL | differenza | p |
|---|---|---|---|---|
| top. **recall** `Cane` | 0,9212 ± 0,0025 | **0,9469 ± 0,0055** | **+0,0257** | **0,007** |
| top. **precision** `Cane` | 0,9325 ± 0,0062 | 0,9027 ± 0,0055 | **−0,0298** | **0,003** |
| top. precision `Cordon` | 0,9065 ± 0,0021 | 0,8966 ± 0,0102 | −0,0100 | 0,228 |
| top. recall `Cordon` | 0,8755 ± 0,0187 | 0,8607 ± 0,0021 | −0,0148 | 0,304 |

**La previsione della sezione 7.6 va corretta su due punti.**

Prevedeva uno scambio precisione↔recall **sulle metriche di area**: precisione
in calo, recall stabile o in crescita. Nessuna delle due cose accade. La
precisione di area **non si muove in modo significativo** (p = 0,29) e il recall
di area **scende** invece di salire. Sull'area non c'è alcuno scambio: c'è un
peggioramento uniforme, guidato dalla IoU media che cala di 0,043 con p = 0,003.

Lo scambio previsto esiste, ma **sulle metriche topologiche**, ed è il risultato
più netto dell'intera ablazione: recall topologico su `Cane` +0,026 e precisione
topologica −0,030, entrambi con p < 0,01. Sono le due misure con i valori p più
bassi di tutta la tabella.

**La lettura corretta è quindi:** la Skeleton Recall Loss fa esattamente ciò che
dichiara, sulla struttura che prende di mira — recupera linea centrale al prezzo
della precisione della linea centrale — **e le metriche di area pagano il conto
senza ricevere nulla in cambio.** Su questo dataset non è un compromesso da
scegliere in base alle priorità: è un costo netto sulla segmentazione di area a
fronte di un guadagno confinato alla topologia di una sola classe.

*Il caso `Node`.* È la classe che perde di più (−0,099) pur **non essendo un
bersaglio della loss**: `skeleton_classes = [1, 2]` copre `Cane` e `Cordon`. Il
meccanismo plausibile è quello misurato in sezione 5 — il 54 % dei pixel di
`Node` giace sul cordone — quindi una loss che spinge massa di probabilità lungo
la linea centrale del cordone assorbe i nodi che le stanno sopra. La SRL rende
inoltre `Node` molto più instabile: σ passa da 0,015 a 0,058, quasi quattro
volte, ed è per questo che una differenza di quasi dieci punti resta a p = 0,09.

*Sulla critica di Arora et al.* La sezione 7.4 aveva stabilito che l'obiezione
sul gradiente costante non regge a livello dei logit, ma che l'asimmetria
recall/precisione della loss è reale. Questi dati confermano l'asimmetria e ne
precisano il luogo: si manifesta sulla topologia, non sull'area, e su questo
dataset il saldo complessivo è negativo. Non è una conferma della critica come
formulata, ed è un risultato indipendente sia dall'articolo originale sia dalla
sua confutazione.

---

## 8. Modelli

### Addestramento da zero, senza encoder pre-addestrati

Le architetture sono usate con encoder inizializzati casualmente
(`ENCODER = "custom"`, `ENCODER_WEIGHTS = None`), escludendo i backbone
pre-addestrati su ImageNet.

*Motivazione.* Un encoder pre-addestrato si aspetta tre canali RGB di immagini
naturali, mentre l'ingresso qui è a cinque canali di natura eterogenea:
intensità ToF, profondità colorata e affidabilità. Adattarlo richiederebbe di
alterare il primo strato convoluzionale — replicando o mediando i pesi sui
canali aggiuntivi — con il risultato che i pesi pre-addestrati risulterebbero
comunque non pertinenti sui canali nuovi. Il vantaggio del transfer learning si
ridurrebbe alla sola intensità, che è l'unico canale assimilabile a
un'immagine naturale.

*Conseguenza sulla normalizzazione.* `USE_IMAGENET_NORM_FOR_RGB = False`: tutti
i canali usano le statistiche del dataset misurate in sezione 5. La modalità
mista resta implementata in `dataset.py` ma non viene usata, ed esiste solo per
il caso in cui si volesse riconsiderare la scelta.

*Conseguenza sulle dipendenze.* Non serve `segmentation_models_pytorch`: quella
libreria dà valore soprattutto attraverso i backbone pre-addestrati, che qui
non si usano. Le tre architetture sono implementate in PyTorch puro, quindi
sono verificabili in locale senza GPU e senza installare nulla.

### Le tre architetture messe a confronto

`UNetPlusPlus` compare in `config.ARCHITECTURE` ma non è implementata: il
confronto è fra tre architetture che rispondono in modo diverso alla stessa
domanda, non fra quattro varianti della stessa idea.

| | Larghezza | Parametri | Come propaga il dettaglio |
|---|---|---|---|
| U-Net | base 32 | 7,76 M | connessioni skip: il decoder riceve le mappe a piena risoluzione |
| SegNet | base 64 | 7,09 M | solo gli **indici** di pooling, non le mappe |
| PSPNet | base 48 | 8,02 M | contesto multi-scala, risoluzione fermata a H/8 |

**U-Net** è la scelta abituale per strutture sottili: le connessioni skip
portano al decoder le feature che l'encoder aveva prima del pooling, quindi un
tralcio largo pochi pixel non va ricostruito da una mappa in cui è ridotto a un
punto.

**SegNet** trasporta solo la posizione in cui ciascun max-pooling ha scelto il
suo valore, e la riusa in fase di unpooling. Costa un quarto dei parametri di
U-Net, ma il decoder deve ricostruire il contenuto da una mappa sparsa. È
architettura consolidata in questo dominio — usata per la segmentazione di
tronchi, rami e fili di sostegno in meleto e in viticoltura — e il confronto
serve a stabilire se gli indici da soli bastino su strutture di pochi pixel.

**PSPNet** aggrega contesto a più scale, utile quando una classe è definita dal
suo ruolo nella scena più che dall'aspetto locale: un cordone si distingue da un
tralcio soprattutto per posizione e continuità.

*Limite noto di PSPNet su questo problema.* Il classificatore lavora a **H/8** e
il risultato viene riportato a piena risoluzione con una singola interpolazione
bilineare. A 512×512 la mappa più fine è 64×64: un tralcio largo otto pixel vi
occupa **un solo pixel**, e il suo contorno non è predetto ma interpolato. È
prevedibile che PSPNet collochi le strutture correttamente e sia la meno precisa
delle tre sui bordi — cioè esattamente la distinzione che clDice e IoU insieme
sono state scelte per separare (sezione 7.5). Se il confronto lo confermasse,
sarebbe una conferma indipendente della scelta delle metriche.

### Capacità pareggiata fra le architetture

Alle loro larghezze naturali le tre architetture **non sono confrontabili**. Il
riscontro sui lavori di riferimento lo conferma: SegNet in Majeed et al. usa
VGG16 con 13 layer convoluzionali e 5 blocchi encoder, PSPNet in Chen et al. un
backbone ResNet o VGG. Riprodotte così, e con U-Net alla sua dimensione
canonica, si otterrebbe:

| | Nostra implementazione naturale | In letteratura |
|---|---|---|
| U-Net | 31,0 M | ~31 M (architettura originale) |
| SegNet | 7,1 M | ~29,5 M (VGG16, 5 blocchi, pre-addestrato) |
| PSPNet | 14,2 M | ~46–65 M (backbone ResNet50) |

*Il problema.* Un confronto fra un U-Net a grandezza naturale e versioni ridotte
delle altre due **confonde architettura e capacità**: se U-Net vincesse, non si
potrebbe dire se il merito è delle connessioni skip o del fatto che ha quattro
volte i parametri di SegNet.

*Scelta adottata: pareggiare verso il basso.* `config.MODEL_BASE_CHANNELS`
fissa la larghezza di ciascuna architettura in modo che i parametri siano
comparabili — 7,76, 7,09 e 8,02 milioni, uno scarto del 13 %.

Tre ragioni, in ordine di peso:

1. **Rende il confronto interpretabile.** È l'unico modo per attribuire una
   differenza di risultato all'architettura invece che alla dimensione.
2. **Costa meno.** Misurato su forward più backward: 2 457 ms per iterazione
   sommando le tre run pareggiate, contro 4 283 ms alle larghezze naturali.
   Pareggiare *verso l'alto* sarebbe costato circa 7 650 ms, e aggiungere una
   sola run di controllo a capacità ridotta 4 969 ms su quattro run. Il
   pareggio verso il basso è la strada metodologicamente pulita **e** la più
   economica.
3. **È più adatto ai dati.** Il training set conta 280 campioni aumentati da 70
   immagini originali. Trentuno milioni di parametri su 280 campioni sono
   molti, e il rischio di overfitting è concreto a prescindere dal confronto.

*Conseguenza da dichiarare.* I risultati di SegNet non saranno confrontabili
con quelli di Majeed et al.: oltre a essere più stretto, il nostro è addestrato
da zero mentre il loro parte da VGG16 pre-addestrato. Il confronto interno fra
le tre architetture resta valido; quello con i numeri pubblicati no.

*Due esperimenti distinti.* Il pareggio di capacità serve al **confronto fra
architetture** e non pretende di produrre il modello migliore in assoluto. Sono
due domande diverse:

| Esperimento | Configurazione | Risponde a |
|---|---|---|
| Confronto | tre architetture a capacità pareggiata | quale architettura è più adatta al problema |
| Configurazione migliore | architettura vincente, larghezza scelta per resa | qual è il massimo ottenibile |

Il secondo costa una run aggiuntiva e non è scontato che la larghezza maggiore
vinca: su 280 campioni di training un modello da 31 milioni di parametri ha un
rischio di overfitting concreto.

*Nota sui risultati del notebook monolitico.* I valori ottenuti prima del
refactoring **non costituiscono un riferimento**, indipendentemente dalla
dimensione del modello: erano su una partizione diversa (93 immagini, impronta
`d57f79eb`), con metriche mediate sui batch anziché accumulate sul dataset —
quindi lo stesso identico modello riporterebbe oggi numeri diversi — con 24
immagini a canali incoerenti (sezione 4) e con una skeleton loss che non
contribuiva al gradiente (sezione 7.2).

*Conseguenza sulla tracciabilità.* `experiments.csv` deve registrare
`base_channels` accanto all'architettura: due run della stessa architettura a
larghezze diverse sarebbero altrimenti indistinguibili.

### Un difetto trovato durante la misura

Il benchmark ha fatto emergere che **PSPNet fallisce in modalità training con
batch di dimensione 1**. Il livello 1×1 della piramide riduce la mappa a un solo
pixel, e la batch normalization non può calcolare statistiche da un unico valore
per canale.

Con 280 campioni e batch 4 il resto è zero, quindi il caso non si presenta oggi;
si presenterebbe cambiando la dimensione del batch o del dataset — cioè proprio
quando il dataset verrà ampliato a ~100 immagini. Il dataloader di training ora
scarta l'ultimo batch **solo quando conterrebbe esattamente un campione**, così
non si sprecano dati negli altri casi. Validation e test non sono toccati:
in modalità di valutazione la batch normalization usa le statistiche correnti.

### Verifica prima della GPU

`test_model()` esegue un forward **e un backward** su un tensore fittizio delle
dimensioni configurate, e riporta il picco di memoria quando gira su CUDA. Il
backward è incluso di proposito: è lì che la memoria raggiunge il massimo, e un
modello che fa forward senza problemi può comunque esaurire la GPU quando
vengono allocati i gradienti. Uno scarto di forma trovato in locale costa
secondi, lo stesso scarto trovato su Colab costa una sessione.

---

## 9. Training e registro degli esperimenti (`training.py`)

**Input:** dataloader, modello, loss.
**Output:** una cartella per run sotto `experiments/`, una riga in
`results/experiments.csv`.

### Scelte

**Un solo corpo per training e valutazione.** `run_epoch()` distingue le due
fasi dalla presenza dell'ottimizzatore. Tenerle separate significherebbe due
implementazioni che possono divergere su *cosa* misurano, ed è esattamente il
tipo di divergenza che non produce errori ma rende i numeri incomparabili.

**Ottimizzatore e scheduler salvati nel checkpoint.** Riprendere una run senza
di loro riparte con momento azzerato e schedule reinizializzata: non è la
stessa run. Vengono salvati due checkpoint — il migliore secondo la metrica
monitorata e l'ultimo — perché servono a due scopi diversi: il primo per
valutare, il secondo per riprendere.

**History e checkpoint scritti su cadenze diverse.** Costano tre ordini di
grandezza l'uno dall'altro, e servono a cose diverse:

| Artefatto | Dimensione | Cadenza | A cosa serve |
|---|---|---|---|
| `history.json` | ~8 KB | **ogni epoca** | le metriche di una run interrotta sopravvivono in forma leggibile |
| `last.pt` | ~83 MB | ogni `CHECKPOINT_INTERVAL` epoche e alla fine | il punto da cui il training riprende |
| `best.pt` | ~83 MB | a ogni miglioramento | il modello con cui si valuta |

*Motivazione.* Scrivere `last.pt` a ogni epoca costerebbe 5,8 GB su 70 epoche,
in scritture sincrone dentro il ciclo — costo sensibile su Drive. Con
`CHECKPOINT_INTERVAL = 10` scende a 580 MB, e il prezzo è ripetere al massimo
dieci epoche dopo un'interruzione. La history invece pesa pochi kilobyte e
viene riscritta sempre: perdere i numeri di sessanta epoche perché la sessione è
caduta all'epoca sessantuno sarebbe molto peggio che ripeterne dieci.

`last.pt` viene scritto **anche all'ultima epoca**, a prescindere dall'intervallo,
così lo stato finale non è mai un multiplo stantio della cadenza.

**Ripresa di una run interrotta.** `find_resume_point(run_id)` restituisce il
checkpoint se esiste, e `train_model(..., resume_from=...)` ripristina pesi,
stato dell'ottimizzatore e dello scheduler, history, valore migliore e contatore
delle epoche. `num_epochs` resta contato **dall'inizio della run**, non dal punto
di ripresa: riprendere una run interrotta a 31 con `num_epochs=70` addestra le
epoche da 32 a 70.

*Verificato:* una run interrotta a 3 epoche e ripresa fino a 6 produce una
history identica, valore per valore, a una run continua di 6 epoche. Le curve
sono le stesse che si otterrebbero se non si fosse mai fermata.

*Il `config.json` non viene sovrascritto dalla ripresa.* Descrive le
impostazioni che le epoche precedenti hanno effettivamente usato; riscriverlo
con quelle di oggi — commit compreso — renderebbe la run non interpretabile.
L'evento di ripresa è invece registrato in `history.json` come
`resumed_from_epoch`.

**L'architettura è registrata nel checkpoint e verificata al caricamento.**
Caricare i pesi di SegNet dentro un PSPNet produrrebbe un errore oscuro o, in
casi sfortunati, un caricamento parziale silenzioso.

**clDice calcolato sulla validation a ogni epoca.** Richiede le predizioni
dell'intera passata in memoria e la scheletrizzazione su CPU, quindi non è
gratis, ma su 14 immagini costa poco e dà la curva. Sul training non viene
calcolato: 256 predizioni a 512×512 in memoria sono mezzo gigabyte.

**Precisione e recall stampati a ogni epoca** accanto alla metrica monitorata,
per la ragione della sezione 7.6: un termine che premia il recall senza
penalizzare i falsi positivi li fa divergere, e la sola IoU lo nasconderebbe.

**Budget fisso di 70 epoche, non convergenza.** Nella prima run definitiva
(`UNet`, `ce`, seed 42) la mean IoU di validazione era ancora in lieve salita
all'epoca 70 e l'early stopping con `patience = 20` non è mai scattato. Si è
comunque scelto di **non allungare il budget**, per tre ragioni misurate sulla
run stessa:

- la **val loss ha smesso di scendere intorno all'epoca 50** (0,199) e da lì
  oscilla fra 0,20 e 0,22, mentre la train loss continua a calare da 0,194 a
  0,146: il divario di generalizzazione si sta aprendo, quindi la salita
  residua della IoU è raccolta di briciole in regime di sovradattamento
- il guadagno è in decelerazione: +0,04 di IoU in ventisette epoche, fra la 42
  e la 69
- con 23 immagini di validazione, oscillazioni di ±0,02 fra epoche consecutive
  sono nell'ordine del rumore

Il confronto è quindi **a budget pareggiato**: tutte le architetture ricevono
70 epoche, e questo va dichiarato come tale invece di essere presentato come
convergenza.

*Il rischio del budget pareggiato, e come viene controllato.* Un budget fisso
penalizza un'architettura che converga più lentamente, confondendo lentezza e
qualità. Il controllo è la **pendenza della mean IoU di validazione sulle
ultime dieci epoche**, dato già presente in `history.json` — ma va letto
contro l'inviluppo misurato sotto, non in assoluto.

### Rumore fra run identiche — misurato

Tre run identiche in tutto tranne `RANDOM_SEED` (42, 123, 456), su
`UNet` + `ce` + AdamW + ReduceLROnPlateau, 5 canali, budget 70 epoche. Lo split
è letto da disco, quindi il seed tocca solo inizializzazione dei pesi e ordine
dell'augmentation: la partizione resta `62de0ec4` in tutte e tre.

Tutta la variazione fra queste run è rumore per costruzione. È il metro con cui
va letta ogni differenza riportata altrove.

*Nota storica.* Una prima misura era viziata: `set_seed` aveva il default
`seed: int = RANDOM_SEED`, valutato all'import, quindi `config.set_seed()`
rimetteva 42 qualunque valore fosse stato assegnato un istante prima. Le tre
run condividevano l'inizializzazione dei pesi e variavano solo per l'ordine dei
campioni. Sono state scartate e rifatte dopo la correzione; i valori qui sotto
vengono dalle run rifatte, tutte con impronta del codice `195c6b62`.

Tutte le grandezze sono misurate **sul test**, dal checkpoint migliore.

| grandezza | media | dev. std | escursione | soglia 2σ fra due run singole |
|---|---|---|---|---|
| IoU media (no bg) | 0,6442 | 0,0076 | 0,0151 | **0,021** |
| best val IoU | 0,6428 | 0,0065 | 0,0120 | 0,018 |
| clDice medio | 0,8996 | 0,0053 | 0,0102 | 0,015 |
| precisione media | 0,7719 | 0,0051 | 0,0091 | 0,015 |
| recall media | 0,8708 | 0,0065 | 0,0124 | 0,018 |
| IoU `Cane` | 0,7167 | 0,0078 | 0,0142 | 0,022 |
| IoU `Cordon` | 0,7760 | 0,0074 | 0,0143 | 0,021 |
| IoU `Node` | 0,4556 | **0,0175** | 0,0306 | **0,049** |
| IoU `Other` | 0,6286 | 0,0018 | 0,0035 | 0,005 |
| clDice `Cane` | 0,9155 | 0,0050 | 0,0088 | 0,014 |
| clDice `Cordon` | 0,8837 | 0,0059 | 0,0118 | 0,017 |

**La griglia a seed singolo è uno strumento di screening, non di
certificazione.** Due configurazioni che differiscono di meno di ~0,02 di test
IoU sono indistinguibili e vanno riportate come tali. Con differenze fra
architetture plausibilmente in quell'ordine, è un esito da mettere in conto.

**`Node` è di gran lunga la classe più instabile:** σ = 0,0175 contro 0,0076
della media, e una soglia di 0,049. Metà del suo intervallo di IoU è rumore.
Qualunque conclusione su `Node` tratta da run singole — inclusa la decisione di
eliminarlo, questione 7 — è priva di fondamento se la differenza in gioco è
inferiore a cinque punti di IoU. `Other`, all'estremo opposto, è la più stabile
di tutte con σ = 0,0018.

**Sull'incertezza di queste stime.** Sono deviazioni standard da tre campioni:
l'intervallo di confidenza al 95 % di una σ stimata con 2 gradi di libertà va
da 0,52·σ̂ a 6,3·σ̂. Le soglie sono quindi stime puntuali con incertezza ampia,
utilizzabili come regola pratica ma non come test statistico. Ne discende che
per l'affermazione più delicata — l'ablazione della sezione 7.6 — conviene
comunque **tre seed per braccio**: confrontare due medie di tre è molto più
solido che confrontare due estrazioni singole, indipendentemente da quale
componente domini il rumore.

*Rettifica.* La misura viziata suggeriva che la precisione fosse tre volte più
rumorosa del recall (0,0114 contro 0,0035), e su questo era stata costruita la
motivazione dei tre seed per l'ablazione. La misura completa rovescia il
rapporto: 0,0051 contro 0,0065. La raccomandazione resta, ma per la ragione
esposta sopra, non per un eccesso di rumore sulla precisione.

*Criterio sulle epoche — inviluppo misurato.* La pendenza della val IoU di fine
training varia col seed, quindi va letta contro il proprio inviluppo e non in
assoluto. Sulle tre run, regressione lineare sulle ultime dieci epoche:

| run | pendenza ×10⁻³ per epoca | IoU guadagnata fra ep. 50 e 70 |
|---|---|---|
| seed 42 | +1,32 | +0,0151 |
| seed 123 | +2,77 | +0,0425 |
| seed 456 | +2,85 | +0,0408 |

**Inviluppo `UNet`: da +1,3 a +2,9.** Un'architettura che esca sensibilmente
sopra questo intervallo è penalizzata dal budget, non dalla propria qualità, e
per quella si alza il budget dichiarandolo.

*Il budget è un troncamento, e va detto senza attenuazioni.* Le epoche migliori
sono 59, 70 e 69, e il guadagno fra l'epoca 50 e la 70 arriva a +0,04, cioè al
doppio della soglia di rumore. Nel frattempo la train loss continua a scendere
(da −0,016 a −0,046 nello stesso intervallo) mentre la val loss è piatta: il
modello affina il confine di decisione oltre il punto in cui la loss smette di
migliorare. Non è convergenza, ed è scorretto presentarla come tale.

Se il tempo lo consente a fine campagna, **una singola run a 150 epoche sulla
configurazione migliore** permette di quantificare ciò che il budget lascia sul
piatto, invece di tacerlo.

**La direzione dello scheduler è derivata, non dichiarata.** `ReduceLROnPlateau`
viene avanzato con `MONITOR_METRIC`, quindi la sua direzione **deve** essere
`MONITOR_MODE`; `create_scheduler()` la legge da lì e `SCHEDULER_CONFIG` non ne
dichiara una propria. Vedere la sezione 11: due dichiarazioni indipendenti della
stessa grandezza sono esattamente ciò che le ha fatte divergere.

### Colonne di `experiments.csv`

La riga registra **come** una run era configurata oltre a **cosa** ha
prodotto: una metrica senza le impostazioni che l'hanno generata non è
confrontabile con niente. Sessantasei colonne, raggruppate così:

| Gruppo | Contenuto |
|---|---|
| Identificazione | `run_id`, timestamp, `git_commit`, `splits_hash` |
| Modello | architettura, `base_channels`, parametri, canali di ingresso, dimensione immagine |
| Loss | tipo, pesi di CE e Dice, metodo dei pesi di classe, skeleton loss attiva, peso, kernel |
| Ottimizzazione | ottimizzatore, learning rate, weight decay, scheduler, batch size |
| Andamento | epoche eseguite, epoca migliore, metrica monitorata e suo valore, secondi per epoca |
| Test | IoU, Dice, precisione, recall e clDice — media e **per classe** |

Tre colonne sono state aggiunte rispetto al notebook monolitico, ciascuna per
un motivo emerso durante il refactoring:

- **`base_channels`**, perché due run della stessa architettura a larghezze
  diverse sarebbero altrimenti indistinguibili (sezione 8)
- **`git_commit`**, con marcatura `-dirty` se ci sono modifiche non
  committate, perché il codice è cambiato molto e cambierà ancora: una metrica
  è interpretabile solo accanto alla versione che l'ha prodotta
- **precisione e recall per classe**, perché sono il sintomo con cui si valuta
  la skeleton loss

**Lo schema può crescere.** Il file viene riletto, concatenato e riscritto, così
una run che registra una metrica che le precedenti non avevano lascia le celle
vuote invece di fallire o di far sparire la colonna.

*Accuratezza per classe non riportata.* Su un problema multiclasse per pixel
coincide con il recall della stessa classe, quindi riportarla significherebbe
cinque colonne duplicate.

---

## 10. Valutazione (`evaluation.py`)

**Input:** cartelle delle run, `experiments.csv`.
**Output:** figure per la tesi, tabella comparativa.

### Scelte

**Ogni run viene ricostruita dal proprio `config.json`**, non dalla
configurazione corrente. Il senso di confrontare più run è che differiscono:
leggere `config.ARCHITECTURE` al momento del caricamento ricostruirebbe la rete
sbagliata. `load_checkpoint()` verifica comunque che l'architettura del
checkpoint coincida con quella del modello ricevuto.

**Le run misurate su una partizione diversa sono nascoste dalla tabella.**
Metriche prodotte su split diversi non sono confrontabili, e mescolarle in
un'unica tabella porta esattamente alla conclusione sbagliata. Il campo
`splits_hash` registrato in ogni riga è ciò che rende possibile il controllo.

### La mappa degli errori

I punteggi di sovrapposizione dicono **quanto** una predizione sbaglia, non
**come**. La mappa colora ogni pixel secondo il tipo di errore:

| Colore | Significato |
|---|---|
| grigio | struttura trovata e classificata correttamente |
| **rosso** | struttura predetta dove non c'è — falso positivo |
| **blu** | struttura presente e non trovata |
| giallo | struttura trovata ma assegnata a un'altra classe |

*Perché conta qui.* Distinguere il rosso dal blu separa due modi di fallire con
lo stesso IoU e conseguenze molto diverse per la potatura: un tralcio gonfiato
è un'imprecisione geometrica, un tralcio mancato è un taglio sbagliato. La
firma attesa della sovrasegmentazione già osservata è un **alone rosso che
segue le strutture vere**, non macchie rosse lontane da esse — cioè bordi
dilatati, non tralci inventati.

**I casi peggiori accanto ai campioni medi.** `worst_samples()` ordina per IoU
per immagine: la media nasconde le debolezze sistematiche, i fallimenti le
mostrano, e una figura di tesi dovrebbe essere onesta su questo.

### Confronto per classe

La media sulle classi nasconde quelle che contano: `Background` è l'84 % dei
pixel ed è facile, `Node` è lo 0,27 % ed è difficile. Una run può guidare la
media perdendo entrambe le classi sottili che sono l'oggetto del problema.

`plot_class_comparison()` traccia una metrica per classe con una barra per run.
Precisione e recall vanno lette **in coppia**: se la Skeleton Recall Loss si
comporta come previsto nella sezione 7.6, l'effetto compare come precisione in
calo su `Cane` e `Cordon` con recall stabile o in crescita.

*Ordinamento della tabella.* Il criterio va **dichiarato**, non assunto.
Ordinare per mean IoU è una scelta fra le tante e per la potatura non è
ovviamente quella giusta. Il notebook mostra di proposito la stessa tabella
ordinata per IoU e per clDice: se l'ordine cambia, le due metriche non
concordano su quale run sia migliore, e questo è un risultato da discutere, non
un problema da nascondere.

---

## 11. Verifica della pipeline prima degli esperimenti

Prima di produrre risultati destinati alla tesi l'intera pipeline è stata
sottoposta a un controllo sistematico: lettura integrale dei nove moduli,
riesecuzione di **ogni misura citata in queste note**, e verifica funzionale su
dati reali. Da qui in avanti una modifica al codice invalida le run già
eseguite, quindi il controllo è stato fatto prima e non dopo.

Il controllo è stato ripetuto integralmente sul **dataset definitivo a 100
immagini**, dopo la revisione delle annotazioni: tutti i valori riportati in
queste note provengono da quella esecuzione.

### Difetti trovati e corretti

**Direzione dello scheduler opposta alla metrica monitorata — critico.**
`SCHEDULER_CONFIG` dichiarava `mode: "min"` mentre `MONITOR_MODE` è `"max"`, e
il ciclo di training avanza `ReduceLROnPlateau` con `mean_iou_no_bg`. Lo
scheduler cercava quindi una **diminuzione** in una quantità che cresce:
simulando una IoU in miglioramento costante, il learning rate veniva dimezzato
ogni tre epoche, da 1e-4 a 1,25e-5 in dieci epoche, e avrebbe raggiunto
`min_lr` intorno all'epoca 30. Il modello avrebbe smesso di apprendere e il
sintomo sarebbe sembrato un limite dell'architettura.

*Correzione.* La direzione non è più dichiarata: `create_scheduler()` la legge
da `MONITOR_MODE`, perché lo scheduler avanza proprio su quella metrica. Due
dichiarazioni indipendenti della stessa grandezza sono ciò che ha permesso la
divergenza.

**`build_config()` restituiva riferimenti alle dizionari del modulo.** Il ciclo
di training vi aggiunge il numero di parametri e il commit; facendolo mutava
`config.SKELETON_LOSS_CONFIG` globale, violando l'invariante dichiarato in
sezione 0 secondo cui `config.py` non ha effetti collaterali. Il `config.json`
di una run era una vista viva della configurazione anziché un'istantanea.

*Correzione.* Copia profonda, e `base_channels` registrato direttamente da
`build_config()` invece di essere aggiunto dal chiamante.

**Interruzione al termine di una run degenere.** Se ogni epoca avesse prodotto
`NaN` sulla metrica monitorata, la riga finale di riepilogo avrebbe sollevato
un'eccezione formattando `None` — dopo che l'addestramento era già concluso.
Ora la condizione è segnalata.

**Imprecisioni minori.** Un commento in `config.py` indicava 7,1 M parametri
per U-Net invece di 7,76 M; un commento in `metrics.py` dichiarava assenza di
sincronizzazioni con l'host mentre la validazione delle etichette ne fa due per
batch; l'accuratezza per classe duplicava il recall, aggiungendo cinque colonne
ridondanti al registro.

### Controlli superati

| Ambito | Verifica |
|---|---|
| Numeri delle note | tutte le misure di queste note riprodotte dai dati attuali |
| Leakage | nessuno stem condiviso fra partizioni; nessuna copia aumentata in validation o test; nessuna immagine originale a cavallo di due partizioni |
| Provenienza | `class_weights.json` e `channel_stats.json` portano l'impronta `62de0ec4`, metodo e nomi delle classi coerenti con la configurazione |
| Normalizzazione | ciascuno dei cinque canali confrontato con il calcolo manuale delle proprie statistiche |
| Skeleton loss | il gradiente cambia quando il termine è attivo, verificato nel ciclo reale e non solo isolatamente |
| Riproducibilità | stesso seed, stesso risultato; seed diverso, risultato diverso; validation sempre nello stesso ordine |
| Coerenza | canali, classi, colori, pesi della loss e larghezza del modello mutuamente consistenti |
| Riferimenti | ogni nome importato dai quattro notebook esiste; nessun riferimento a funzioni rimosse nel refactoring |

### Una nota sull'aggregazione

Sia le metriche di sovrapposizione sia clDice sono **accumulate sull'intero
split** e calcolate una volta alla fine, non mediate per immagine. È la stessa
scelta della sezione 7.2 e per la stessa ragione: una media per immagine
peserebbe un frammento di cordone quanto uno intero. Va dichiarato perché in
letteratura clDice è talvolta riportato come media per immagine, e i due valori
non coincidono.

---

## 12. Risultati — confronto fra architetture a budget 70

### Disegno

Cinque run, identiche in tutto tranne l'architettura e, per `UNet`, il seed:
`ce` con pesi `sqrt_inverse`, AdamW `1e-4`, ReduceLROnPlateau, batch 4, 512×512,
5 canali, 70 epoche, skeleton loss spenta, split `62de0ec4`. Impronta del
codice `195c6b62` per le tre `UNet`, `d90d5da1` per `SegNet` e `PSPNet` — le due
differiscono per quattro righe dentro `build_result_row`, che costruisce la riga
CSV a training concluso e non può influire su modello, dati o metriche
(verificato con `git diff`).

Le metriche sono sul **test**, dal checkpoint migliore per `mean_iou_no_bg`.
Le soglie sono `2σ√2` calcolate sulla dispersione fra i tre seed di `UNet`.

### Sulla metrica principale non c'è differenza

| metrica | `UNet` (n=3) | `SegNet` | `PSPNet` | soglia |
|---|---|---|---|---|
| IoU media (no bg) | 0,6442 ± 0,0076 | 0,6496 | 0,6398 | 0,0214 |
| precisione media | 0,7719 ± 0,0051 | 0,7759 | **0,7882** | 0,0145 |
| recall media | 0,8708 ± 0,0065 | 0,8720 | **0,8459** | 0,0184 |
| clDice medio | 0,8996 ± 0,0053 | 0,8974 | **0,8785** | 0,0149 |

`SegNet` +0,0054 e `PSPNet` −0,0045 rispetto alla media `UNet`: entrambe entro
il rumore. **Sulla mean IoU le tre architetture sono indistinguibili**, e va
riportato così invece di ordinarle.

### La media aggregata nasconde però una differenza reale

| IoU per classe | `UNet` (n=3) | `SegNet` | `PSPNet` | soglia |
|---|---|---|---|---|
| `Cane` | 0,7167 ± 0,0078 | −0,0030 | **−0,0573** | 0,0220 |
| `Cordon` | 0,7760 ± 0,0074 | +0,0002 | +0,0123 | 0,0208 |
| `Node` | 0,4556 ± 0,0175 | −0,0022 | +0,0129 | 0,0494 |
| `Other` | 0,6286 ± 0,0018 | **+0,0264** | **+0,0143** | 0,0050 |

`PSPNet` è spostata lungo il compromesso precisione/recall — più precisa, meno
recall, peggiore sulla topologia — e perde **0,057 di IoU su `Cane`**, la
differenza più grande dell'intero confronto e sulla classe che il progetto deve
segmentare meglio.

*La spiegazione è architetturale, ed è la stessa misura della sezione 8 vista
dall'altro lato.* `PSPNet` non ha **nessuna convoluzione a piena risoluzione**:
scende subito, lavora la piramide a bassa risoluzione e risale con un solo
upsample bilineare. È il motivo per cui costa 1,25 GB contro i 3,34 di `UNet` e
i 5,32 di `SegNet`, ed è il motivo per cui i tralci — strutture larghe pochi
pixel — non sopravvivono. Economia di attivazioni e perdita di dettaglio fine
sono la stessa proprietà.

`SegNet` è invece indistinguibile da `UNet` ovunque tranne su `Other`, e costa
**52 s/epoca contro 32**, il 62 % in più a parità di parametri.

### Le tre non sono alla stessa distanza dalla convergenza

Regressione lineare della mean IoU di validazione sulle ultime dieci epoche:

| | pendenza ×10⁻³ per epoca | IoU guadagnata ep. 50→70 |
|---|---|---|
| `UNet` (inviluppo su 3 seed) | da **+1,32 a +2,85** | da +0,015 a +0,043 |
| `PSPNet` | +1,45 — dentro l'inviluppo | +0,019 |
| `SegNet` | **+0,44 — sotto l'inviluppo** | +0,011 |

**Questo è il risultato più importante dei cinque run, ed è metodologico.**
`SegNet` ha spianato mentre `UNet` e `PSPNet` stanno ancora salendo — ed è
proprio `SegNet` quella che nominalmente guida sulla IoU. Il confronto a 70
epoche è quindi **dipendente dal budget**: estendendolo, le due che salgono
guadagnerebbero e la terza quasi no, e un pareggio potrebbe diventare un
distacco senza che nessun modello sia cambiato.

Il criterio era stato formulato per intercettare un'architettura *troncata a
metà salita*; il caso reale è l'opposto — una che ha già finito mentre le altre
no. La conseguenza è la stessa: il budget pareggiato non basta a rendere il
confronto equo.

### Come viene chiusa la questione

Una run sonda: `UNet` + `ce` + seed 42 con `NUM_EPOCHS = 200`, lasciando
lavorare l'early stopping a `patience 20`, che a 70 epoche non poteva mai
scattare. Il punto in cui si ferma **è** la risposta.

Regola fissata prima di eseguirla:

- se si ferma entro ~110 epoche **e** la IoU a 70 è entro il 3 % del massimo →
  il budget di 70 resta, e la sonda diventa una misura riportabile invece di un
  dubbio taciuto
- se il guadagno oltre l'epoca 70 supera **0,03**, cioè una volta e mezza la
  soglia di rumore → il budget va alzato e le cinque run vanno rifatte prima di
  proseguire con la griglia

### Esito della sonda: il budget di 70 epoche resta

La run si è fermata **da sola all'epoca 135**, con l'early stopping a
`patience 20`, massimo all'epoca 115. Il meccanismo non era mai potuto scattare
a 70 epoche; alla prima occasione ha funzionato come previsto.

Il confronto va fatto **fra massimi, non fra epoche omologhe**: una run da 70
epoche non riporta il valore dell'epoca 70, riporta il miglior checkpoint fino
a lì, e la mean IoU di validazione oscilla di ±0,02 fra epoche consecutive.

| | best val IoU |
|---|---|
| miglior checkpoint entro l'epoca 70 | 0,6501 (epoca 69) |
| miglior checkpoint sulle 135 | 0,6625 (epoca 115) |
| **guadagno** | **+0,0124 (+1,9 %)** |
| soglia di rumore fra run identiche | 0,0214 |

**Raddoppiare il tempo di addestramento produce meno del rumore fra run
identiche.** Il budget di 70 epoche cattura il 98,1 % della mean IoU raggiunta
a convergenza, e il deficit non è distinguibile dalla dispersione fra seed.

*Ricaduta sull'equità del confronto.* La preoccupazione era che `SegNet`, già
piatta a 70 epoche, fosse avvantaggiata rispetto a `UNet` e `PSPNet` ancora in
salita. La sonda quantifica quel vantaggio: **al massimo +0,012 per le due che
salgono, meno per `SegNet`**, cioè circa metà della soglia di rumore. La
conclusione «le tre architetture sono indistinguibili» vale quindi a entrambi i
budget; cambierebbe solo l'ordine nominale, che è comunque privo di
significato. La questione è chiusa con una misura invece che con un'assunzione.

*Un risultato collaterale che vale la pena riportare.* La val loss tocca il
minimo all'**epoca 50** e da lì peggiora lentamente, mentre la mean IoU
continua a salire fino all'**epoca 115**: **65 epoche in cui la loss dice di
fermarsi e la metrica dice di continuare.** È la giustificazione empirica di
`MONITOR_METRIC = "mean_iou_no_bg"` invece della val loss — con la loss come
criterio, l'early stopping avrebbe interrotto l'addestramento perdendo 0,048 di
IoU, più del doppio della soglia di rumore.

Costo della sonda: 81 minuti per 135 epoche, 35,8 s/epoca.

### Confronto fra loss su `UNet` — parziale

Stesso disegno, architettura fissa a `UNet`, seed 42, budget 70, pesi di classe
`sqrt_inverse` attivi in tutti e tre i casi. Tenere i pesi accesi anche per la
Focal è deliberato: spegnerli solo lì confonderebbe «Focal contro CE» con
«pesata contro non pesata». Riferimento: le tre run `ce` a seed diverso.

| metrica | `ce` (n=3) | `combined` | `focal` | soglia |
|---|---|---|---|---|
| IoU media | 0,6442 ± 0,0076 | 0,6560 (+0,0118) | 0,6263 (−0,0179) | 0,0214 |
| precisione | 0,7719 ± 0,0051 | **0,7977 (+0,0258)** | 0,7575 (−0,0144) | 0,0145 |
| recall | 0,8708 ± 0,0065 | 0,8559 (−0,0149) | 0,8640 (−0,0068) | 0,0184 |
| clDice | 0,8996 ± 0,0053 | 0,8922 (−0,0074) | 0,8881 (−0,0115) | 0,0149 |
| IoU `Cane` | 0,7167 ± 0,0078 | 0,7127 (−0,0040) | 0,7005 (−0,0162) | 0,0220 |
| IoU `Cordon` | 0,7760 ± 0,0074 | 0,7715 (−0,0045) | 0,7596 (−0,0164) | 0,0208 |
| IoU `Node` | 0,4556 ± 0,0175 | 0,4842 (+0,0286) | 0,4357 (−0,0200) | 0,0494 |
| IoU `Other` | 0,6286 ± 0,0018 | **0,6556 (+0,0270)** | **0,6095 (−0,0191)** | 0,0050 |

**`combined` sposta il modello verso la precisione.** La IoU media resta dentro
il rumore, ma la precisione sale di 0,026 oltre soglia e il divario
recall−precisione passa da **+0,099 a +0,058**, quasi dimezzato. È il termine
Dice che penalizza i falsi positivi in modo che la sola cross-entropy pesata non
fa. Costa il 9 % di tempo per epoca.

Il dato è rilevante per la sezione 7.6: l'ablazione prevede che la Skeleton
Recall Loss *allarghi* quel divario. Avere una loss che lo restringe fornisce il
contrasto con cui leggere quel risultato.

**`focal` è peggiore su tutte e otto le metriche, e su nessuna migliore.**
Presa una per una, quasi nessuna differenza supera la soglia; è la
**concordanza dei segni** a essere informativa — otto su otto nella stessa
direzione. Sotto indipendenza sarebbe p ≈ 0,008, ma le metriche sono fra loro
correlate, quindi il valore va letto come indizio e non come test. Il divario
recall−precisione resta a +0,107, quindi non è nemmeno un compromesso diverso:
è semplicemente peggio.

*Interpretazione, e la verifica che ne discende.* La `FocalLoss` del progetto è
cross-entropy pesata per classe moltiplicata per `(1−p)^γ`: i pesi di classe
svolgono il ruolo di **α** in Lin et al. Nell'articolo originale, con γ = 2 il
valore migliore era **α = 0,25 sulla classe rara** — gli autori la
*abbassavano*, perché γ già concentra sugli esempi difficili. I nostri pesi
`sqrt_inverse` alzano `Node` di 17,7 volte. I due meccanismi correggono lo
sbilanciamento **nella stessa direzione**, molto più forte di qualunque
configurazione che gli autori abbiano testato, ed è la spiegazione più naturale
del risultato. Una run sonda con `apply_weights = False` sulla sola Focal
verifica l'ipotesi al costo di una run.

*Stato:* il confronto è per ora limitato a `UNet` e a un solo seed per loss.
`SegNet` e `PSPNet` completano la griglia.

### I pesi di classe danneggiano due architetture su tre

La sonda sulla Focal senza pesi ha prodotto un risultato inatteso — precisione
in forte crescita, recall in calo, IoU media la migliore fino a quel momento —
ma **confondeva due fattori**: la loss e la pesatura. Il controllo è stato
`ce` senza pesi, poi replicato su tre seed per entrambe le configurazioni.

Due gruppi da tre run, identici in tutto tranne `apply_weights`. Test t di
Welch, `UNet`, budget 70, seed 42/123/456.

| metrica | pesata (n=3) | senza pesi (n=3) | differenza | p |
|---|---|---|---|---|
| IoU media | 0,6442 ± 0,0076 | **0,6674 ± 0,0075** | +0,0232 | 0,020 |
| precisione | 0,7719 ± 0,0051 | **0,8500 ± 0,0132** | +0,0781 | 0,004 |
| recall | 0,8708 ± 0,0065 | 0,8120 ± 0,0030 | −0,0587 | 0,001 |
| clDice | 0,8996 ± 0,0053 | 0,9087 ± 0,0056 | +0,0092 | 0,109 |
| IoU `Cane` | 0,7167 ± 0,0078 | **0,7481 ± 0,0067** | +0,0314 | 0,007 |
| IoU `Cordon` | 0,7760 ± 0,0074 | **0,8009 ± 0,0068** | +0,0250 | 0,013 |
| IoU `Node` | 0,4556 ± 0,0175 | 0,4356 ± 0,0147 | −0,0201 | 0,204 |
| IoU `Other` | 0,6286 ± 0,0018 | **0,6849 ± 0,0070** | +0,0563 | 0,003 |

**Togliere i pesi migliora `Cane`, `Cordon`, `Other` e la IoU media, e inverte
il compromesso:** il divario recall−precisione passa da **+0,099 a −0,038**. Per
la prima volta il modello è più preciso che sensibile — smette di gonfiare i
tralci, che è il difetto noto fin dall'inizio.

**`Node`, la classe che i pesi esistevano per proteggere, non ne trae
beneficio** (p = 0,20). I pesi ne spostano solo il compromesso interno: recall
da 0,52 a 0,57, precisione da 0,75 a 0,63. Si pagano tre classi su quattro per
riequilibrare la quarta senza migliorarla.

*Meccanismo, ed è lo stesso della Focal.* `mean_iou_no_bg` è una media **macro**:
le quattro classi in primo piano pesano uguale a prescindere dalla frequenza,
quindi **la metrica bilancia già le classi**. Aggiungere `sqrt_inverse` nella
loss — `Node` ×17,7 — le bilancia una seconda volta, nella stessa direzione. È
la doppia correzione già vista fra γ e α nella Focal; la Focal pesata era la
peggiore di tutte perché ne accumulava tre.

*Cautele.* Tre run per gruppo: il test t ha pochi gradi di libertà e i valori p
sono approssimati. Con una correzione di Bonferroni su otto confronti
(α = 0,00625) sopravvivono precisione, recall, `Cane` e `Other`; IoU media e
`Cordon` no. La direzione è però concorde su tutte le metriche di area e le
dimensioni dell'effetto sono grandi. Il risultato è su `UNet` soltanto.

### La replica sulle altre due architetture: l'effetto non è universale

Il risultato è stato replicato su `SegNet` e `PSPNet`, un seed per braccio,
tutto il resto invariato. Le soglie sono quelle misurate su `UNet`.

| metrica | `UNet` (n=3) | `SegNet` | `PSPNet` | soglia |
|---|---|---|---|---|
| IoU media | **+0,0232** | **+0,0320** | −0,0122 | 0,0214 |
| precisione | **+0,0781** | **+0,0813** | **+0,0333** | 0,0145 |
| recall | **−0,0588** | **−0,0472** | **−0,0541** | 0,0184 |
| clDice | +0,0091 | +0,0103 | **−0,0172** | 0,0149 |
| IoU `Cane` | **+0,0314** | **+0,0332** | −0,0027 | 0,0220 |
| IoU `Cordon` | **+0,0249** | **+0,0255** | −0,0065 | 0,0208 |
| IoU `Node` | −0,0200 | +0,0295 | **−0,0494** | 0,0494 |
| IoU `Other` | **+0,0563** | **+0,0397** | **+0,0097** | 0,0050 |

`SegNet` replica `UNet` quasi punto per punto — precisione, `Cane`, `Cordon` e
clDice cadono entro un punto percentuale dell'effetto misurato con tre seed.
**`PSPNet` no.** La sua IoU media non migliora (−0,0122, entro il rumore),
`Cane` e `Cordon` restano invariate, e clDice e `Node` peggiorano.

*Perché.* Il divario recall−precisione di partenza distingue le architetture:

| | pesata | senza pesi |
|---|---|---|
| `UNet` | +0,099 | −0,038 |
| `SegNet` | +0,096 | −0,033 |
| `PSPNet` | **+0,058** | −0,030 |

Tutte e tre finiscono nello stesso punto, ma `PSPNet` partiva **già a metà
strada**: è l'architettura senza convoluzioni a piena risoluzione, produce
maschere più grossolane e quindi meno falsi positivi, ed è intrinsecamente
spostata verso la precisione (sezione 12, dove perde 0,057 di IoU su `Cane`).
I pesi di classe ne stavano **compensando la distorsione nativa**; toglierli la
porta oltre l'ottimo. Lo stesso intervento è un guadagno per due architetture e
un pareggio per la terza, a seconda di dove ciascuna parte.

*Un effetto collaterale sul confronto fra architetture.* La conclusione della
sezione 12 dipende dalla baseline:

| | pesata | senza pesi |
|---|---|---|
| `SegNet` vs `UNet` | +0,0054 (indist.) | +0,0142 (indist.) |
| `PSPNet` vs `UNet` | −0,0044 (indist.) | **−0,0399 (distinguibile)** |

Sulla baseline pesata le tre architetture sono indistinguibili; su quella non
pesata **`PSPNet` resta indietro in modo misurabile**. Va detto esplicitamente:
la classifica fra architetture non è una proprietà dei soli modelli, dipende
dalla configurazione su cui li si confronta.

*Conseguenze.* Si adotta la **baseline non pesata** come configurazione di
riferimento: migliora due architetture su tre in modo significativo e sulla
terza non peggiora in modo distinguibile. Le run precedenti restano valide come
braccio «pesato» del confronto, non come baseline. L'ablazione della sezione 7.6
va eseguita sulla nuova base, o misurerebbe l'effetto della Skeleton Recall Loss
su un modello mal configurato.

La questione 7 si irrigidisce. Su `UNet` i pesi peggiorano `Node` di 0,020, su
`SegNet` lo migliorano di 0,030, su `PSPNet` lo migliorano di 0,049: variazioni
tutte al limite o dentro la soglia di quella classe, e **di segno discorde**. La
lettura corretta non è che i pesi aiutino o danneggino `Node`, ma che **`Node` è
sostanzialmente indifferente**, mentre le altre tre classi ne sono danneggiate
in modo consistente su due architetture su tre. Il meccanismo che doveva
proteggere la classe rara non la protegge.

### Confronto definitivo fra architetture — baseline non pesata, 3 seed ciascuna

Diciotto run in tutto (tre architetture × tre seed × due bracci di pesatura),
`ce`, budget 70, batch 4, split `62de0ec4`, impronta `d90d5da1`. Questa è la
tabella che va in tesi.

| metrica | `UNet` | `SegNet` | `PSPNet` |
|---|---|---|---|
| IoU media | 0,6674 ± 0,0075 | **0,6820 ± 0,0019** | 0,6343 ± 0,0075 |
| precisione | 0,8500 ± 0,0132 | 0,8615 ± 0,0039 | 0,8381 ± 0,0145 |
| recall | 0,8120 ± 0,0030 | 0,8216 ± 0,0034 | 0,7891 ± 0,0048 |
| clDice | 0,9087 ± 0,0056 | 0,9076 ± 0,0017 | 0,8664 ± 0,0061 |
| IoU `Cane` | 0,7481 ± 0,0067 | 0,7496 ± 0,0025 | 0,6628 ± 0,0114 |
| IoU `Cordon` | 0,8009 ± 0,0068 | 0,8035 ± 0,0028 | 0,7857 ± 0,0044 |
| IoU `Node` | 0,4356 ± 0,0147 | **0,4826 ± 0,0031** | 0,4210 ± 0,0057 |
| IoU `Other` | 0,6849 ± 0,0070 | 0,6924 ± 0,0024 | 0,6678 ± 0,0131 |
| s/epoca | 32,7 | **52,7** | 29,4 |

Test t di Welch contro `UNet`:

| | differenza | p |
|---|---|---|
| `SegNet` IoU media | +0,0146 | 0,069 |
| `SegNet` recall | +0,0096 | 0,022 |
| `SegNet` IoU `Node` | +0,0470 | 0,027 |
| `PSPNet` IoU media | −0,0331 | **0,006** |
| `PSPNet` clDice | −0,0423 | **0,001** |
| `PSPNet` IoU `Cane` | **−0,0853** | **0,001** |
| `PSPNet` recall | −0,0229 | **0,004** |

**`PSPNet` è nettamente ultima**, e non per un margine ambiguo: perde 0,085 di
IoU su `Cane` con p = 0,001 e 0,042 di clDice con p = 0,0009. Sono i due valori
p più bassi della tabella, ed entrambi riguardano le strutture sottili. La
spiegazione è quella già data: nessuna convoluzione a piena risoluzione, quindi
i tralci non sopravvivono all'upsample finale. Anche contro `SegNet` il distacco
è netto (+0,048 di IoU, p = 0,006).

**`SegNet` è nominalmente prima ma non in modo conclusivo**: +0,0146 su `UNet`
con p = 0,069, sopra la soglia convenzionale del 5 %. Guadagna in modo
significativo su recall e su `Node`, ma la IoU media resta un'indicazione, non
un risultato. **E costa il 61 % di tempo per epoca in più.** Con questi numeri,
`UNet` resta la scelta ragionevole di default e `SegNet` un'alternativa
difendibile solo se il tempo di addestramento non è un vincolo.

*Nota su `SegNet`.* La sua dispersione fra seed è notevolmente più bassa delle
altre — σ = 0,0019 sulla IoU contro 0,0075 di `UNet` e `PSPNet`, quattro volte
meno. Non abbiamo una spiegazione misurata; va riportato come osservazione.

*Convalida del metodo.* Le stime provvisorie ricavate con una sola run per
architettura davano `PSPNet` a −0,0399 e `SegNet` a +0,0142 contro `UNet`; con
tre seed diventano −0,0331 e +0,0146. Usare la σ di `UNet` come metro per run
singole si è dimostrato affidabile, il che dà fiducia alle altre affermazioni
costruite allo stesso modo in attesa di replica.

---

## Riepilogo delle questioni aperte

### Risolte

| # | Questione | Esito |
|---|---|---|
| 1 | Frame indipendenti o sequenze continue? | Acquisizioni indipendenti: split casuale valido |
| 2 | `dilation_kernel_size`: 2 o 3? | Mantenuto 2, sulla base della copertura misurata |
| 3 | Formato dei file di intensità | **Due lotti con formati diversi**; riconciliati in lettura da `read_modality_image()` — sezione 4 |
| 4 | Pesi delle classi: calcolo in EDA, salvataggio, ricarica nel training | Fatto: notebook 02 li scrive, `config.load_class_weights()` li rilegge con verifica dell'impronta |
| 6 | Adattamento della classe `Dataset` all'intensità a canale singolo | Chiuso a monte: il dato che arriva alla classe è già conforme a `MODALITY_CHANNELS` |
| 5 | Definizione delle colonne di `experiments.csv` | Definite: 66 colonne, elenco in sezione 9 |
| 8 | Riannotazione e revisione delle annotazioni | Fatte: dataset portato a 100 immagini, campione reintegrato, revisione documentata in sezione 1 |
| 9 | Formule di Skeleton Recall Loss e clDice | Verificate sui PDF in `reference/` — sezione 7.1 |
| 10 | Critica alla Skeleton Recall Loss | Letta e analizzata: l'obiezione sul gradiente costante non regge sui logit, ma l'asimmetria recall/precisione sì — sezione 7.4 |
| 11 | Peso della Skeleton Recall Loss | Fissato a **0,5**, cioè l'1,0 dell'articolo corretto per la diversa scala della loss principale — sezione 7.3 |

La questione 3 era stata dichiarata risolta con una conclusione **errata**: la
verifica originale ispezionava un campione casuale e ne riportava un formato
arbitrario. Vedere la sezione 4 per la ricostruzione dell'errore.

### Aperte

| # | Questione | Quando |
|---|---|---|
| 7 | Eliminazione della classe `Node` via `LABELS_TO_IGNORE` — misure in sezione 5 | Solo se i risultati la giustificassero |
| 11 | Scelta del peso della Skeleton Recall Loss, tenuto conto della diversa scala della loss principale — sezione 7.3 | Notebook 03 |
| 12 | Ablazione con e senza Skeleton Recall Loss | **Risolta:** 3 seed per braccio, sezione 7.7. La previsione della 7.6 è corretta sul meccanismo ma sbagliata sul piano: lo scambio precisione↔recall avviene sulle metriche topologiche (p < 0,01), mentre sull'area il saldo è un peggioramento uniforme (IoU −0,043, p = 0,003) |
| 13 | Criterio di ordinamento della tabella comparativa — sezione 7.5. La scelta di `MONITOR_METRIC` è invece **chiusa**: la sonda a 200 epoche mostra che la val loss tocca il minimo 65 epoche prima del massimo della IoU, e fermarsi su di essa costerebbe 0,048 di IoU | Prima della tabella finale |
| 14 | **Configurazione di riferimento senza pesi di classe** — sezione 12. **Risolta:** adottata. Migliora `UNet` (+0,023, n=3) e `SegNet` (+0,032) in modo significativo, su `PSPNet` non produce effetti distinguibili perché quella architettura era già spostata verso la precisione. Da rifare su questa base: ablazione SRL, e la griglia sulle loss | Fatto |

---

## Stato dell'infrastruttura

**Controllo di versione.** Il codice è tracciato con Git e ospitato su un
repository privato GitHub. I dati (`data/`), gli output degli esperimenti
(`experiments/`, `results/`) e le impostazioni locali dell'editor sono esclusi
dal versionamento. Gli output dei notebook vengono rimossi automaticamente al
momento del commit tramite `nbstripout`, per mantenere i diff leggibili.

**Ambiente di esecuzione.** Ambiente conda dedicato (`vine`, Python 3.11.15),
con le versioni delle librerie registrate in `requirements.txt`. Il file era
salvato in UTF-16 con BOM, effetto di `pip freeze >` eseguito in PowerShell: in
quella codifica `pip install -r` può rifiutarlo su un'altra macchina, e avrebbe
rotto la configurazione dell'ambiente su Colab. È stato riscritto in UTF-8.

La preparazione dei dati viene eseguita in locale — l'intero notebook 01
richiede circa 20 secondi sul dataset completo, contro tempi dell'ordine dei
minuti su Colab, dove la lettura da Drive costituisce il collo di bottiglia. Il training sarà eseguito su
Colab per disporre di una GPU con memoria adeguata.

**Nota sulla comparabilità.** Tutte le run destinate alla tabella comparativa
finale devono essere eseguite sullo stesso hardware: la dimensione del batch,
vincolata dalla memoria GPU disponibile, influenza il risultato, e run con batch
diversi non sarebbero confrontabili.
