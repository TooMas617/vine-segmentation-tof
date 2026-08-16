# Note sulla pipeline — Segmentazione semantica di componenti di vite da dati ToF

Documento di lavoro: registra **cosa fa** ogni fase della pipeline, **quali scelte
implementative** sono state prese e **perché**. Serve come base per il capitolo
metodologico della tesi e come riferimento per chi riprenderà il codice.

Ultimo aggiornamento: notebook `01_Data_Preparation` completato ed eseguito
sull'intero dataset (93 immagini) in ambiente locale.

---

## 0. Architettura del progetto

### Struttura

```
VineProject_ToF/
├── src/                    moduli Python condivisi
│   ├── config.py           configurazione centralizzata
│   └── data_preparation.py logica di preparazione dei dati
├── notebooks/
│   └── 01_Data_Preparation.ipynb
├── data/
│   ├── raw/                intensity, depth, confidence, annotations
│   ├── processed/          label maps, skeleton, maschere colorate
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

L'esecuzione sull'intero dataset ha segnalato **4 forme scartate** su 93 file,
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

*Decisione:* lasciarle scartate. L'incidenza è trascurabile (4 forme su 93
immagini) e la segnalazione esplicita a schermo rende il fenomeno tracciabile.
L'alternativa — interpretare i poligoni degeneri delle classi filiformi come
polilinee — introdurrebbe una regola implicita a fronte di un beneficio minimo.

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

**Risultati sulla classe `Cane`** (93 immagini):

| | kernel 2×2 | kernel 3×3 |
|---|---|---|
| copertura media | 34,4 % | 53,0 % |
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

**Ripartizione effettiva** su 93 immagini: 65 train (69,9 %), 14 validation
(15,1 %), 14 test (15,1 %). Impronta della partizione: `d57f79eb`.

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

**Validation e test copiati senza modifiche.** Le tre cartelle hanno la stessa
struttura, così la classe `Dataset` le tratta in modo identico; solo il contenuto
di train è aumentato.

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

**Formato effettivo dei dati.** La verifica sui file di intensità ha dato
`uint8` con forma `(480, 640)`: immagini a 8 bit e canale singolo. Non si è
quindi verificata alcuna perdita di profondità di bit con il codice precedente.
Resta però una differenza rilevante: `cv2.imread()` senza flag restituiva
`(480, 640, 3)` in formato BGR, ovvero tre canali identici, triplicando
inutilmente lo spazio occupato. Con `IMREAD_UNCHANGED` i file aumentati sono a
canale singolo, coerentemente con `MODALITY_CHANNELS['intensity'] = 1`.

> **Da verificare nella classe `Dataset`:** se il codice originale assumeva
> immagini di intensità a 3 canali, va adattato al formato a canale singolo.

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

## Riepilogo delle questioni aperte

### Risolte

| # | Questione | Esito |
|---|---|---|
| 1 | Frame indipendenti o sequenze continue? | Acquisizioni indipendenti: split casuale valido |
| 2 | `dilation_kernel_size`: 2 o 3? | Mantenuto 2, sulla base della copertura misurata |
| 3 | Formato dei file di intensità | `uint8` a canale singolo; nessuna perdita di bit |

### Aperte

| # | Questione | Quando |
|---|---|---|
| 4 | Pesi delle classi: calcolo in EDA, salvataggio su file, ricarica nel training | Notebook 02 |
| 5 | Definizione delle colonne di `experiments.csv` | Notebook 03 |
| 6 | Adattamento della classe `Dataset` all'intensità a canale singolo | Notebook 03 |

---

## Stato dell'infrastruttura

**Controllo di versione.** Il codice è tracciato con Git e ospitato su un
repository privato GitHub. I dati (`data/`), gli output degli esperimenti
(`experiments/`, `results/`) e le impostazioni locali dell'editor sono esclusi
dal versionamento. Gli output dei notebook vengono rimossi automaticamente al
momento del commit tramite `nbstripout`, per mantenere i diff leggibili.

**Ambiente di esecuzione.** Ambiente conda dedicato (`vine`, Python 3.11.15),
con le versioni delle librerie registrate in `requirements.txt`. La preparazione
dei dati viene eseguita in locale — l'intero notebook 01 richiede circa 20
secondi su 93 immagini, contro tempi dell'ordine dei minuti su Colab, dove la
lettura da Drive costituisce il collo di bottiglia. Il training sarà eseguito su
Colab per disporre di una GPU con memoria adeguata.

**Nota sulla comparabilità.** Tutte le run destinate alla tabella comparativa
finale devono essere eseguite sullo stesso hardware: la dimensione del batch,
vincolata dalla memoria GPU disponibile, influenza il risultato, e run con batch
diversi non sarebbero confrontabili.
