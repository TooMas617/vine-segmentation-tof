# Note sulla pipeline — Segmentazione semantica di componenti di vite da dati ToF

Documento di lavoro: registra **cosa fa** ogni fase della pipeline, **quali scelte
implementative** sono state prese e **perché**. Serve come base per il capitolo
metodologico della tesi e come riferimento per chi riprenderà il codice.

Ultimo aggiornamento: notebook `01_Data_Preparation` e `02_EDA` completati ed
eseguiti; moduli `dataset.py`, `losses.py` e `metrics.py` scritti. Il dataset è
passato da 93 a 92 immagini per l'esclusione di un campione con annotazione
incompleta (sezione 3).

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
│   ├── excluded/           campioni non utilizzabili, con il motivo (README.md)
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
EDA ha mostrato che il dataset non usa un vocabolario unico: 39 file su 92 usano
etichette italiane, 53 usano etichette inglesi.

| Convenzione | File | Vocabolario |
|---|---|---|
| Italiana | 39 | `tralcio`, `tralcio_secco`, `cordone`, `curva`, `nodo`, `altro`, `ferro` |
| Inglese | 53 | `cane`, `cordon`, `node`, `other` |

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
è quella che contribuisce di più. La granularità italiana tornerebbe rilevante
solo se il problema cambiasse — per esempio distinguendo il tralcio secco da
quello vivo, distinzione rilevante in potatura — e in quel caso il sottoinsieme
utilizzabile sarebbe meno della metà del dataset.

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

### Esclusione di campioni

Un campione, `blaze-101-25367037-20087`, è stato tolto dal dataset. Il cordone è
presente e ben visibile nell'immagine, ma **non è annotato**: nel JSON non
compare alcuna forma `cordone`/`cordon`, e la label map risultava priva di pixel
di quella classe. Era l'unica immagine su 93 in questa condizione — la seconda
più povera di cordone ne conta 4 682 pixel, contro una mediana di 11 052.

*Meccanismo adottato.* I quattro file del campione (annotazione più le tre
modalità grezze) sono stati spostati in `data/excluded/`, che rispecchia la
struttura di `data/raw/`. La pipeline parte dai JSON presenti in
`data/raw/annotations/`, quindi il campione sparisce da solo da label map,
split e dataset aumentato, **senza alcuna modifica al codice**.

*Motivazione della scelta.* Le alternative erano cancellare i file o
introdurre una lista di esclusione in `config.py`. La cancellazione rende un
campione mancante indistinguibile da una perdita accidentale; una lista nel
codice mescola una questione di qualità del dato con la configurazione del
modello. Lo spostamento conserva i file, rende l'esclusione evidente a chi
guarda le cartelle e permette di reintegrare il campione spostandolo indietro
una volta annotato. Il `README.md` in `data/excluded/` elenca i campioni e la
procedura di recupero.

*Contesto.* Il dataset è destinato a crescere fino a circa 100 immagini con
annotazioni aggiunte manualmente; in quell'occasione le annotazioni esistenti
verranno ricontrollate e questo campione potrà rientrare. La partizione cambierà
comunque a quel punto, quindi ricalcolarla ora non ha costo.

### Ripartizione effettiva

Su 92 immagini: 64 train (69,6 %), 14 validation (15,2 %), 14 test (15,2 %).
Impronta della partizione: `7afaa392`.

> La partizione precedente, su 93 immagini, era 65/14/14 con impronta
> `d57f79eb`. Rimuovendo uno stem l'intera partizione viene ridisegnata, non
> solo il campione tolto: `train_test_split` mescola l'elenco completo, quindi
> l'assegnazione di tutti gli altri campioni cambia. Metriche prodotte con
> impronte diverse non sono confrontabili.

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
`depth` è invece genuinamente a tre canali su tutti i file (0 su 93 con piani
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

Su 92 immagini, partizione `7afaa392`. Quota di pixel per classe:

| Classe | train | val | test | totale |
|---|---|---|---|---|
| Background | 83,55 % | 83,37 % | 85,35 % | 83,80 % |
| Cane | 11,61 % | 11,66 % | 9,75 % | 11,33 % |
| Cordon | 3,58 % | 3,61 % | 3,67 % | 3,60 % |
| Node | 0,26 % | 0,37 % | 0,24 % | 0,27 % |
| Other | 1,01 % | 0,98 % | 0,99 % | 1,00 % |

Numero di immagini in cui ciascuna classe compare:

| Classe | train | val | test | totale |
|---|---|---|---|---|
| Background | 64/64 | 14/14 | 14/14 | 92/92 |
| Cane | 64/64 | 14/14 | 14/14 | 92/92 |
| Cordon | 64/64 | 14/14 | 14/14 | 92/92 |
| Node | 45/64 | 12/14 | 11/14 | 68/92 |
| Other | 62/64 | 13/14 | 12/14 | 87/92 |

**Lo sbilanciamento è forte ma non estremo.** Background è circa 310 volte più
frequente di `Node`. Con `sqrt_inverse` questo si traduce in un peso di circa
17,6 volte quello dello sfondo — elevato ma trattabile, ed è esattamente la
ragione per cui quel metodo è preferito a `inverse`, che produrrebbe un fattore
310 e renderebbe l'addestramento instabile sulle classi rare.

**`Node` non è solo raro, è spesso assente.** Compare in 68 immagini su 92: manca
del tutto da un quarto del dataset. È un'informazione che la sola quota di pixel
nascondeva, e ha due conseguenze — le metriche calcolate per immagine su `Node`
sono indefinite in 24 casi, e la varianza del suo IoU sarà alta. È il dato che
motiva la valutazione riportata più sotto.

**Il controllo di presenza passa.** Ogni classe compare in ogni partizione, quindi
la mean IoU resta ben definita ovunque. Il margine più stretto è `Node` nel test,
presente in 11 immagini su 14.

**Le partizioni sono comparabili.** Lo scostamento maggiore è su `Cane`, 9,75 %
nel test contro 11,61 % nel train. Su 14 immagini è entro la variabilità attesa e
non giustifica una suddivisione stratificata; resta un valore da riguardare se il
test dovesse comportarsi in modo anomalo.

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

**Valori misurati sul training set** (92 immagini, partizione `7afaa392`):

| Canale | Media | Dev. std |
|---|---|---|
| intensity | 82,61 | 28,10 |
| depth_B | 84,04 | 118,65 |
| depth_G | 62,68 | 95,19 |
| depth_R | 33,97 | 76,95 |
| confidence | 10,83 | 19,57 |

### Pesi delle classi

Calcolati sulle sole frequenze del training set. Con `sqrt_inverse` e
normalizzazione al minimo:

| Classe | Quota pixel (train) | Peso |
|---|---|---|
| Background | 83,55 % | 1,00 |
| Cane | 11,61 % | 2,68 |
| Cordon | 3,58 % | 4,83 |
| Node | 0,26 % | 18,08 |
| Other | 1,01 % | 9,11 |

Il rapporto fra la classe più pesante e la più leggera è **18,1×**. Con
`inverse` sarebbe **327×**: un fattore di quell'ordine rende l'addestramento
instabile sulle classi rare, perché ogni pixel di `Node` sbagliato pesa quanto
327 pixel di sfondo e il gradiente viene dominato da poche decine di pixel per
immagine. È la ragione per cui `sqrt_inverse` resta il metodo predefinito.

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
| Background | 67,9 % |
| Cane | 12,8 % |
| Cordon | 1,8 % |
| Node | 0,6 % |
| Other | 12,7 % |

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
sovrascritti. Eseguita sulle 92 immagini riproduce i valori già documentati.

| | kernel 2×2 | kernel 3×3 |
|---|---|---|
| copertura media `Cane` | 34,4 % | 53,0 % |
| copertura massima `Cane` | 56,6 % | **83,7 %** |
| immagini `Cane` oltre l'80 % | **0** | 3 |
| copertura media `Cordon` | 13,0 % | 19,9 % |
| copertura massima `Cordon` | 33,2 % | 50,2 % |

Il `Cordon` non si avvicina alla soglia con nessuno dei due kernel: è una
struttura più spessa dei tralci, quindi il tubo dello scheletro ne occupa una
frazione minore. Il vincolo viene interamente dalla classe `Cane`.

Il grafico mostra la distribuzione completa per immagine invece di due barre
riassuntive, perché la decisione dipende dalla **coda superiore**: un kernel è
inutilizzabile se alcune immagini superano la soglia, per quanto bene si
comporti sulle altre.

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
si trovino sul cordone. Le misure non la sostengono (dati su 93 immagini, prima
dell'esclusione del campione della sezione 3):

| Misura | Valore |
|---|---|
| Pixel di `Node` interni alla maschera del cordone | 54,6 % |
| Bordo dei pixel di `Node` su `Cordon` | 38,4 % |
| Bordo dei pixel di `Node` su `Cane` | 32,1 % |
| Bordo dei pixel di `Node` su `Background` | 28,9 % |

Quasi metà dei nodi **non** si trova sul cordone, coerentemente con il fatto che
un nodo è tale su qualsiasi germoglio. Il caso limite lo conferma: il campione
escluso in sezione 3 conteneva due nodi annotati e nessun cordone. Una fusione
uniforme verso `Cordon` etichetterebbe come cordone circa il 45 % di pixel che
stanno sui tralci, corrompendo entrambe le classi.

**Meccanismo corretto, se si decidesse di procedere:** aggiungere `nodo`/`node`
a `config.LABELS_TO_IGNORE`. I poligoni dei nodi non verrebbero rasterizzati e
quei pixel conserverebbero la struttura sottostante — cordone dove il nodo sta
sul cordone, tralcio dove sta sul tralcio. È una fusione corretta pixel per
pixel invece che una regola uniforme sbagliata in metà dei casi, e costa una
riga di configurazione.

*Da verificare prima di adottarlo:* quanti pixel di `Node` finirebbero su
`Background`. Il 28,9 % di bordo su sfondo suggerisce che una parte dei poligoni
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

**Il vincolo che limita la correzione.** `CenterCrop` alla dimensione target
richiede che l'immagine sia almeno grande quanto il ritaglio, cioè
`target · f · s ≥ target`, ovvero **`f · s ≥ 1`** — che è esattamente la
condizione dell'ingrandimento di valutazione. Uno schema
*ingrandisci-e-ritaglia* può quindi **raggiungere** la scala di valutazione ma
non scendere sotto: il limite si tocca quando il ritaglio consuma l'immagine
intera.

**Valore adottato.** `scale_limit` passa da `(-0,30 ; 0,30)` a
`(-0,32 ; 0,30)`, cioè `s ∈ [0,68 ; 1,30]`. Con `f = 1,5` questo dà
`f · s = 1,02` all'estremo inferiore: il minimo di training scende da 0,84 a
0,816, passando dal 5 % al 2 % sopra la scala di valutazione. Il valore esatto
che azzererebbe il divario è `s = 2/3`, ma `f · s` sarebbe esattamente 1 e un
arrotondamento a 511 pixel farebbe fallire il ritaglio; 0,68 lascia 10 pixel di
margine.

**Cosa questa modifica non risolve.** `RandomScale` ha `p = 0,5`: metà dei
campioni conserva `s = 1,0`, quindi ingrandimento 1,20, il 50 % più zoomato
della valutazione. Spostare anche quel caso richiederebbe di abbassare `f` da
1,5 a circa 1,25, che porterebbe il caso tipico a 1,00. Il margine geometrico
basterebbe — una rotazione di 15° su un lato `L` lascia un quadrato inscritto
di `L / (cos 15° + sin 15°) = L / 1,225`, cioè 522 px su 640, più dei 512
necessari — ma `OpticalDistortion` non ha un margine calcolabile
analiticamente e richiederebbe una verifica visiva. Rimandato: se i risultati
mostrassero un divario anomalo fra training e validation, è il primo posto
dove guardare.

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

*Verifica* sulle 92 immagini, applicando il loro algoritmo alla lettera:

| | per classe (nostro) | unione poi divisione |
|---|---|---|
| pixel di scheletro del cordone | 1 358 | 1 664 |
| **componenti connesse** | **9,32** | **14,36** |
| immagini con più frammenti | — | **86 / 92** |

L'unione produce più pixel ma il 54 % di frammenti in più, e peggiora in 86
immagini su 92.

**Kernel di dilatazione quadrato 2×2 invece del diamante di raggio 2.**
L'articolo dilata con un diamante di raggio 2, cioè 13 pixel.

*Motivazione.* Non è una questione di sconfinamento: entrambe le
implementazioni ritagliano il tubo sulla maschera — il progetto con
`tubed & binary_mask`, l'articolo con la moltiplicazione `Y_skel × Y`. Il
problema è la **copertura**:

| Kernel | Pixel | Copertura media di `Cane` | Massima | Immagini oltre l'80 % |
|---|---|---|---|---|
| quadrato 2×2 | 4 | 34,4 % | 56,6 % | 0 / 92 |
| quadrato 3×3 | 9 | 53,0 % | 83,7 % | 3 / 92 |
| diamante r2 | 13 | 65,7 % | 92,8 % | 7 / 92 |

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
quindi **dimezzato**: `w = 0,2` qui equivale a circa `0,4` là, fra i due valori
che gli autori hanno testato.

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
3. **È più adatto ai dati.** Il training set conta 256 campioni aumentati da 64
   immagini originali. Trentuno milioni di parametri su 256 campioni sono
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
vinca: su 256 campioni di training un modello da 31 milioni di parametri ha un
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

Con 256 campioni e batch 4 il resto è zero, quindi il caso non si presenta oggi;
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

### Colonne di `experiments.csv`

La riga registra **come** una run era configurata oltre a **cosa** ha
prodotto: una metrica senza le impostazioni che l'hanno generata non è
confrontabile con niente. Settantadue colonne, raggruppate così:

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
| 5 | Definizione delle colonne di `experiments.csv` | Definite: 72 colonne, elenco in sezione 9 |
| 9 | Formule di Skeleton Recall Loss e clDice | Verificate sui PDF in `reference/` — sezione 7.1 |
| 10 | Critica alla Skeleton Recall Loss | Letta e analizzata: l'obiezione sul gradiente costante non regge sui logit, ma l'asimmetria recall/precisione sì — sezione 7.4 |

La questione 3 era stata dichiarata risolta con una conclusione **errata**: la
verifica originale ispezionava un campione casuale e ne riportava un formato
arbitrario. Vedere la sezione 4 per la ricostruzione dell'errore.

### Aperte

| # | Questione | Quando |
|---|---|---|
| 7 | Eliminazione della classe `Node` via `LABELS_TO_IGNORE` — misure in sezione 5 | Dopo l'ampliamento del dataset |
| 8 | Riannotazione del campione escluso e revisione delle annotazioni esistenti | All'ampliamento del dataset a ~100 immagini |
| 11 | Scelta del peso della Skeleton Recall Loss, tenuto conto della diversa scala della loss principale — sezione 7.3 | Notebook 03 |
| 12 | Ablazione con e senza Skeleton Recall Loss, per verificare la previsione della sezione 7.6 | Esperimenti definitivi |
| 13 | Criterio di ordinamento della tabella comparativa e scelta di `MONITOR_METRIC` — sezione 7.5 | Prima degli esperimenti definitivi |

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
