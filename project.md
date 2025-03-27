# Documentazione

## Table of Contents
### Introductione
### Installation and Setup
### Datasets
### Conclusioni
### Vantaggi di Applicazioni Multimodale e Casi D'uso

## Introduzione
L'affective computing, lo studio e lo sviluppo di sistemi in grado di riconoscere, interpretare e simulare le emozioni umane, è diventato un campo in rapida evoluzione nell'intelligenza artificiale e nell'interazione uomo-computer. Il crescente interesse nello sviluppo di un'applicazione di riconoscimento delle emozioni multimodale deriva dalla crescente domanda di macchine in grado di comprendere le emozioni in tempo reale in vari contesti del mondo reale. Sfruttando la combinazione di espressioni facciali e modelli di linguaggio, questo progetto mira a migliorare le capacità dei sistemi di riconoscimento delle emozioni. Le emozioni umane sono complesse e possono essere espresse attraverso più canali, rendendo gli approcci multimodali cruciali per catturare le sfumature degli stati emotivi.
Rilevare le emozioni in tempo reale attraverso espressioni facciali e linguaggio è particolarmente affascinante perché queste modalità forniscono informazioni complementari. Le espressioni facciali sono spesso l'indizio visivo più immediato di come si sente una persona, mentre il linguaggio può rivelare sottigliezze di tono, tono e ritmo che le espressioni facciali da sole potrebbero non trasmettere. L'integrazione di queste due modalità consente un rilevamento delle emozioni più accurato e completo, che può essere cruciale nelle applicazioni del mondo reale, come il miglioramento dell'interazione uomo-computer, il miglioramento degli assistenti virtuali o il supporto delle valutazioni della salute mentale.
Il riconoscimento multimodale delle emozioni è una branca dell’affective computing che combina diverse modalità di input (come espressioni facciali, linguaggio, tono di voce, gesti e segnali fisiologici) per identificare in modo più accurato lo stato emotivo di una persona.
A differenza dei sistemi tradizionali, che si basano su un'unica fonte di dati (es. solo il volto o solo il testo), l’approccio multimodale integra più canali comunicativi, migliorando l’affidabilità e la profondità dell’analisi emotiva.
Da un punto di vista scientifico, il riconoscimento multimodale delle emozioni ha un potenziale significativo. Contribuisce all'obiettivo più ampio di creare sistemi intelligenti in grado di comprendere e rispondere alle emozioni umane, il che può rivoluzionare campi come l'assistenza sanitaria, l'istruzione e l'intrattenimento. Inoltre, arricchisce il discorso scientifico nell'affective computing offrendo nuove intuizioni su come diversi canali di dati possono lavorare insieme per migliorare i sistemi di rilevamento delle emozioni, spingendo così i confini della tecnologia attuale.

## Installation and Setup
Per l'installazione e setup fare riferimento a [README.md](/README.md)

## Datasets
#### Face Emotion Recognition Dataset
Per il modello adibito al riconiscemento delle emozioni da espressioni facciali, si é optato per il dataset piú utilizzato in ambito dell emotion recognition, il **Fer2013**, un dataset giá ben testato e di buona performance formato da circa 35k immagini di espressioni facciali suddivisi in train e test set.
I dati sono costituiti da immagini in scala di grigi di volti di 48x48 pixel. I volti sono stati registrati automaticamente in modo che il viso sia più o meno centrato e occupi circa la stessa quantità di spazio in ogni immagine.
Le label delle emozioni sono ***0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral***.
Il dataset delle immagini, dopo alcuni step di pre-elaborazione, é stato l'input per un modello di deep learning.

#### Speech Emotion Recognition Dataset
Per il modello relativo al riconoscimento emotivo vocale, data l'assenza di buoni e performanti dataset, sia di lingua inglese che multilingua, si é optata la creazione di un dataset custom raccogliento clip di audio di 1 secondo di durata da video di Youtube. Sono stati selezione una serie di video di contesti reali o rumorosi, poi da ogni audio raccolto, sono state clippate i migliori classificabili ed etichettati con la label relativa.
Inoltre per aumentare la dimensione del dataset, é stato effettuato un agumentation sugli audio clip raccolti ***tramite la tecnica di overlapping***.
Questo approccio oltre che aumentare gli esempi per il modello, crea nuovi audio ottenuti combinando audio gia esistenti cercando di simulare scenari reali in cui si verificano più suoni contemporaneamente(ad esempio, persone che parlano sopra un rumore di fondo o più strumenti che suonano insieme). In questo modo il modello impara a distinguere i suoni o i rumori che si sovrappongono.
Il dataset effettivo per il modello e' stato ottenuto **convertendo gli audio in spectrogrammi**(di scal Mel); la collezione di questi spectrogrammi sotto forma di immagini, dopo alcuni step di pre-elaborazione, sono stati gli input di modello di deep learning basano su reti neurali di convoluzione.
(**CNN**).

## Conclusioni

Sviluppare un sistema di riconoscimento delle emozioni multimodale offre un'opportunità preziosa per applicare concetti teorici a problemi concreti. Difatti integrare conoscenze di diversi ambiti, come l'apprendimento automatico, la visione artificiale, l'elaborazione della voce e del linguaggio, permettono di analizzare in maniera piú profonda il riconoscimento delle emozioni.

Il progetto ha anche un importante ruolo nel colmare il divario tra teoria e pratica scientifica. Affrontando le sfide del rilevamento delle emozioni umane, gli studenti contribuiscono a una ricerca che ha implicazioni in vari settori, dall'intelligenza artificiale al benessere mentale e alla sanità.

Infine, questo progetto è particolarmente rilevante per il campo dell'affective computing, in quanto i sistemi capaci di comprendere le emozioni non solo migliorano l'interazione uomo-macchina, ma aprono nuove possibilità in ambiti come l'apprendimento personalizzato, il monitoraggio della salute mentale e l'ottimizzazione dell'esperienza utente. Integrando più modalità, il progetto contribuisce a una ricerca sempre più importante sull'intelligenza emotiva nelle macchine, e rappresenta una base solida per futuri sviluppi in grado di avvicinare la comprensione emotiva umana e quella artificiale.

Difatti lo sviluppo di questo tipi di applicativi é risultato come una incoraggiante e stimolante esperienza didattica nell'applicare concretamente concetti, studi teorici grazie alle moderne tecnologie che si ha disposizione. 
Inoltre é una rilevante e valida introduzione non solo nel affrontare sfide future di ricerca riguardo all affective computing, ma in generale nei progetti scientifici e industriali dell ambito del Computer Vision.
Gli attuali progressi che oggigiorno portano alla continua pubblicazione di nuovi modelli, applicazione, frameworks, incoraggia sempre di piú la ricerca alla continua innovazione nell ambito dell intelligenza artificiale, mirando a contribuire allo sviluppo di sistemi sempre piú intelligenti in vasti campi di applicazione della la medicina come la riabilitazione, l healtcare; campi educativi come la didattica, o ambito sviluppo software relativo alla sicurezza e molti altri.

## Vantaggi di Applicazioni Multimodale e Casi D'uso
Il riconoscimento emotivo multimodale in tempo reale è cruciale perché permette alle macchine di interagire in modo dinamico e contestuale con gli esseri umani, adattandosi alle loro emozioni mentre accadono. Questo è fondamentale per:
- Migliorare l’interazione uomo-macchina
- Modificare il comportamento in base all’umore dell’utente (es. un chatbot che diventa più paziente se rileva frustrazione).
- Ridurre errori e incomprensioni
- Un sistema che analizza solo il testo potrebbe fraintendere il sarcasmo o l’ironia, mentre l’aggiunta di tono di voce ed espressioni facciali aumenta l’accuratezza.
- Abilitare risposte immediate in scenari critici

#### Casi d'Uso

    1. Salute Mentale e Telemedicina
    Rilevamento precoce di depressione e ansia: Analizzando voce, volto e linguaggio in sessioni terapeutiche online.
    Monitoraggio di pazienti anziani o con disturbi neurologici (es. Alzheimer) attraverso emozioni e cambiamenti nel linguaggio.
    Supporto a terapisti e psichiatri con report automatizzati sullo stato emotivo del paziente.

    2. Educazione e E-Learning
    Tutoraggio adattivo in tempo reale:
    Se uno studente mostra confusione, il sistema può ripetere un concetto in modo diverso.
    Se rileva noia, può introdurre elementi più coinvolgenti.
    Analisi del coinvolgimento in aule virtuali per migliorare la didattica.

    3. Customer Experience e Servizi
    Chatbot e call center empatici:
    Riconoscono la frustrazione del cliente e trasferiscono la chiamata a un operatore umano.
    Adattano tono e risposte per migliorare la soddisfazione.
    Analisi del feedback emotivo su prodotti/servizi attraverso video-recensioni e interazioni vocali.

    4. Sicurezza e Giustizia
    Rilevamento di menzogne o stress in interrogatori (usato in alcuni aeroporti e forze dell’ordine).
    Monitoraggio di operatori in contesti ad alto stress (es. controllori di volo, militari) per prevenire errori.

    5. Guida Autonoma e Trasporti
    Rilevamento di stanchezza o distrazione del guidatore:
    Se il sistema capta sonnolenza (sbadigli, chiusura occhi, tono di voce rallentato), attiva allarmi o guida autonoma.
    Miglioramento dell’esperienza passeggeri in auto a guida autonoma (es. regolazione musica/clima in base all’umore).

    6. Intrattenimento e Gaming
    Adattamento dinamico di storie e gameplay in base alle emozioni del giocatore (es. un horror game che diventa più pauroso se rileva paura).
    Film e pubblicità interattivi che cambiano in base alle reazioni emotive del pubblico.

    7. Robotica Sociale e Assistenza
    Robot caregiver per anziani che riconoscono solitudine o disagio e reagiscono con compagnia o allertano familiari.
    Bambini con autismo: Aiuto nell’interpretare emozioni altrui attraverso feedback in tempo reale.