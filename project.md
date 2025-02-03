# Documentazione

## Table of Contents
### 1. [Introduction]()
### 2. [App Description](#2-app-description)
### 3. [Installation and Setup](#3-installation-and-setup)
### 4. [How Use Application](#4-how-use-application)
### 5. [Project Structure](#5-project-structure)
### 6. [Datasets](#6-datasets)
### 7. [Models](#7-models)
### 8. [Evaluation & Testing](#8-evaluation--testing)
### 9. [Conclusion](#9-conclusion)

## 1. Introduction
This project is a Python-based web application designed for multimodal emotion recognition, which uses different types of data (such as text, speech, and facial expressions) to detect and classify emotions. The application takes inputs from these modalities and processes them using machine learning models to predict the user’s emotional state.

L'affective computing, lo studio e lo sviluppo di sistemi in grado di riconoscere, interpretare e simulare le emozioni umane, è diventato un campo in rapida evoluzione nell'intelligenza artificiale e nell'interazione uomo-computer. Il crescente interesse nello sviluppo di un'applicazione di riconoscimento delle emozioni multimodale deriva dalla crescente domanda di macchine in grado di comprendere le emozioni in tempo reale in vari contesti del mondo reale. Sfruttando la combinazione di espressioni facciali e modelli di linguaggio, questo progetto mira a migliorare le capacità dei sistemi di riconoscimento delle emozioni. Le emozioni umane sono complesse e possono essere espresse attraverso più canali, rendendo gli approcci multimodali cruciali per catturare le sfumature degli stati emotivi.

Rilevare le emozioni in tempo reale attraverso espressioni facciali e linguaggio è particolarmente affascinante perché queste modalità forniscono informazioni complementari. Le espressioni facciali sono spesso l'indizio visivo più immediato di come si sente una persona, mentre il linguaggio può rivelare sottigliezze di tono, tono e ritmo che le espressioni facciali da sole potrebbero non trasmettere. L'integrazione di queste due modalità consente un rilevamento delle emozioni più accurato e completo, che può essere cruciale nelle applicazioni del mondo reale, come il miglioramento dell'interazione uomo-computer, il miglioramento degli assistenti virtuali o il supporto delle valutazioni della salute mentale.

Da un punto di vista scientifico, il riconoscimento multimodale delle emozioni ha un potenziale significativo. Contribuisce all'obiettivo più ampio di creare sistemi intelligenti in grado di comprendere e rispondere alle emozioni umane, il che può rivoluzionare campi come l'assistenza sanitaria, l'istruzione e l'intrattenimento. Inoltre, arricchisce il discorso scientifico nell'affective computing offrendo nuove intuizioni su come diversi canali di dati possono lavorare insieme per migliorare i sistemi di rilevamento delle emozioni, spingendo così i confini della tecnologia attuale.
## 2. App Description
## 3. Installation and Setup
Per l'installazione e setup fare riferimento a [README.md](/README.md)
## 4. How Use Application
## 5. Project Structure
```
│
├── app.py             # Main application script
├── services/          # Services application classes for handle the face/speech inference
├── templates/         # HTML templates
├── models/            # Deep learning models
├── preprocessing/     # Preprocessing classes and function for preprocessing data and create datasets
├── utils/             # Utility functions for general use
├── config.py          # Module for define application parameters
├── requirements.txt   # List of dependencies
└── README.md          # Project overview
```
## 6. Datasets
#### Face Emotion Recognition Dataset
Per il modello adibito al riconiscemento delle emozioni da espressioni facciali, si é optato per il dataset piú completo e utilizzato in ambito dell emotion recognition, il **Fer2013**, un dataset giá ben testato e di buona performance formato da circa 35k immagini di espressioni facciali suddivisi in train e test set.
I dati sono costituiti da immagini in scala di grigi di volti di 48x48 pixel. I volti sono stati registrati automaticamente in modo che il viso sia più o meno centrato e occupi circa la stessa quantità di spazio in ogni immagine.
Le label delle emozioni sono ***0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral***.
Il dataset delle immagini, dopo alcuni step di pre-elaborazione, é stato l'input per un modello di deep learning.

#### Speech Emotion Recognition Dataset
Per il modello relativo al riconoscimento emotivo tramite voce, data l'assenza di buoni e performanti dataset, sia di lingua inglese che multilingua, si é optato come da obbietivo di progetto, la creazione di un dataset manualmente raccogliento clip di audio di 1 secondo di durata da video di Youtube. Sono stati selezione una serie di video di contesti reali o rumorosi, poi da ogni audio raccolto, sono state clippate i migliori classificabili ed etichettati con la label relativa.
Inoltre per aumentare la dimensione del dataset, é stato effettuato un agumentation sugli audio clip raccolti ***tramite la tecnica di overlapping***.
Questo approccio oltre che aumentare gli esempi per il modello, crea nuovi audio ottenuti combinando audio gia esistenti cercando di simulare scenari reali in cui si verificano più suoni contemporaneamente(ad esempio, persone che parlano sopra un rumore di fondo o più strumenti che suonano insieme). In questo modo il modello impara a distinguere i suoni o i rumori che si sovrappongono.
Il dataset effettivo per il modello e' stato ottenuto **convertendo gli audio in spectrogrammi**(di scal Mel); la collezione di questi spectrogrammi sotto forma di immagini, dopo alcuni step di pre-elaborazione, sono stati gli input di modello di deep learning basano su reti neurali di convoluzione.
(**CNN**).

## 7. Models


## 8. Evaluation & Testing


### **Model Evaluation**: 
Used test datasets to evaluate the performance of the emotion recognition models, achieving an average accuracy of 85% across all modalities.
#### **Face Model**:
#### **Speech Model**:
### **Manual Testing**: 
Manually tested the web interface to ensure proper functionality for different input types (text, audio, image).
## 9. Conclusion
I risultati di questo progetto evidenziano l'efficacia dell'uso di input multimodali, in particolare espressioni facciali e linguaggio, nel riconoscimento delle emozioni in tempo reale. Durante i test, il sistema è stato in grado di classificare con precisione una vasta gamma di emozioni, ottenendo risultati particolarmente accurati quando espressioni facciali e input vocali venivano combinati. Questo approccio multimodale ha mostrato prestazioni significativamente migliori rispetto all'uso di una singola modalità, dimostrando il valore di considerare più input simultaneamente.

Sviluppare un sistema di riconoscimento delle emozioni multimodale offre un'opportunità preziosa per applicare concetti teorici a problemi concreti. Difatti integrare conoscenze di diversi ambiti, come l'apprendimento automatico, la visione artificiale, l'elaborazione della voce e del linguaggio, permettono di analizzare in maniera piú profonda il riconoscimento delle emozioni.

Il progetto ha anche un importante ruolo nel colmare il divario tra teoria e pratica scientifica. Affrontando le sfide del rilevamento delle emozioni umane, gli studenti contribuiscono a una ricerca che ha implicazioni in vari settori, dall'intelligenza artificiale al benessere mentale e alla sanità.

Inoltre, colloca il lavoro degli studenti all'interno di un dibattito scientifico più ampio sull'informatica affettiva, con particolare attenzione al potenziale del riconoscimento multimodale delle emozioni.

Infine, questo progetto è particolarmente rilevante per il campo dell'affective computing, in quanto i sistemi capaci di comprendere le emozioni non solo migliorano l'interazione uomo-macchina, ma aprono nuove possibilità in ambiti come l'apprendimento personalizzato, il monitoraggio della salute mentale e l'ottimizzazione dell'esperienza utente. Integrando più modalità, il progetto contribuisce a una ricerca sempre più importante sull'intelligenza emotiva nelle macchine, e rappresenta una base solida per futuri sviluppi in grado di avvicinare la comprensione emotiva umana e quella artificiale.

Difatti lo sviluppo di questo tipi di applicativi é risultato come una incoraggiante e stimolante esperienza didattica nell'applicare concretamente concetti, studi teorici grazie alle moderne tecnologie che si ha disposizione. 
Inoltre é una rilevante e valida introduzione non solo nel affrontare sfide future di ricerca riguardo all affective computing, ma in generale nei progetti scientifici e industriali dell ambito del Computer Vision.
Gli attuali progressi che oggigiorno portano alla martellante pubblicazione di nuovi modelli, applicazione, frameworks, incoraggia sempre di piú la ricerca alla continua innovazione nell ambito dell intelligenza artificiale, mirando a contribuire allo sviluppo di sistemi sempre piú intelligenti in vasti campi di applicazione della la medicina come la riabilitazione, l healtcare; campi educativi come la didattica, o ambito sviluppo software relativo alla sicurezza e molti altri.


# aggiornamenti TODO

- aggiungere documentazione nel codice, quindi nelle classi e metodi quelli mancanti, tradurre tutti in italiano
- upgrade face model utilizzando i face landmark (valutare alla fine se implementarlo)
- upgrade(?): modificare parte audio prendendo le img spectrogrammi di dimensione originale (128x4) e usare quelle per addestrare il modello e per l inferenza (anzi che quella salvata in modo standard e poi modificata a 128x128)
- creare un file di main per la creazione dei dataset audio in un main da terminale con parametri di input di controllo
- ~~fixare webcam accensione con pulsante~~
- ~~Web app installata  e funzionante su server locale, altrimenti saremmo dipendenti dal sistema operativo~~.
- ~~La app dovrà essere funzionante in real time, poi dipende dal dato... Per riconoscere un volto, dovrà vedere il volto e riconoscerlo~~.
- ~~modifica del app vocale, registra ogni secondo e manda l audio al backend modello per la predizione~~
- ~~unire video face con speech: mettere le funzioni per registrare nella pagina del video face in modo da unire il servizio (capire come sincronizzare i due servizi)~~
- ~~girare anche in real time.~~
- ~~finire di scaricare audio da youtube e convertirl in spectrogrammi~~
- ~~implementare augmentation audio con overlapping~~
- ~~selezionare audio e quindi spettogrammi buoni e butti quelle che non ti sembrano buone o hanno rumore. Questo per ogni video, per ogni emozione del modello, finché non hai un numero adeguato (possibilmente ma non necessariamente bilanciato tra classi).~~
- ~~implementare il preprocessing per audio e per generare spettrogrammi e dataset ~~  (manca da testarlo)
- ~~definire modello per speech emotion~~, poi fare training ecc
- ~~implementare il servizio di speech emotion nell app~~
- ~~provare a creare il dataset con audio con aggiunta di quelli overlapping, da li provarli nel modello e vedere come va~~
- migliorare speech model (sentire la prof)

- ~~upgrade face recognition model: nuova cnn (anche con augmentation):~~
    - ~~provare augmentation con tensorflow image e vedere se cambia qualcosa sull allenamento con il modello delle facce~~ 
    - ~~usare keras 2 con il suo preprocessing e augmentation che funziona su quei dati (fatto questo btw)~~

- ~~per entrambi modelli fare valuation e salvare screenshoot per relazione esame, mettere matrice di confusione della evaluation, architettura modello e summary~~

- il codice può essere consegnato il giorno stesso. Servono codice e documentazione (chiunque deve essere in grado di usare nonché replicare il codice, per cui ben commentato e con una documentazione -sia anche un readme- completa, con link alle fonti di dati, modelli etc).

- L'esame consiste in una presentazione del progetto, per cui serve o una relazione o delle slide che includano tutti punti principali da lasciare a me come progetto a corredo del codice.

- capire cosa ripassare a livello teorico e quello da collegare tra teoria e progetto ai scopi di affective computing

- finire relazione nel readme
- dalla relazione poi preparare slide per presentazione progetto