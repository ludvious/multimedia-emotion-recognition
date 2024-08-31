# multimedia-emotion-recognition


#### fedora troubleshooting
for install PyAudio:
- ``sudo dnf install portaudio-devel python3-devel``

### TODO
provare augmentation con tensorflow image e vedere se cambia qualcosa sull allenamento con il modello delle facce
(https://www.tensorflow.org/tutorials/images/data_augmentation?hl=it#using_tfimage)


## TODO ESAME

- chiedere alla prof:
    - se l app deve essere real time oppure deve testare solo l emozioni caricando i file
    - se va bene utilizzare il dataset dello speech con piu il pre training, oppure se per lo scopo del progetto devo fare dataset a mano prendendo i campioni da youtube
    - dire che fare il dataset prendendo campioni da youtube é insano
    - cosa consegnare precisamente all esame e o prima

- implementare modello speech recognition
    - prima prova usando dataset emozionalmente
    - fare traning e fare test sui dati
    - (plus) provare test con realtime che rappresenta una features aggiuntiva per l app

- upgrade face recognition model: provare minix piu grande e vedere se funziona (anche con augmentation)
- alternativa limite, usare keras 2 con il suo preprocessing e augmentation che funziona su quei dati

- finire di implementare l app
- breve documentazione concisa dell app

- capire cosa ripassare a livello teorico e quello da collegare tra teoria e progetto ai scopi di affective computing;



## aggiornamenti TODO

- Web app installata  e funzionante su server locale, altrimenti saremmo dipendenti dal sistema operativo.
La app dovrà essere funzionante in real time, poi dipende dal dato... Per riconoscere un volto, dovrà vedere il volto e riconoscerlo. Se mi fai un pulsante per caricare degli esempi di foto da disco e mi mostri che li riconosce come esempio di precisione (perché ci sta che riconosca meglio un volto di un altro) va bene, ma deve girare anche in real time.

- fare dataset speech emotion: Per la raccolta dati, usa direttamente la label nella ricerca dei video (ad esempio "Joy"). Se trovi troppi video non correnti, raffina la ricerca (joyful girl, laughing man...), poi selezioni le clip funzionali: il modo più rapido è trasformarle in immagini (lo sono già, in numero pari alla frequenza.... un girato a 1/30 avrà 30 immagini al secondo) e selezioni e butti quelle che non ti sembrano buone o hanno rumore. Dal sistema operativo, mettendo la visualizzazione con icone grandi, si fa al volo col mouse. Questo per.ogni video, per ogni emozione del modello, finché non hai un numero adeguato (possibilmente ma non necessariamente bilanciato tra classi).

- il codice può essere consegnato il giorno stesso. Servono codice e documentazione (chiunque deve essere in grado di usare nonché replicare il codice, per cui ben commentato e con una documentazione -sia anche un readme- completa, con link alle fonti di dati, modelli etc). 
- L'esame consiste in una presentazione del progetto, per cui serve o una relazione o delle slide che includano tutti punti principali da lasciare a me come progetto a corredo del codice.
