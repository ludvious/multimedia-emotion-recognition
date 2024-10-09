# multimedia-emotion-recognition


#### run applicazione
``flask --app app run``
``flask --app app --debug run``






## aggiornamenti TODO

- upgrade face model utilizzando i face landmark
- fixare webcam accensione con pulsante
- ~~Web app installata  e funzionante su server locale, altrimenti saremmo dipendenti dal sistema operativo~~.
- ~~La app dovrà essere funzionante in real time, poi dipende dal dato... Per riconoscere un volto, dovrà vedere il volto e riconoscerlo~~.
- ~~modifica del app vocale, registra ogni secondo e manda l audio al backend modello per la predizione~~
- unire video face con speech: mettere le funzioni per registrare nella pagina del video face in modo da unire il servizio (capire come sincronizzare i due servizi)
- ~~girare anche in real time.~~
- ~~finire di scaricare audio da youtube e convertirl in spectrogrammi~~
- ~~implementare augmentation audio con overlapping~~
- selezionare audio e quindi spettogrammi buoni e butti quelle che non ti sembrano buone o hanno rumore. Questo per ogni video, per ogni emozione del modello, finché non hai un numero adeguato (possibilmente ma non necessariamente bilanciato tra classi).
- ~~implementare il preprocessing per audio e per generare spettrogrammi e dataset ~~  (manca da testarlo)
- ~~definire modello per speech emotion~~, poi fare training ecc
- ~~implementare il servizio di speech emotion nell app~~
- testare se gli audio selezioni sono buoni e quindi testare il modello
- provare a creare il dataset con audio con aggiunta di quelli overlapping, da li provarli nel modello e vedere come va

- ~~upgrade face recognition model: nuova cnn (anche con augmentation):~~
    - ~~provare augmentation con tensorflow image e vedere se cambia qualcosa sull allenamento con il modello delle facce~~
        ~~(https://www.tensorflow.org/tutorials/images/data_augmentation?hl=it#using_tfimage)~~ 
    - ~~usare keras 2 con il suo preprocessing e augmentation che funziona su quei dati (fatto questo btw)~~

- per entrambi modelli fare valuation e salvare screenshoot per relazione esame, mettere matrice di confusione della evaluation, architettura modello e summary

- il codice può essere consegnato il giorno stesso. Servono codice e documentazione (chiunque deve essere in grado di usare nonché replicare il codice, per cui ben commentato e con una documentazione -sia anche un readme- completa, con link alle fonti di dati, modelli etc).

- L'esame consiste in una presentazione del progetto, per cui serve o una relazione o delle slide che includano tutti punti principali da lasciare a me come progetto a corredo del codice.

- capire cosa ripassare a livello teorico e quello da collegare tra teoria e progetto ai scopi di affective computing