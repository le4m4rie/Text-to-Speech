# Zeitserienanalyse
von *Andreas Greiner, Julian Ivanov und Stephan Zahnweh*

## Abstract

***TODO:*** Andreas

## 1 Einleitung / Motivation

***TODO:*** Andreas

## 2 Methoden

Bei der Recherche rund um das Theam Zeitserienanlyse stößt man immer wieder auf eine mögliche Definition für Zeitserien.

!!! info "Definition"

    Die Zeitserienanalyse trägt der Tatsache Rechnung, dass Datenpunkte, die im Laufe der Zeit aufgenommen wurden, eine interne Struktur aufweisen können (wie Autokorrelation, Trend oder saisonale Schwankungen), die berücksichtigt werden sollte.

Damit grenzt sich diese von anderen Datenanalysen ohne Zeitkomponente ab. Im Folgenden werden die internen Strukturen von Zeitsereien näher erläutert.


### 2.1 Merkmale von Zeitserien

Hier werden die Merkmale von Zeitserien (Autokorrelation, Trend und Saison) beschrieben.

#### Autokorrelation

Autokorrelation bezieht sich auf die statistische Beziehung oder Ähnlichkeit zwischen den Werten einer Zeitreihe und ihren verzögerten Versionen. In einer Zeitreihe werden aufeinanderfolgende Beobachtungen in regelmäßigen Zeitintervallen erfasst. Autokorrelation misst die Stärke und Richtung des linearen Zusammenhangs zwischen den Werten einer Zeitreihe und den Werten zu verschiedenen Verzögerungszeiten. Zum veranschaulichen dieser und eiterer Methoden wird ein Beispiel Datensatz (Fig. 1) verwendet. Hierbei handelt es sich um einen Datensatz mit monatlich gemessenen CO2 ausstoß über mehrere Jahre hinweg. 

<figure markdown>
  ![CO2 Data](./img/Zeitserienanalyse/data.png){ width="400" }
  <figcaption>Fig. 1 Beispiel Datensatz</figcaption>
</figure>

Die Autokorrelationsfunktion (ACF) zeigt den Autokorrelationskoeffizient für verschieden Verzögerungen. Er ist ein Maß für die Autokorrelation. Er variiert zwischen -1 und 1. Ein Wert von 1 zeigt eine perfekte positive Autokorrelation an, was bedeutet, dass eine hohe Ähnlichkeit zwischen den Werten der Zeitreihe und ihren verzögerten Versionen besteht. Ein Wert von -1 zeigt eine perfekte negative Autokorrelation an, was darauf hindeutet, dass die Werte der Zeitreihe und ihre verzögerten Versionen genau entgegengesetzte Muster aufweisen. Ein Wert von 0 deutet auf keine Autokorrelation hin, was bedeutet, dass es keine systematische Beziehung zwischen den Werten und ihren Verzögerungen gibt.

<figure markdown>
  ![ACF Plot](./img/Zeitserienanalyse/acf.png){ width="400" }
  <figcaption>Fig. 2 ACF Plot</figcaption>
</figure>

#### Trend

Der Trend bezieht sich auf die langfristige Veränderung des Mittelwerts einer Zeitreihe im Laufe der Zeit. Es ist wichtig den Trend zu bestimmen, da einige Modelle nur mit stationären Zeitserien umgehen können. In der Zeitreihenanalyse bedeutet "stationär", dass die statistischen Eigenschaften einer Zeitreihe im Laufe der Zeit konstant bleiben oder nicht von der Zeit abhängen. Um die Stationarität einer Zeitreihe zu überprüfen, können verschiedene statistische Tests wie der Augmented Dickey-Fuller (ADF)-Test verwendet werden. Dieser Test bewertet die Nullhypothese, dass eine Zeitreihe nicht stationär ist, und liefert Informationen darüber, ob die Zeitreihe einer Transformation bedarf, um stationär zu werden. Für den Beispiel Datensatz (Fig. 1) ergibt sich mit dem ADF-Test ein p-Wert von über 0.9, was einen eindeutigen Trend belegt.

#### Saison

Bei der Saison handelt es sich um wiederkehrende Muster oder periodische Schwankungen in den Daten einer Zeitreihe, die mit bestimmten Zeiträumen zusammenhängen. Diese Muster können tägliche, wöchentliche, monatliche, quartalsweise oder jährliche Perioden umfassen. Für den Beispiel Datensatz (Fig. 1) lässt sich eine Saison Komponente aus dem ACF Plot (Fig.2) sehr gut ablesen. Man erkennt ein wiederholtes lokales Maximum, dass in einem festen Intervall von 12 auftaucht. Dies deutet auf eine Jährliche Periode hin.

<figure markdown>
  ![ACF Plot Season](./img/Zeitserienanalyse/acf_season.png){ width="400" }
  <figcaption>Fig. 3 ACF Plot Saison</figcaption>
</figure>


### 2.2 Anwendungsgebiete

Die Zeitreihenanalyse ist ein leistungsstarkes Werkzeug zur Untersuchung und Extraktion von Informationen aus sequenziellen Daten. Sie findet in verschiedenen Anwendungsgebieten Anwendung, darunter die Analyse der Merkmale, Vorhersage, Glättung und Anomalieerkennung. Jedes dieser Gebiete spielt eine wichtige Rolle bei der Auswertung und Interpretation von Zeitreihendaten.

#### Analyse der Merkmale
Die Analyse der Merkmale befasst sich mit der Untersuchung und Identifizierung von Mustern, Trends, Saisonalität und anderen Charakteristika in einer Zeitreihe. Sie hilft dabei, grundlegende Informationen über die Daten zu gewinnen und Einblicke in vergangene Muster und Veränderungen zu gewinnen. Die Werkzeuge zur Merkmalsyanlyse wurden bereits im vorherigen Kapitel ausführlich erläutert.

#### Vorhersage
Die Vorhersage ist ein wichtiger Anwendungsbereich der Zeitreihenanalyse. Sie befasst sich mit der Schätzung zukünftiger Werte oder Ereignisse aufgrund vergangener Daten. Durch die Analyse von Trends, saisonalen Mustern und anderen Zeitreiheneigenschaften können Modelle entwickelt werden, um Vorhersagen für die Zukunft zu treffen. Methoden wie ARIMA (Autoregressive Integrated Moving Average), Exponential Smoothing und Machine-Learning-Algorithmen werden häufig verwendet, um Prognosen zu generieren. Diese Ansätze werden in den nächsten Kapiteln genauer betrachtet.

#### Glättung
Die Glättung bezieht sich auf den Prozess der Reduzierung von Rauschen, Unregelmäßigkeiten oder kurzfristigen Schwankungen in einer Zeitreihe, um den zugrunde liegenden Trend oder das Muster deutlicher sichtbar zu machen. Durch die Anwendung von Glättungstechniken können saisonale Effekte, Ausreißer und zufällige Schwankungen geglättet werden, um den Trend oder das langfristige Verhalten der Daten zu analysieren.

#### Anomalieerkennung
Die Anomalieerkennung in Zeitreihen beschäftigt sich mit der Identifizierung von ungewöhnlichen oder abweichenden Mustern in den Daten. Anomalien können Ausreißer, unerwartete Veränderungen, Ausfälle oder andere abnormale Ereignisse sein, die in der Zeitreihe auftreten. Die Zeitreihenanalyse kann verwendet werden, um solche Anomalien zu erkennen und zu charakterisieren, indem statistische Methoden, Mustererkennungsalgorithmen und maschinelles Lernen eingesetzt werden.

### 2.3 Klassische Ansätze

Bei der Zeitreihenanalyse gibt es verschiedene klassische Modelle, die zur Modellierung und Vorhersage von Zeitreihendaten verwendet werden. Diese Modelle basieren auf statistischen Methoden und Annahmen über die Struktur der Daten. Holt-Winters exponentielles Glätten und Box-Jenkins SARIMA Modelle sind nach deren Erfindern benannt. Beide werden in den nächsten Abschnitten vorgestellt.

#### Exponentielles Glätten

Exponentielles Glätten ist eine gängige Methode in der Zeitreihenanalyse, um saisonale Muster, Trends und kurzfristige Schwankungen zu reduzieren und den zugrunde liegenden Trend oder das Muster einer Zeitreihe deutlicher zu erkennen. Es handelt sich um eine einfache und effektive Methode, die auf der Annahme basiert, dass aktuelle Werte einer Zeitreihe stärker gewichtet werden sollten als vergangene Werte.

Der Hauptgedanke beim exponentiellen Glätten besteht darin, jedem Datenpunkt ein Gewicht zuzuweisen, wobei die Gewichte exponentiell abnehmen, je weiter der Datenpunkt in der Vergangenheit liegt. Das Gewichtungsschema wird durch einen Glättungsfaktor (auch als Glättungsparameter oder Alpha-Faktor bezeichnet) gesteuert, der zwischen 0 und 1 liegt.

Der Prozess des **einfachen exponentiellen Glättens** kann in mehreren Schritten zusammengefasst werden:

- Initialisierung:  
  Der erste geschätzte Wert wird als der erste Beobachtungswert der Zeitreihe angenommen.

- Glättungsschritt:  
  Für die folgenden Beobachtungen wird der geschätzte Wert durch die Kombination des aktuellen Beobachtungswerts und des vorherigen geschätzten Werts berechnet. Dies wird durch die Formel dargestellt:  
    
    geschätzter Wert = (1 - a) * aktueller Wert + a * vorheriger geschätzter Wert  
    
    Dabei ist "a" der Glättungsfaktor, der angibt, wie stark der aktuelle Wert gewichtet werden soll. Ein kleinerer Wert von "a" gibt den vergangenen Werten ein höheres Gewicht, während ein größerer Wert von "a" den aktuellen Wert stärker berücksichtigt.

- Wiederholung:  
  Dieser Glättungsschritt wird für jede Beobachtung in der Zeitreihe wiederholt, wobei der geschätzte Wert bei jedem Schritt aktualisiert wird.

<figure markdown>
  ![EXP 1](./img/Zeitserienanalyse/exp1.png){ width="800" }
  <figcaption>Fig. 4 Einfaches Exponentielles Glätten</figcaption>
</figure>

Die linke Grafik zeigt die geglättete Zeitserie und die Rechte das Ergebniss einer Vorhersage auf den in Test- und Trainingsdaten aufgeteilten Beispieldatensatz (Fig. 1). Die Vorhersagen sind nicht sonderlich gut, da sowohl Trand als auch Saison Komponenten nicht berücksichtigt werden.

Um den Trend zu erfassen, wird das Modell um eine Trendkomponente erwitert. Hier handelt es sich um das **zweifache exponentielle Glätten**. 

<figure markdown>
  ![EXP 2](./img/Zeitserienanalyse/exp2.png){ width="800" }
  <figcaption>Fig. 5 Zweifaches Exponentielles Glätten</figcaption>
</figure>

Hier wird nun der Trend mit berücksichtigt. Man sieht bei der Glättung, dass der Trend erkannt wird, jedoch eine Änderung im Trend erst verzögert wahrgenommen wird. Auch die Vorhersagen können nur den aktuellen Trend weiterführen und keine Änderungen vorhersagen. Das liegt daran, dass die Saison Komponente feht um brauchbare Vorhersagen treffen zu können.

Beim **dreifachen exponentiellen Glätten** wird auch diese Berücksichtigt.

<figure markdown>
  ![EXP 3](./img/Zeitserienanalyse/exp3.png){ width="800" }
  <figcaption>Fig. 6 Dreifaches Exponentielles Glätten</figcaption>
</figure>

Das Ergebnis liefert eine gute Glättung und brauchbare Vorhersagen. Es ist wichtig zu beachten, dass diese Methode für Zeitreihen geeignet ist, die keine komplexe Struktur oder starke saisonale Komponenten aufweisen. Für Zeitreihen mit starken saisonalen Mustern oder anderen komplexen Eigenschaften können fortgeschrittenere Modelle wie ARIMA oder saisonale ARIMA (SARIMA) verwendet werden. Diese werden im nächsten Abschnitt erläutert.

#### SARIMA

Das saisonale ARIMA-Modell (SARIMA) ist eine Erweiterung des ARIMA-Modells (Autoregressive Integrated Moving Average) und wird verwendet, um Zeitreihendaten mit saisonalen Mustern zu modellieren und Vorhersagen zu generieren. SARIMA kombiniert Autoregression (AR), Integration (I) und Moving Average (MA) mit saisonalen Komponenten.

Das SARIMA-Modell besteht aus mehreren Parametern, die seine Eigenschaften definieren:

- Autoregressive (AR)-Komponente:  
  Die AR-Komponente des SARIMA-Modells berücksichtigt die Abhängigkeit des aktuellen Werts einer Zeitreihe von vergangenen Werten. Der Parameter p gibt die Anzahl der vorherigen Werte an, die berücksichtigt werden sollen.

- Differenzierung (I):  
  Die Differenzierungskomponente des SARIMA-Modells wird verwendet, um die Zeitreihe zu stationarisieren. Stationarität bedeutet, dass der Mittelwert und die Varianz der Daten über die Zeit konstant bleiben. Der Parameter d gibt die Anzahl der Differenzierungen an, die erforderlich sind, um die Stationarität zu erreichen.

- Moving Average (MA)-Komponente:  
  Die MA-Komponente des SARIMA-Modells bezieht sich auf den Einfluss der vorherigen Fehler auf den aktuellen Wert einer Zeitreihe. Der Parameter q gibt die Anzahl der vorherigen Fehler an, die berücksichtigt werden sollen.

- Saisonale AR-Komponente:  
  Die saisonale AR-Komponente berücksichtigt die saisonale Abhängigkeit einer Zeitreihe. Sie bezieht sich auf die Abhängigkeit des aktuellen Werts von vergangenen Werten, die eine bestimmte Anzahl von Zeitschritten vor der aktuellen Periode liegen. Der Parameter P gibt die Anzahl der saisonalen AR-Terme an.

- Saisonale Differenzierung:  
  Die saisonale Differenzierung wird verwendet, um die saisonale Komponente der Zeitreihe zu entfernen und die Daten zu stationarisieren. Der Parameter D gibt die Anzahl der saisonalen Differenzierungen an.

- Saisonale MA-Komponente:  
  Die saisonale MA-Komponente bezieht sich auf den Einfluss der vergangenen saisonalen Fehler auf den aktuellen Wert einer Zeitreihe. Der Parameter Q gibt die Anzahl der saisonalen MA-Terme an.

- Saisonaler Index:  
  Der Saison Index m gibt die länge einer Periode an.

Zusammen definieren diese Parameter die Struktur des SARIMA-Modells. 

    SARIMA(p,d,q)(P,D,Q,m)

Durch die Schätzung dieser Parameter und die Anpassung des Modells an die Daten können Vorhersagen für zukünftige Werte der Zeitreihe generiert werden. Dies ermöglicht die Berücksichtigung von saisonalen Mustern und den Einfluss vergangener Werte und Fehler auf die Vorhersagen.

<figure markdown>
  ![SARIMA](./img/Zeitserienanalyse/sarima.png){ width="400" }
  <figcaption>Fig. 7 SARIMA</figcaption>
</figure>

Die Grafik zeigt eine Vorhersage mit Hilfe des SARIMA Modells für den Beispiel Datensatz (Fig. 1). Die Ergebnisse sind sehr gut, da es sich um eine nicht sonderlich komplexe Zeitserie handelt. Obwohl SARIMA-Modelle für die Modellierung von einfachen Zeitreihen mit saisonalen Mustern recht nützlich sind, haben sie einige Nachteile beim modellieren komlexer Strukturen. Hier haben moderne Ansätze weitaus besser Lösungen um komplexe Muster zu erfassen. 

### 2.4 Moderne Ansätze

#### RNN

#### LSTM

***TODO:*** Stephan

## 3 Anwendungen
***TODO:*** Julian

## 4 Fazit
***TODO:*** Julian

## 5 Weiterführendes Material

### Podcast
Hier Link zum Podcast.

### Talk
Hier einfach Youtube oder THD System embedden.

### Demo
Hier Link zum Demo Video + Link zum GIT Repository mit dem Demo Code.


## 6 Literaturliste
Hier können Sie auf weiterführende Literatur verlinken. 

***TODO:*** Stephan
