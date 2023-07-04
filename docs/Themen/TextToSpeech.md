# Text-to-Speech
von *Lea Wagner, Michael Schmidbauer*

## Abstract
In diesem Blogbeitrag widmen wir uns dem Thema Text-to-Speech (TTS). Bei Text-to-Speech handelt es sich um eine vielseitig einsetzbare und faszinierende Technologie zur computergestützten Generierung von natürlicher und menschenähnlicher Sprache.

In dem Beitrag betrachten wir einerseits die vielfältigen Einsatzmöglichkeiten von TTS wie beispielsweise die Unterstützung von sehbehinderten Personen. Andererseits beleuchten wir den Aufbau und die Funktionsweise dieser Technologie genauer. Für die Erklärung betrachten wir die beiden Modelle Tacotron 2 und Vall-E.

## Einleitung / Motivation
Die Text-to-Speech-Technologie hat, wie viele Anwendungen der Künstlichen Intelligenz, in den letzten Jahren erhebliche Fortschritte gemacht. Ihr Einfluss auf verschiedene Bereiche unseres täglichen Lebens ist beachtlich. Prominente Beispiele für ihre Anwendung wären Sprachassistenten wie beispielsweise Siri oder Alexa, aber sie waren längst nicht die ersten. Schon lange wird die TTS-Technologie beispielweise in Navigationsgeräten verwendet.

Mit Hilfe dieser Technologie wird es Maschinen ermöglicht Text in natürliche Sprache umzuwandeln. Dabei wurde erst mit Einsatz von Künstlicher Intelligenz ein nahezu menschlicher Klang ermöglicht. Doch gibt es noch immer Stolpersteine wie beispielsweise Dialekte, die von den Systemen heute noch nicht perfekt erzeugt werden können.

Nichtsdestotrotz ist es unbestreitbar, dass die Text-to-Speech-Technologie ein wesentlicher Bestandteil der heutigen Mensch-Computer-Interaktion geworden ist. Sie erhöht die Sicherheit im Straßenverkehr oder ermöglicht es seebehinderten Personen besser an der zunehmend digitalen Welt teilzunehmen.

Gerade für diese Personengruppe kann die TTS-Technologie einen deutlich besseren Zugang zu digitalen Informationen ermöglichen. Mit TTS können ihnen digitale Inhalte vorgelesen werden und damit der Zugang zum digitalen Leben erleichtert werden.

Darüber hinaus bietet die Integration der Text-to-Speech-Technologie für die Unterhaltungsindustrie viele neue Möglichkeiten. Hörbücher, Podcasts, Videospiele, Filme und Serien sind einige der Einsatzgebiete, bei denen TTS-Systeme heute oder in naher Zukunft eine große Rolle spielen können. So wäre es möglich ein deutlich immersiveres Spieleerlebnis zu erzeugen oder eine Sprachfassung eines Films für eine Sprache zu generieren, bei der heutzutage der Markt zu klein wäre, um die Kosten zu rechtfertigen.

Durch die ganzen Möglichkeiten sollte man allerdings nicht außeracht lassen, welche potenziellen Gefahren diese Technologie mit sich bringen kann. Gerade mit dem Fortschreiten von Zero-Shot-Systemen, die mit nur wenigen Sekunden Audio eine Stimme nachahmen können, entsteht auch ein großes Gefährdungspotential, dass von Identitätsdiebstahl bis hin zu politischer Einflussnahme reicht.

In diesem Block werden wir und mit den technischen Grundlagen von Text-to-Speech-Systemen befassen. Außerdem werden wir zwei moderne Systeme genauer betrachten.

## Stand der Forschung

Hier möchten wir zwei unterschiedliche Text-to-Speech Systeme vorstellen.

End-to-end Text-to-Speech:
Zunächst stellen wir Tacotron 2 vor, ein end-to-end neurales Text-to-Speech System, welches im Auftrag von Google entwickelt wurde. Die Technologie wandelt einen vorgegebenen Text in natürlich-klingende Sprache um. Systeme wie Tacotron 2 sind entscheidend für Anwendungen wie sprachgesteuerte Systeme und assistive Technologien. Tacotron 2 sollte beispielsweise in der Zukunft für die Sprachsynthese in Google Translate und Google Home verwendet werden. Aber auch für die barrierefreiheit für Sehbehinderte Menschen stellen solche Systeme eine Hilfestellung und Bereicherung der Lebensqualität dar.
Die Besonderheit von Tacotron 2 liegt in der natürlich klingenden Sprache. Viele TTS Systeme weisen Probleme bei der Betonung, Prosodie und Semantik auf. Die Beispiele von Tacotron 2 weisen eine nahezu menschlich klingende Sprachausgabe auf. Durch die Anwendung von vielschichtigen Deep Learning Algorithmen kann das System komplexe Muster in der Sprache erfassen und so eine möglichst natürliche Sprache erzeugen.
Bei einer Bewertung menschlicher Zuhörer erzielte das System einen MOS (mean opinion score) von 4.53 verglichen mit 4.58 für professionell aufgenommene Audios. MOS bedeutet, dass eine bestimmte Anzahl von Menschen bewertet, wie gut sich die Audio anhört.

Erkennen Sie, welche Audio von Tacotron 2 stammt und welche von einem echten Sprecher?
<audio controls>
  <source src= "https://google.github.io/tacotron/publications/tacotron2/demos/lipstick_gt.wav" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

<audio controls>
  <source src= "https://google.github.io/tacotron/publications/tacotron2/demos/lipstick_gen.wav" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

Funktionsweise von Tacotron 2:
Tacotron 2 besteht aus zwei Hauptkomponenten, nämlich einem Encoder und einem Decoder. Der Encoder wandelt eine Textsequenz in eine Hidden Feature Repräsentation um, während der Decoder basierend auf der enkodierten Sequenz Frame für Frame ein Mel Spektrogramm erstellt. 
Als Input erhält das System einen beliebigen Text. Daraus werden Character Embeddings generiert. Hier wurde zuvor ein Modell trainiert, welches jedem Buchstaben einen Vektor zuweist. In diesem Fall hat ein Vektor 512 Dimensionen, in dem die sprachlichen Eigenschaften dieses Buchstabens festgehalten werden. Diese Vektoren werden anschließend in einer Matrix zusammengefasst und an ein 3-schichtiges Convolutional Neural Network übergeben. Dieses CNN ist darauf ausgelegt, n-grams mit längerfristigem Kontext zu modellieren. Dieser Output geht dann weiter an ein bi-directional LSTM. In einem normalen LSTM wird ein Zustand zum Zeitpunkt t berechnet auf dem Input und auf dem Zustand des vorherigen Zeitpunktes t-1. In diesem LSTM werden die Daten vorwärts und rückwärts verarbeitet, wobei kontextuelle Informationen sehr gut erfasst werden können. Die Ausgabe dieses LSTM stellt die Encoder Ausgabe dar, welche jetzt high-level Informationen über die Textsequenz enthält.
Das (location sensitive) Attention Network nimmt den Output des LSTM und einen Output des Decoder Teils zum Zeitpunkt t-1, um relevante Informationen zu erhalten, mit welchen dann die Vorhersage zum Zeitpunkt t erstellt wird. Diese wird an ein 2-schichtiges LSTM übergeben. Für jeden Zeitschritt dieses LSTM wird ein Mel-Spektrogramm-Vektor vorhergesagt. Diese Ausgabe geht an eine lineare Projektion. Diese wird einmal verwendet für den Stop Token. Hier wird die Wahrscheinlichkeit berechnet, dass die Output Sequenz fertig generiert wurde. So kann das Modell dynamisch bestimmen, wann die Generierung beendet wird und ist nicht an eine feste Vorgabe von Iterationen gebunden.
Die Ausgabe der linearen Projektion geht außerdem an ein Pre-Net, welches seinen Output wieder an das 2-schichtige LSTM übergibt, um den nächsten Frame vorherzusagen. Das 5-schichtige Post-Net am Schluss berechnet einen bestimmten Restwert, welcher für ein glattes Mel Spektrogramm verantwortlich ist. Sobald alle Frames durchlaufen wurden, enthält man dann ein komplettes Mel Spektrogramm. Das Mel Spektrogramm wird dann an das WaveNet gegeben, welches als Vocoder agiert und eine Wellenform synthetisiert.

Limitationen von Tacotron 2:
Obwohl Sprachmodelle wie Tacotron 2 erstaunliche Fortschritte im Bereich der Aussprache gemacht haben, zeigen sich hier immer wieder Probleme auf. Schwierigkeiten bei der Aussprache von Wörtern mit komplexer Phonologie oder ungewöhnlicher Betonung bleiben auch bei modernen Text-to-Speech Systemen wie Tacotron 2 bestehen. 
Ein weiterer Punkt ist die Generierung von Audios in Echtzeit. Da die Text-to-Speech Synthese des Systemns auf einer komplexen Architektur mit vielen Schichten beruht, ist eine Generierung in Echtzeit derzeit noch nicht möglich. Deshalb wird dieses System momentan auch nicht für die Sprachsynthese in Google Translate und Google Home benutzt.
Darüber hinaus ist es bisher nicht möglich, die Emotionen der generierten Sprache gezielt zu steuern. Obwohl Tacotron 2 in der Lage ist, natürliche Sprachausgauben zu erzeugen, fehlt die Fähigkeit, die emotionale Ausrucksweise bewusst und gezielt zu beeinflussen. Dies stellt jedoch einen eigenen Bereich der Text-to-Speech Forschung dar. 

Zero-Shot Text-to-Speech:
Das zweite System, das wir vorstellen möchten, nennt sich VALL-E. VALL-E ist ein Zero-Shot TTS System, welches 2023 von Microsoft vorgestellt wurde. Das System kann basierend auf einem Text Prompt und einem Audio Prompt einen Text in Sprache mit der Stimme des Audio Prompts umwandeln. Das bedeutet, VALL-E kann Stimmen imitieren, welche nicht in den Trainingsdaten vorkommen. Auch die akustische Umgebung kann berücksichtigt werden. Wenn der Audio Prompt sich beispielsweise anhört, als würde die Stimme aus einem Telefon kommen, kann VALL-E auch das imitiere. Die Trainingsdaten stammen aus dem LibriLight Datensatz von Meta und enthalten insgesamt 60K Stunden Audio Material, welches größtenteils aus Hörbüchern stammt. Dadurch kann ein System wie VALL-E in Zukunft Anwendung in der Welt der Podcasts und Hörbücher finden.
Die Bewertung der von VALL-E generierten Audios erzielte sogar leicht bessere Ergebnisse als die Ground Truths. 
Die Besonderheit des VALL-E Systems ist der extrem kurze benötigte Audio Input. Während das Vorgänger System noch einen Input von 30 Minuten benötigte, benötigt VALL-E lediglich 3 Sekunden. Durch diese erhebliche Verbesserung entsteht nicht nur eine vereinfachte Anwendung, sondern auch ein vergrößertes Missbrauchspotzenzial. Darauf wird im Abschnitt Limitationen/Ethik nochmal genauer eingegangen.

Funktionsweise von VALL-E:
Ähnlich wie Tacotron 2, nutzt VALL-E eine Encoder-Decoder-Architektur. Es gibt zwei Inputs, den Text Prompt und den Acoustic Prompt. Der Text Prompt wird zunächst in Phoneme und dann in entsprechende Embeddings umgewandelt. Der Audio Prompt geht an den Encoder. Hierbei handelt es sich um den Audio Codec Encoder von Facebook Research. Dieser stellt das “Arbeitstier” hinter VALL-E dar und hat nochmal einen eigenen Encoder und Decoder. Der Encoder nimmt die Wellenform und führt eine Convolution durch für Downsampling. Darauffolgend wird ein LSTM genutzt für die Sequenz Modellierung. Das Ergebnis dieses Encoders ist eine kompaktere Repräsentation mit 75 beziehungsweise 100 latenten Zeitschritten im Vergleich zu 24.000 beziehungsweise 48.000 im Input. Der Decoder ist eine gespiegelte Form des Encoders, welcher wieder ein Upsampling durchführt und daraus eine Wellenform erzeugt. Dazwischen befindet sich der Quantizer. 
Für diesen gibt es 8 sogenannte Codebooks. Codebooks sind Dictionaries gefüllt mit Vektoren, woraus sich 1024 Einträge ergeben. Der Input Vektor wird repräsentiert, indem er auf den ähnlichsten Vektor im Codebook gemapt wird. Diese Ähnlichkeit wird gemessen mit dem euklidischen Abstand. Dadurch gehen Informationen verloren, welche man aber gerne erhalten möchte. Mit Hilfe der Residual Vector Quantization (RVQ) wird der Restwert berechnet. Dieser wird dann auf einen weiteren Vektor im Codebook gemapt. Die finale Repräsentation ist eine Liste der Indexe, auf die die Vektoren gemapt wurden. 
Sobald der Audio Codec Encoder seine Arbeit erledigt hat, wird die Repräsentation an den Decoder von VALL-E übergeben. Dieser besteht aus einem Non-Auto-Regressive (NAR) und aus einem Auto-Regressive (AR) Decoder. Der AR Decoder ist dafür verantwortlich, die Input Daten des ersten Codebooks zu verarbeiten. Der NAR Decoder ist für die restlichen Codebooks verwendet. Hier wird aus diesen Repräsentationen der Codebooks die Wellenform generiert, aus der die Output Sprache entsteht. 

Limitationen und Ethik von VALL-E:
Trotz der herausragenden Ergebnisse von VALL-E gibt es dennoch einige Einschränkungen, die es zu beachten gibt. Einerseits können manche Wörter unklar oder schwer verständlich sein. Darüber hinaus ist die Leistung des Systems bei Sprechern mit Akzent schlechter im Vergleich zu den Sprechern ohne Akzent. Dies liegt an den Trainingsdaten, die zu einem sehr großen Teil aus Hörbuchmaterial bestehen. Auch kann VALL-E die Emotionen der Sprache noch nicht gezielt beeinflussen. Abschließend bestehen ethische Risiken, beispielsweise im Zusammenhang mit Impersonation und Spoofing. Im Zusammenhang mit Deep Fake Videos könnten mit Hilfe von VALL-E falsche Informationen verbreitet werden. Auch könnte VALL-E genutzt werden, um beispielsweise über Telefon an sensible Daten zu gelangen.
Microsoft äußert sich zu diesen möglichen negativen Folgen in ihrem Paper. Es wird eine Möglichkeit genannt, ein System zu erstellen, welches klassifizieren kann, ob eine Audio von VALL-E generiert wurde oder nicht. Ein Solches System gibt es zum jetzigen Stand aber noch nicht. 

Trotz der unterschiedlichen Funktionsweise und Anwendung weisen VALL-E und Tacotron 2 einige Gemeinsamkeiten auf. Zum Einen wäre das die Encoder-Decoder-Architektur, welche es ermöglicht, den Input in eine diskrete Repräsentation umzuwandeln und zu verarbeiten und anschließend die Sprachausgabe zu generieren.
Außerdem nutzen beide Systeme die Mel Spektrogramme als intermediäre Repräsentation. In Tacotron 2 kommt diese direkt im System zum Einsatz, bei VALL-E jedoch nur indirekt, im Encodec Encoder. 
Zudem nutzen beide Systeme auch autoregressive Technologien. Hier wird die Sprachausgabe Schritt für Schritt generiert, wodurch natürliche und flüssige Sprache erzeugt werden kann. 
Diese Merkmale tragen dazu bei, dass beide Systeme qualitativ hochwertige Ergebnisse liefern können und somit zur State-of-the-Art gehören oder sogar übertreffen.

## Methoden

## Anwendungen
Die Anwendungsmöglichkeiten für Text-to-Speech Systeme sind vielseitig. Sie haben schon lange ihren Weg in unseren Alltag gefunden.
Zu häufigen Anwendungen derartiger Systeme gehören unter anderen:
<ul>
<li>Navigationsgeräte</li>
<li>Automatische Ansagen in Zügen oder an Bahnhöfen</li>
<li>Sprachassistenten wie beispielsweise Siri oder Alexa</li>
<li>Barrierefreiheitsfunktionen wie beispielsweise das Vorlesen dessen, was auf dem Bildschirm angezeigt wird.</li>
<li>Vorlesen der Text Ein- oder Ausgabe bei Übersetzungssoftware</li>
</ul>

Beispiele für Anwendungen bei denen Text-to-Speech bereits eingesetzt wird oder ein zukünftiger Einsatz denkbar wäre:
<ul>
<li>Nachbearbeitung von Podcasts, um einzelne Wörter zu ändern</li>
<li>Aufnahme von Podcasts oder Hörbüchern</li>
<li>Erstellen von Tonaufnahmen in unterschiedlichen Sprachen für Videospiele, Filme und Serien</li>
</ul>

Gerade mit dem Blick auf Zero-Shot-Systeme sollte man die Einsatzgebiete nicht außer Acht lassen, die nicht dem Wohle des Großteils der Bevölkerung dienen. Dabei werden Tonspuren oder Videos erstellt, um gezielt Personen, Unternehmen oder Ländern zu schaden. Beispielsweise, um Berichte oder Aussagen zu erstellen, die Wahlen oder den Aktienkurs in eine bestimmte Richtung drängen sollen.

## Fazit
Text-to-Speech ist eine faszinierende und vielversprechende Technologie, mit großem Potential und vielseitigen Anwendungsmöglichkeiten. Die Entwicklung der letzten Jahre hat inzwischen Systeme zur Generierung nahezu natürlicher menschlicher Sprache hervorgebracht.

Die Einsatzgebiete sind dabei sehr vielseitig. Von dem Vorlesen von Texten auf dem Bildschirm für Sehbehinderte Personen, über die Unterstützung zum lernen von Sprachen, bis hin zur sicheren Verwendung von Navigationsgeräten. Darüber hinaus sind die Anwendungsmöglichkeiten in der Werbebranche, dem Kundenservice oder der Unterhaltungsindustrie nahezu grenzenlos.

Allerdings gibt es auch einige Herausforderungen im Zusammenhang mit TTS. Die Generierung von natürlichen Stimmen erfordert eine komplexe Verarbeitung von Sprache, Intonation und Betonung. Obwohl TTS-Systeme bereits erstaunlich realistische Ergebnisse erzielen können, gibt es immer noch Raum für Verbesserungen, insbesondere in Bezug auf die emotionale Ausdrucksstärke und die Anpassungsfähigkeit an unterschiedliche Textarten.

Darüber hinaus sollten ethische Aspekte bei der Entwicklung und Anwendung von TTS-Technologien berücksichtigt werden. Insbesondere der potenzielle Missbrauch von TTS für Fälschungen oder Manipulationen von Audioinhalten ist ein ernstzunehmendes Risiko. Es ist wichtig, Richtlinien und Standards zu entwickeln, um die Verbreitung von gefälschten oder irreführenden Stimmen zu verhindern und die Integrität von Audioquellen zu gewährleisten. Dabei sollte nicht außer Acht gelassen werden, dass die Risiken sowohl auf gesellschaftlicher als auch privater Ebene bestehen. Besonderes Augenmerk sollte man dabei auch auf die Zero-Shot Systeme legen.

Insgesamt bietet TTS enorme Vorteile und Chancen, aber auch Herausforderungen und ethische Überlegungen. Die weitere Forschung und Entwicklung auf diesem Gebiet sind von großer Bedeutung, um die Qualität der generierten Stimmen zu verbessern, neue Anwendungsbereiche zu erschließen und sicherzustellen, dass TTS-Technologien verantwortungsbewusst eingesetzt werden. Mit den richtigen Anstrengungen und Maßnahmen kann Text-to-Speech dazu beitragen, die Kommunikation und den Zugang zu Informationen für Menschen weltweit zu verbessern.

## Weiterführendes Material

### Podcast
Hier Link zum Podcast.

### Talk
Hier einfach Youtube oder THD System embedden.

### Demo
Hier Link zum Demo Video + Link zum GIT Repository mit dem Demo Code.


## Literaturliste
1. https://ieeexplore.ieee.org/abstract/document/10057419
   T. Yanagita, S. Sakti, und S. Nakamura, „Japanese Neural Incremental Text-to-Speech Synthesis Framework With an Accent Phrase Input“, IEEE Access, Bd. 11, S. 22355–22363, 2023, doi: 10.1109/ACCESS.2023.3251657.

2. https://vall-e.io/
   C. Wang u. a., „Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers“. arXiv, 5. Januar 2023. Zugegriffen: 1. Mai 2023. [Online]. Verfügbar unter: http://arxiv.org/abs/2301.02111

3. https://www.researchgate.net/profile/Hazem-El-Bakry/publication/228673642_An_overview_of_text-to-speech_synthesis_techniques/links/553fa8270cf2320416eb23ed/An-overview-of-text-to-speech-synthesis-techniques.pdf
   M. Rashad, H. El-Bakry, R. Isma, und N. Mastorakis, „An overview of text-to-speech synthesis techniques“, International Conference on Communications and Information Technology - Proceedings, Juli 2010.

4. J. Shen u. a., „Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions“. arXiv, 15. Februar 2018. doi: 10.48550/arXiv.1712.05884.

5. „Audio samples from ‚Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions‘“. https://google.github.io/tacotron/publications/tacotron2/ (zugegriffen 4. Juli 2023).

6. „Tacotron 2: Generating Human-like Speech from Text“, 19. Dezember 2017. https://ai.googleblog.com/2017/12/tacotron-2-generating-human-like-speech.html (zugegriffen 4. Juli 2023).
   
