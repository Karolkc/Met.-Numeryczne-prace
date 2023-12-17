W repozytorium bazowo znajdują się 2 pliki: without_hidden.py oraz with_1_hidden.py  
	
Są to "ręcznie" napisane sieci neuronowe, odpowiednio bez i z warstwą ukrytą.

W przypadku sieci bez warstw ukrytych można łatwo przejść do postaci liniowej modelu regresji pobierając z modelu ostatnią obliczoną wagę oraz bias.

Dla sieci z warstwą ukrytą nie ma to większego sensu, ponieważ każdy neuron w tej warstwie jest funkcją, a na końcowe predykcje składa się kombinacja tych funkcji.

Plik forced_degree.py to po prostu wersja regresji z użyciem modelu bez warstw ukrytych, w którym mamy więcej niż 1 cechę. Cechy dodatkowe tworzonę są przez skalowanie wartości x do żądanej potęgi, aby uzyskane wagi były współczynnikami wielomianu n-stopnia. Skalowanie jest robione ręcznie :(
