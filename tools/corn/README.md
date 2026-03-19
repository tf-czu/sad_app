### Fáze 1

#### Prerekvizity

* Je potřeba mít složku "videos" a v ní soubor "corn.mp4".

#### Spuštění

* `python3 anotator.py`

#### Současné limitace

* Po kliknutí je na framu zobrazen pouze poslední kliknutý bod.
* Při překlikávání mezi framy se zobrazený bod nezachovává - slouží pouze jako vizuální zpětná vazba, že jsme skutečně v danou chvíli na dané místo klikli.
* Soubor "clicks.csv" může teoreticky obsahovat více kliknutí, než kolik může být uložených snímků ve složce "anotated-frames" - a to ve chvíli, kdy na nějaký frame klikneme vícekrát (do "clicks.csv" se propisují všechny kliky, zatímco v rámci "anotated-frames" složky jsou evidovány pouze poslední "finální" kliky). Dojde-li ke kliku na jednom framu opakovaně, může být následně potřeba duplikáty manuálně odstranit.
* Ukládané soubory se momentálně při spuštění kompletně přepisují od počátku.

### Fáze 2

#### Prerekvizity

* Je potřeba mít složku "videos" a v ní soubor "calibration.MOV".
* Je potřeba mít složku "results" a v ní soubor "clicks-alltogether-noduplicates.csv".

#### Spuštění

* `python3 camera-calibration.py`
* `python3 recalculator.py`

#### Současné limitace

* Výsledky kalibrace jsou vyprintěny do terminálu a ohnisková vzdálenost musí být následně manuálně vložena do "recalculator.py".
