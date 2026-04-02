#### Fáze 1

* Načíst video do OpenCV -> bude zobrazovat snímky (obrázky), já budu klikat myší, jakou chci vrátit pozici
* Provést anotaci snímků - označit si bod, kde se kytka zapíchává do půdy (mělo by jít přímo v OpenCV)
* Provést alespoň pro 700 snímků (přibližně rovnoměrně rozděleno průběhem videa)
* Výstup - CSV se dvěma údaji:
* &nbsp;	Buď čas, kterému odpovídá NEBO alespoň číslo snímku
* &nbsp;	Odchýlení na X-ové ose (odchýlení od středu kamery - ve směru osy X)
* Výstup do "sab\_app/tools"
* K datům - není stoprocentní, že je pevná snímkovací frekvence, přibližně 30 snímků za sekundu



#### Fáze 2

* Na základě kalibračního videa přepočítat odchylku na vzdálenost v metrech
* Mohla by pomoct AI - "Připrav mi skript na kalibraci kamery"
* Čtverečky na kalibraci mají rozměr 100mm
* X = (h\*a)/f
* &nbsp;	X ... hledaná vzdálenost (přepočtená odchylka v m)
* &nbsp;	a ... v pixelech (z fáze 1)
* &nbsp;	h ... zřejmě 0.5m (nad natáčeným povrchem)
* &nbsp;	f ... zjistíme z kalibračního skriptu



---



Alternativně by se dalo vyřešit tak, že bychom:

* Oanotovali 200-300 snímků (bounding box by mohl být čtvereček)
* Natrénovali model (šlo by využít např. RoboFlow) - šel by znovupoužívat
* Z modelu získali obdobná data jako v případě fáze 1 a následně aplikovali fázi 2 jako v základním použitém řešení

(Alternativní řešení se může hodit později, pokud by se ukázalo, že bychom chtěli postup opakovat.)
