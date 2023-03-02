import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

class LiczAuta:
    def __init__(self):
        #styl tekstu na klatkach wynikowych
        self.red = (0,0,255)
        self.blue = (255,0,0)
        self.black = (0,0,0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontSize = 2
        self.thickness = 2

    def center(self, x):
        x1,y1,x2,y2 = x
        return (x1+x2)/2,(y1+y2)/2

    def ktory_pas(self, wspolrzedne_xy_srodka):
        xx,_ = self.center(wspolrzedne_xy_srodka)
        if self.linie_pasow[0]<xx<self.linie_pasow[1]:
            return 0
        elif self.linie_pasow[1]<xx<self.linie_pasow[2]:
            return 1
        elif self.linie_pasow[2]<xx<self.linie_pasow[3]:
            return 2
        else: 
            return 3

    def czy_jest_w_obszarze_wykrywania(self,prostokat_pojazdu):
        srodek_x,srodek_y=self.center(prostokat_pojazdu)
        x1,y1,x2,y2 = self.obszar_wykrywania
        if (x1<srodek_x<x2) and (y2<srodek_y<y1):
            return True
        return False

    def inicjalizacja_detectron(self):
        #inicjalizacja modelu DETECTRON2
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)

    #"rtsp://77.91.32.5:6009/ch0_0.h264"
    def przetworz_plik(self, plik_wejsciowy, plik_wynikowy, klatka_start):
        self.cap = cv2.VideoCapture(plik_wejsciowy) 
        #lista właściwości obiektu cv2.VideoCapture(): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        self.szerokosc_klatki = self.cap.get(3)
        self.wysokosc_klatki = self.cap.get(4)
        self.co_ile_klatek_przetwarzac = 4
        self.liczba_wszystkich_klatek = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 4*25
        self.biezaca_klatka = klatka_start
        self.suma_aut_pasy = np.zeros(4)         #suma_aut_pasy = [0.0, 0.0, 0.0, 0.0]
        self.lines_old = np.zeros(4)             #lines_old = [0.0, 0.0, 0.0, 0.0]
        self.objects_old = np.array(['empty']*4) #objects_old = ['empty', 'empty', 'empty', 'empty']
        self.oznaczenia_pojazdow = {2 : 'car', 7 : 'truck', 3 : 'motor', 5: 'bus',  0 : 'motor'}
        self.zliczone_pojazdy = {'car' : 0, 'truck' : 0, 'bus' : 0, 'motor' : 0}
        #obszar wykrywania (prostokąt)
        self.obszar_wykrywania_x1 = 246
        self.obszar_wykrywania_y1 = 285
        self.obszar_wykrywania_x2 = 730
        self.obszar_wykrywania_y2 = 250
        self.obszar_wykrywania = [self.obszar_wykrywania_x1,self.obszar_wykrywania_y1,self.obszar_wykrywania_x2,self.obszar_wykrywania_y2]
        #definicja pionowych linii rozdzielających pasy
        self.linie_pasow = [246,370,500,640,730]

        #inicjalizuj model wykrywania samochodow
        self.inicjalizacja_detectron()

        #plik wynikowy
        self.video_wynikowe = cv2.VideoWriter(plik_wynikowy,cv2.VideoWriter_fourcc(*'mp4v'),25,(int(self.szerokosc_klatki),int(self.wysokosc_klatki)),True)
        output = ''
        self.color = self.blue

        #główna pętla przez klatki (nie wszystkie, co tyle ile parametr )
        while(self.cap.isOpened()):
            #sprawdź czy to nie jest ostatnia klatka
            if self.biezaca_klatka > self.liczba_wszystkich_klatek:
                break
            
            #ustaw numer klatki 
            self.cap.set(1,self.biezaca_klatka-1)
            ret, klatka = self.cap.read()
            
            #sprawdź czy wykrywać auta na tej klatce (zależne od tego co którą klatkę przetwarzać)
            if (self.biezaca_klatka + self.co_ile_klatek_przetwarzac) % self.co_ile_klatek_przetwarzac == 0:
                #wyzeruj zmienne dla klatki
                lines_new = np.zeros(4)
                objects_new = np.array(['empty']*4)
                prostokaty_wykrytych_pojazdow = []
                i = 0

                #wykryj auta i pokaż prostokąty wokół nich
                viz = Visualizer(klatka[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE)
                predictions = self.predictor(klatka)
                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
                
                #przepisz współrżedne prostokątów z tensor do zwykłych tablic (koordynaty w pikselach) i sprawdź czy wsp.x mieszczą się w całkowitym obszarze wykrywania
                for pojazd_wykryty in predictions['instances'].pred_boxes:
                    x1 = int(pojazd_wykryty[0].item())
                    y1 = int(pojazd_wykryty[1].item())
                    x2 = int(pojazd_wykryty[2].item())
                    y2 = int(pojazd_wykryty[3].item())
                    if x1 > self.obszar_wykrywania_x1 and x1 < self.obszar_wykrywania_x2:
                        prostokaty_wykrytych_pojazdow.append([[x1,y1,x2,y2],predictions['instances'].pred_classes[i].item()])
                    i +=1

                #sprawdź który z wykrytych pojazdów jest na którym pasie i jaki to typ
                for pojazd_wykryty in prostokaty_wykrytych_pojazdow:
                    #srodek_x_pojazdu,srodek_y_pojazdu = self.center(pojazd_wykryty[0])
                    if self.czy_jest_w_obszarze_wykrywania(pojazd_wykryty[0]):
                        na_ktorym_pasie_jest = self.ktory_pas(pojazd_wykryty[0])
                        typ_wykrytego_pojazdu = pojazd_wykryty[1]
                        #na którym pasie jest auto (1 - jest, 0 - nie ma)
                        lines_new[na_ktorym_pasie_jest] = 1 
                        #jakiego typu to jest auto
                        objects_new[na_ktorym_pasie_jest] = self.oznaczenia_pojazdow[typ_wykrytego_pojazdu] 
                
                #sprawdź każdy pas osobno
                for i in range(4): 
                    #jeżeli wcześniej nie było auta na pasie a teraz jest to dodaj jedno auto danego typu
                    if (self.lines_old[i] != lines_new[i]) and lines_new[i] == 1:
                        self.suma_aut_pasy[i] +=1
                        self.zliczone_pojazdy[objects_new[i]] += 1 
                    if 1 in lines_new:
                        self.color = self.blue = self.red
                    else:
                        self.color = self.blue = self.blue

                #skopiuj aktualne wartości żeby z nimi porównać zawartość pasów w kolejnej klatce
                self.lines_old = np.copy(lines_new)
                self.objects_old = np.copy(objects_new)
            
            #generowanie prostokąta wykrywania
            x1,y1,x2,y2 = self.obszar_wykrywania
            czarny_prostokat = np.zeros(klatka.shape, np.uint8)
            cv2.rectangle(czarny_prostokat, (x1, y1), (x2, y2), self.blue, cv2.FILLED)
            #nałożenie na klatkę źródłową "klatka" obszaru wykrywania "czany prostokat" utworoznego z zer numpy z przezroczystością 0.25
            klatka = cv2.addWeighted(klatka, 1.0, czarny_prostokat, 0.25, 1)
            #dodanie napisów
            klatka = cv2.putText(klatka,'Suma = ' + str(sum(self.suma_aut_pasy)),(195,100),self.font,self.fontSize,self.black,self.thickness)
            klatka = cv2.putText(klatka,'Pas1=' + str(int(self.suma_aut_pasy[0])),(210,150),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,'Pas2=' + str(int(self.suma_aut_pasy[1])),(350,150),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,'Pas3=' + str(int(self.suma_aut_pasy[2])),(490,150),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,'osobowy',(450,50),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,str(int(self.zliczone_pojazdy['car'])),(600,50),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,'ciezarowy',(450,80),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,str(int(self.zliczone_pojazdy['truck'])),(600,80),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,'motor',(450,110),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,str(int(self.zliczone_pojazdy['motor'])),(600,110),self.font,1,self.black,self.thickness)
            #wypisanie do debugowania
            print("Klatka " + str(self.biezaca_klatka) + "/" + str(self.liczba_wszystkich_klatek) + ": Pas1=" + str(self.suma_aut_pasy[0]) + " Pas2:" + str(self.suma_aut_pasy[1])+ " Pas3:" + str(self.suma_aut_pasy[2])+ " Pas4:" + str(self.suma_aut_pasy[3])) 
            #jeżeli coś jest wykryte to nałóż to na klatkę wynikową
            if output != '': klatka = cv2.addWeighted(klatka, 1.0, output.get_image()[:,:,::-1], 0.3, 1)
            #pokaż aktualną klatkę
            cv2.imshow("Wynik", klatka)
            #sprawdź czy zamknąć program klawiszem q
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            #zwieksz numer klatki
            self.biezaca_klatka += 1
            #zapisz klatkę do wideo wynikoweog
            self.video_wynikowe.write(klatka)
        #zakończenie pliku wynikowego
        self.cap.release()
        self.video_wynikowe.release()
        cv2.destroyAllWindows()

    def przetworz_stream(self, link_do_streamu):
        self.cap = cv2.VideoCapture(link_do_streamu) 
        #lista właściwości obiektu cv2.VideoCapture(): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        self.szerokosc_klatki = self.cap.get(3)
        self.wysokosc_klatki = self.cap.get(4)
        self.suma_aut_pasy = np.zeros(4)         #suma_aut_pasy = [0.0, 0.0, 0.0, 0.0]
        self.lines_old = np.zeros(4)             #lines_old = [0.0, 0.0, 0.0, 0.0]
        self.objects_old = np.array(['empty']*4) #objects_old = ['empty', 'empty', 'empty', 'empty']
        self.oznaczenia_pojazdow = {2 : 'car', 7 : 'truck', 3 : 'motor', 5: 'bus',  0 : 'motor'}
        self.zliczone_pojazdy = {'car' : 0, 'truck' : 0, 'bus' : 0, 'motor' : 0}
        output = ''
        self.color = self.blue
        self.biezaca_klatka = 1
        self.co_ile_klatek_przetwarzac = 1
        #obszar wykrywania (prostokąt)
        self.obszar_wykrywania_x1 = 246
        self.obszar_wykrywania_y1 = 285
        self.obszar_wykrywania_x2 = 730
        self.obszar_wykrywania_y2 = 250
        self.obszar_wykrywania = [self.obszar_wykrywania_x1,self.obszar_wykrywania_y1,self.obszar_wykrywania_x2,self.obszar_wykrywania_y2]
        #definicja pionowych linii rozdzielających pasy
        self.linie_pasow = [246,370,500,640,730]

        #inicjalizuj model wykrywania samochodow
        self.inicjalizacja_detectron()

        #główna pętla przez klatki
        while(self.cap.isOpened()):
            #odczytaj klatkę
            ret, klatka = self.cap.read()
            
            #sprawdź czy wykrywać auta na tej klatce (zależne od tego co którą klatkę przetwarzać)
            if (self.biezaca_klatka + self.co_ile_klatek_przetwarzac) % self.co_ile_klatek_przetwarzac == 0:
                #wyzeruj zmienne dla klatki
                lines_new = np.zeros(4)
                objects_new = np.array(['empty']*4)
                prostokaty_wykrytych_pojazdow = []
                i = 0

                #wykryj auta i pokaż prostokąty wokół nich
                viz = Visualizer(klatka[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE)
                predictions = self.predictor(klatka)
                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
                
                #przepisz współrżedne prostokątów z tensor do zwykłych tablic (koordynaty w pikselach) i sprawdź czy wsp.x mieszczą się w całkowitym obszarze wykrywania
                for pojazd_wykryty in predictions['instances'].pred_boxes:
                    x1 = int(pojazd_wykryty[0].item())
                    y1 = int(pojazd_wykryty[1].item())
                    x2 = int(pojazd_wykryty[2].item())
                    y2 = int(pojazd_wykryty[3].item())
                    if x1 > self.obszar_wykrywania_x1 and x1 < self.obszar_wykrywania_x2:
                        prostokaty_wykrytych_pojazdow.append([[x1,y1,x2,y2],predictions['instances'].pred_classes[i].item()])
                    i +=1

                #sprawdź który z wykrytych pojazdów jest na którym pasie i jaki to typ
                for pojazd_wykryty in prostokaty_wykrytych_pojazdow:
                    #srodek_x_pojazdu,srodek_y_pojazdu = self.center(pojazd_wykryty[0])
                    if self.czy_jest_w_obszarze_wykrywania(pojazd_wykryty[0]):
                        na_ktorym_pasie_jest = self.ktory_pas(pojazd_wykryty[0])
                        typ_wykrytego_pojazdu = pojazd_wykryty[1]
                        #na którym pasie jest auto (1 - jest, 0 - nie ma)
                        lines_new[na_ktorym_pasie_jest] = 1 
                        #jakiego typu to jest auto
                        objects_new[na_ktorym_pasie_jest] = self.oznaczenia_pojazdow[typ_wykrytego_pojazdu] 
                
                #sprawdź każdy pas osobno
                for i in range(4): 
                    #jeżeli wcześniej nie było auta na pasie a teraz jest to dodaj jedno auto danego typu
                    if (self.lines_old[i] != lines_new[i]) and lines_new[i] == 1:
                        self.suma_aut_pasy[i] +=1
                        self.zliczone_pojazdy[objects_new[i]] += 1 
                    if 1 in lines_new:
                        self.color = self.blue = self.red
                    else:
                        self.color = self.blue = self.blue

                #skopiuj aktualne wartości żeby z nimi porównać zawartość pasów w kolejnej klatce
                self.lines_old = np.copy(lines_new)
                self.objects_old = np.copy(objects_new)
            
            #generowanie prostokąta wykrywania
            x1,y1,x2,y2 = self.obszar_wykrywania
            czarny_prostokat = np.zeros(klatka.shape, np.uint8)
            cv2.rectangle(czarny_prostokat, (x1, y1), (x2, y2), self.blue, cv2.FILLED)
            #nałożenie na klatkę źródłową "klatka" obszaru wykrywania "czany prostokat" utworoznego z zer numpy z przezroczystością 0.25
            klatka = cv2.addWeighted(klatka, 1.0, czarny_prostokat, 0.25, 1)
            #dodanie napisów
            klatka = cv2.putText(klatka,'Suma = ' + str(sum(self.suma_aut_pasy)),(195,100),self.font,self.fontSize,self.black,self.thickness)
            klatka = cv2.putText(klatka,'Pas1=' + str(int(self.suma_aut_pasy[0])),(210,150),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,'Pas2=' + str(int(self.suma_aut_pasy[1])),(350,150),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,'Pas3=' + str(int(self.suma_aut_pasy[2])),(490,150),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,'osobowy',(450,50),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,str(int(self.zliczone_pojazdy['car'])),(600,50),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,'ciezarowy',(450,80),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,str(int(self.zliczone_pojazdy['truck'])),(600,80),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,'motor',(450,110),self.font,1,self.black,self.thickness)
            klatka = cv2.putText(klatka,str(int(self.zliczone_pojazdy['motor'])),(600,110),self.font,1,self.black,self.thickness)
            #wypisanie do debugowania
            print("Klatka " + str(self.biezaca_klatka) + "/" + str(self.liczba_wszystkich_klatek) + ": Pas1=" + str(self.suma_aut_pasy[0]) + " Pas2:" + str(self.suma_aut_pasy[1])+ " Pas3:" + str(self.suma_aut_pasy[2])+ " Pas4:" + str(self.suma_aut_pasy[3])) 
            #jeżeli coś jest wykryte to nałóż to na klatkę wynikową
            if output != '': klatka = cv2.addWeighted(klatka, 1.0, output.get_image()[:,:,::-1], 0.3, 1)
            #pokaż aktualną klatkę
            cv2.imshow("Wynik", klatka)
            #sprawdź czy zamknąć program klawiszem q
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            #zwieksz numer klatki
            self.biezaca_klatka += 1
            #zapisz klatkę do wideo wynikoweog
            self.video_wynikowe.write(klatka)
        #zakończenie pliku wynikowego
        self.cap.release()
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    licz_auta = LiczAuta()
    licz_auta.przetworz_plik("film.mp4","wyjsciowy.mp4",53)