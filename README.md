# Detekce, extrakce a měření délky záprsní kostí ze snímků

* src
    * Mask_RCNN - Implementace [MaskRCNN](https://github.com/matterport/Mask_RCNN)
    * detect_MaskRCNN.py - Zpracování datasetu
    * train_MaskRCNN.py - Trénování MaskRCNN
    * process_output.py - Zpracování výstupu z detect_MaskRCNN.py
* data
    * mask_rcnn_coco.h5 - Váhy pro trénování MaskRCNN
    * mask_rcnn_mc.h5 - Váhy MaskRCNN natrénované na datasetu
    * dataset - Snímky kostí pro zpracování
    * train_data - Trénovací a validační dataset s anotacemi
* output
    * lengths_contours - Výstupy z process_output.py
        * Json klíče:
            * Image - název souboru s obrázkem kosti
            * TopPoint - Koordináty vrchního bodu
            * BottomPoint - Koordináty spodního bodu 
            * Length - Vzdálenost mezi koordináty
            * LengthUnit - Jednotka vzdálenosti
            * Contour - Body kontury kosti
        * TPS
            * Obsahuje pouze název snímků a body kontury
        * Soubory jsou pojmenovány podle sady a umístění kosti
    * masks - Výstupy z detect_MaskRCNN.py
    
### Nejdříve je potřeba nainstalovat závislosti pro MaskRCNN.
pip install -r src/Mask_RCNN/requirements.txt\
pip install imgaug\
pip install h5py==2.10.0\
python setup.py install\

### Trénování MaskRCNN
Trénování bylo prováděno pomocí jupyter notebooku src/bc_train.ipynb na stránce https://www.colab.research.google.com.
Do instance je potřeba nahrát MaskRCNN v adresáři src. A dále složku train_data a váhy data/mask_rcnn_coco.h5.

### Segmentace snímků
Segmentace byla prováděna pomocí jupyter notebooku src/bc_detect.ipynb na stránce https://www.colab.research.google.com.
Do instance je potřeba nahrát soubor  data/mask_rcnn_mc.h5, dataset obrázků a MaskRCNN v adresáři src.

### Zpracování výstupů z detect_MaskRCNN.py
./process_output.py -d ../output/masks/ -i /data/public/Bakalářka/dataset/ -o ../output/lengths_contours/