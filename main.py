import cv2
import numpy as np
import time

class ShapeDetection:

    def check(self):
        pass

    def __init__(self):
        self.b = 255
        self.g = 0
        self.r = 0

    def imgRead(self, img, imgContour, b=None, g=None, r=None):

        "resim üzerine çizdireceğimiz kare daire vb. şekillerin çizgi renklerini ayarlayabiliriz."
        if b != None or g != None or r != None:
            pass
        self.img = img
        self.imgContour = imgContour

        imgGray = self.imgGrayF(img)
        imgBlur = self.imgGaussianBlurF(imgGray)
        imgCanny = self.imgCanny(imgBlur)

        shape = self.getContours(imgCanny)

        return shape

    def imgGrayF(self, img):
        """
        Resmimizi Gri tonuna getirmemizi sağlar
        """
        self.imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.imgGray

    def imgGaussianBlurF(self, imgGray):
        """
        Gelen resmi Blur haline getirerek görünmez bir hale sokabilir
        GaussianBlur fonksiyonunu google dan araştırarak resimlere bakınız.
        """
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        return imgBlur

    def imgMedianBlurF(self, imgGray):
        """
        Gelen Blur resmini daha görülebilir hale getirerek resimde seçme yapmamızı
        sağlar
        """
        imgBlur = cv2.medianBlur(imgGray, 15)
        return imgBlur

    def imgCanny(self, imgBlur):
        """
        Resmimiz de kenar veyahut keskin hatlarını bularak göstermemize yarar bir resim
        elde etmemizi sağlar...
        """
        #Canny Trackbar
        canny1 = cv2.getTrackbarPos('Canny 1', 'Canny Trackbar')
        canny2 = cv2.getTrackbarPos('Canny 2', 'Canny Trackbar')

        imgCanny = cv2.Canny(imgBlur, canny1, canny2)
        return imgCanny

    def getContours(self, imgCanny):
        """
        getContours fonksiyonu bizden Canny alarak kenarları tespit edilen resim üzerinden
        findContours fonksiyonu cisimin kenarlarını bularak bize geri dönzerir
        """
        cv2.circle(self.imgContour, (int(self.imgContour.shape[1] / 2), int(self.imgContour.shape[0] / 2)), 1,
                   (255, 0, 0), 5)
        cv2.line(self.imgContour, (int(self.imgContour.shape[1] / 2) - 50, int(self.imgContour.shape[0] / 2)),
                 (int(self.imgContour.shape[1] / 2) + 50, int(self.imgContour.shape[0] / 2)), (0, 255, 0), 1)
        cv2.line(self.imgContour, (int(self.imgContour.shape[1] / 2), int(self.imgContour.shape[0] / 2) - 25),
                 (int(self.imgContour.shape[1] / 2), int(self.imgContour.shape[0] / 2) + 25), (0, 255, 0), 1)
        say = 0
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:

            "contours bize list olarak döndükten sonra contourArea sayesinde pixellerinin yerlerini alıyoruz"
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)

            if area > 2000:  # 2000 pixellerin altındakiler dikkate alınmaz
                say = say + 1
                "contourArea sayesinde oluşan pixelleri drawContours sayesinde cismin etrafını ciziyoruz" \
                "imgContour imgRead fonksiyonu içerisinde img karesi üzerinden kopyalanıyor" \
                "-1, tüm konturları çizmemiz gerektiğini gösterir"
                cv2.drawContours(self.imgContour, cnt, -1, (255, 0, 0), 3)

                "konturumuzun kapalı olduğunu teyit ediyoruz"
                perimeter = cv2.arcLength(cnt, True)

                "Bu yöntem, yaklaşık kontür sayısını bulmak için kullanılır." \
                "approx nesnelerimizin kenar sayısını vermektedir."
                approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
                "cisimlerimizin kenar sayını objCorner içerisine atadık"
                objCorner = len(approx)
                # print("objCorner")
                "Burada nesnenin etrafına çizeceğimiz sınırlayıcı kutumuzun değerlerini elde ederiz."
                x, y, w, h = cv2.boundingRect(approx)
                if M["m00"] == 0.0:
                    cX = int(M["m10"] / 1)
                    cY = int(M["m01"] / 1)

                if M["m00"] != 0.0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                yield cX, cY

                # Yüzdeleme işlemi **BIR GEOMETRIK SEKLE BENZERLIK ORANINI HESAPLAR**
                percent=float(objCorner*10)
                self.mainPercent=percent
                if percent>100:
                    percent=100

                text="%"+str(percent)
                print(text)
                if (percent<10):
                    print("Araç etrafı taramaya devam etsin")
                elif (percent>=10 and percent<20):
                    print("Araç bu veriyi kaydetsin. Etrafı taramaya devam ederek daha iyi bir sonuç arasın.")
                elif (percent >= 20 and percent<30):
                    print("Araç bu veriyi kaydetsin. Etrafı taramaya devam ederek daha iyi bir sonuç arasın.")
                elif (percent >= 30 and percent<40):
                    print("Araç bu veriyi kaydetsin. Etrafı taramaya devam ederek daha iyi bir sonuç arasın.")
                elif (percent >= 40 and percent<50):
                    print("Araç bu veriyi kaydetsin. Etrafı taramaya devam ederek daha iyi bir sonuç arasın.")
                elif (percent >= 50 and percent<60):
                    print("Araç bu veriyi kaydetsin. Etrafı taramaya devam ederek daha iyi bir sonuç arasın.")
                elif (percent >= 60 and percent<70):
                    print("Araç bu veriye doğru yaklaşıp tekrar görüntüyü işlesin. (Eğer bulamazsa etrafı tarasın. İyi bir sonuç bulamazsa biraz daha yaklaşsın.)")
                elif (percent >= 70 and percent<80):
                    print("Araç bu veriye doğru yaklaşıp tekrar görüntüyü işlesin. (Eğer bulamazsa etrafı tarasın. İyi bir sonuç bulamazsa biraz daha yaklaşsın.)")
                elif (percent >= 80 and percent<90):
                    print("Araç bu veriye doğru yönelsin ve her 3 saniyede 1 kez görüntüyü işlesin. Kalenin yüzdelik değerinde düşüş yoksa devam etsin.")
                elif (percent >= 90 and percent<100):
                    print("Araç bu veriye doğru hızlı ve sabit bir şekilde yönelsin ve 2.5 saniyede 1 kez görüntüyü işlesin. Kaleye yaklaştıkça kaleyi ortalasın.")
                elif (percent >= 100):
                    print("Araç kaleyi uzaktan ortalayıp hızlı ve sabit bir şekilde yönelsin ve her 2.5 saniyede 1 görüntüyü işlesin. Kaleye yaklaştıkça kaleyi ortalasın.")
                else:
                    print("Bir hata ile karşılaşıldı")

                #Kaleye uzaklık
                if (percent>50):
                    if (w>0 and w<100):
                        print("Kaleye çok uzaksın")
                    elif (w>=100 and w<200):
                        print("Kaleye uzaksın")
                    elif (w>=200 and w <300):
                        print("Kaleye orta uzaklıktasın")
                    elif (w>=300 and w<400):
                        print("Kaleye yakınsın")
                    elif (w>=400 and w<500):
                        print("Kaleye çok yakınsın")
                    elif (w>=500):
                        print("Kaleye aşırı yakınsın")
                    else:
                        print("Bir hata ile karşılaşıldı.")
                else:
                    print("Kale ile Sualp arası uzaklık yalnızca kale benzerlik oranı %50'nin üzerindeyken kesin yapılabilir.")

                #Yazı renkleri
                textR=0
                textG=0
                textB=0
                if percent >= 0 and percent < 25:
                    textR=255
                    textG=0
                    textB=65
                elif percent >=25 and percent < 50:
                    textR=245
                    textG=206
                    textB=66
                elif percent>=50 and percent<75:
                    textR=66
                    textG=206
                    textB=245
                elif percent>=75 and percent<=100:
                    textR=0
                    textG=255
                    textB=65
                else:
                    textR, textG, textB = 0, 0, 0

                # Kale ve yüzdelemeyi gösterir
                objectType = "Kale "+text
                cv2.rectangle(self.imgContour, (x, y), (x + w, y + h), (0, 0, 255),
                              2)  # Şeklin çevresine diktörgen çizer
                cv2.putText(self.imgContour, objectType, (x + (w // 2) - 10, y + (h // 2) - 10),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (textR, textG, textB), 1)  # Şeklin ortasına yazı ekler

    def objectDetection(self, imgRed, frame):
        (contours, hierarchy) = cv2.findContours(imgRed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
                yield img, x, y

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        print("çıktı")

if __name__ == '__main__':
    print(cv2.__version__)
    cam = cv2.VideoCapture(
        "C:\\Users\\Burak\\Desktop\\Video\\test.mp4")
    cam.set(3, 1376)
    cam.set(3, 776)
    shape = ShapeDetection()

    #Trackbar

    #Renk tanıma
    cv2.namedWindow('Renk Trackbar')
    cv2.resizeWindow('Renk Trackbar', 800, 350)

    # Renk tanıma trackbar
    cv2.createTrackbar('Min H', 'Renk Trackbar', 38, 255, ShapeDetection.check)
    cv2.createTrackbar('Min S', 'Renk Trackbar', 100, 255, ShapeDetection.check)
    cv2.createTrackbar('Min V', 'Renk Trackbar', 100, 255, ShapeDetection.check)

    cv2.createTrackbar('Max H', 'Renk Trackbar', 75, 255, ShapeDetection.check)
    cv2.createTrackbar('Max S', 'Renk Trackbar', 255, 255, ShapeDetection.check)
    cv2.createTrackbar('Max V', 'Renk Trackbar', 255, 255, ShapeDetection.check)

    #Blur, Canny
    cv2.namedWindow('Canny Trackbar')
    cv2.resizeWindow('Canny Trackbar', 800, 350)

    #Canny trackbar
    cv2.createTrackbar('Canny 1', 'Canny Trackbar', 50, 150, ShapeDetection.check)
    cv2.createTrackbar('Canny 2', 'Canny Trackbar', 50, 150, ShapeDetection.check)

    #Contrast, Bright
    cv2.namedWindow('Contrast Bright')
    cv2.resizeWindow('Contrast Bright', 800, 350)

    #Contrast, Bright
    cv2.createTrackbar('Alpha', 'Contrast Bright', 1, 5, ShapeDetection.check)
    cv2.createTrackbar('Beta', 'Contrast Bright', 10, 50, ShapeDetection.check)

    while True:
        ret, img = cam.read()
        frame = img.copy()

        #Renk Tanıma Trackbar
        minH = cv2.getTrackbarPos('Min H', 'Renk Trackbar')
        minS = cv2.getTrackbarPos('Min S', 'Renk Trackbar')
        minV = cv2.getTrackbarPos('Min V', 'Renk Trackbar')

        maxH = cv2.getTrackbarPos('Max H', 'Renk Trackbar')
        maxS = cv2.getTrackbarPos('Max S', 'Renk Trackbar')
        maxV = cv2.getTrackbarPos('Max V', 'Renk Trackbar')

        alpha = cv2.getTrackbarPos('Alpha', 'Contrast Bright')
        beta = cv2.getTrackbarPos('Beta', 'Contrast Bright')

        new_image = cv2.addWeighted(frame, alpha, np.zeros(frame.shape, frame.dtype), 0, beta)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_grean = np.array([minH, minS, minV])  # Sualtında ki kale
        upper_grean = np.array([maxH, maxS, maxV])  # Sualtında ki kale
        mask = cv2.inRange(hsv, lower_grean, upper_grean)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('mask', mask)

        h, w, _ = img.shape
        h = int(h / 2)
        w = int(w / 2)
        _, imgContour = cam.read()
        _, img0 = cam.read()
        data = None
        shapeGoal = shape.imgRead(img=res, imgContour=imgContour)
        object = False
        x, y = None, None

        checkStr, checkStr2 = "", ""

        for cX, cY in shapeGoal:
            if cX > w:
                checkStr="Sağ"
            elif cX < w:
                checkStr="Sol"
            else:
                checkStr="Merkez"

            if cY > h:
                checkStr2="Alt"
            elif cY < h:
                checkStr2="Üst"
            else:
                checkStr2="Merkez"

            print("****************************************************************************************************************************************************************")
            text="cx : "+ str(cX)+ " cy : "+ str(cY)+ " Kale Nişangahın "+ str(checkStr)+ " Tarafında, Kale Nişangahın "+ str(checkStr2)+" tarafında kalmaktadır."
            print(text)

        #Görüntüyü yavaşlatma sebebim rahat çalışma alanı oluşturmak.
        time.sleep(0.1)
        cv2.imshow("Resim", imgContour)
        cv2.imshow("Kontrast", new_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            break
