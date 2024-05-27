import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

public class ReconhecimentoFacial {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat imagem = Imgcodecs.imread("images/rostos5.jpg");
        
        DetectarESalvar(imagem);
    }

	private static void DetectarESalvar(Mat imagem) {
		
		MatOfRect rostos = new MatOfRect();
		
		Mat grayScale = new Mat();
		Imgproc.cvtColor(imagem, grayScale, Imgproc.COLOR_BGR2GRAY);
		
		Imgproc.equalizeHist(grayScale, grayScale);
		
		int height = grayScale.height();
		int tamanhoRosto = 0;
		if(Math.round(height * 0.2f) > 0) {
			tamanhoRosto = Math.round(height *0.2f);
		}
		
		CascadeClassifier faceCascade = new CascadeClassifier();
		
		faceCascade.load("data/haarcascade_frontalface_alt2.xml");
		faceCascade.detectMultiScale(grayScale, rostos, 1.1, 2, 0|Objdetect.CASCADE_SCALE_IMAGE, new Size(tamanhoRosto,tamanhoRosto), new Size());
		
		Rect[] rostoArray = rostos.toArray();
		for(int i = 0; i < rostoArray.length; i++) {
			Imgproc.rectangle(imagem, rostoArray[i], new Scalar(0,0,255), 3);
		}
		
		Imgcodecs.imwrite("images/resultado.jpg", imagem);
		
		System.out.println(rostoArray.length + " Rostos foram detectados");
	}
}
